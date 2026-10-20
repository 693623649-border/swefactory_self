"""LoRA smoke v2: chunked cross-entropy to avoid materializing full logits.

Qwen3.5-4B is a hybrid (24/32 linear_attention layers, vocab 248,320, mamba_ssm_dtype
fp32). A plain forward-with-labels allocates ~4 GiB of bf16 logits at 8K already and
upcasts to fp32, which OOMs the 31.8 GiB 5090. This smoke follows the training plan's
anticipated remedy: take hidden states, then compute lm_head + CE in sequence chunks
wrapped in torch.utils.checkpoint so backward recomputes chunk logits instead of
storing them.

Also probes whether the reference (non-fla) gated-delta-rule path is the remaining
memory driver: reports peak memory with chunked CE at 2K/8K/32K/64K.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.utils.checkpoint as cp
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = Path(r"E:\Deeplearning\local_train\model\Qwen3.5-4B")
OUT = Path(__file__).resolve().parents[1] / "results" / "smoke"
LENGTHS = [2048, 8192, 32768, 65536]
STEPS = 3
CONTEXT_SHARE = 0.72
CHUNK = 1024


def chunk_loss(lm_head, hidden_chunk: torch.Tensor, labels_chunk: torch.Tensor) -> torch.Tensor:
    logits = lm_head(hidden_chunk)  # [1, c, V] bf16, transient
    logits = logits.float()
    labels = labels_chunk.clone()
    mask = labels != -100
    labels = labels.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits, dim=-1)
    tok = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)
    return ((tok * mask).sum(), mask.sum())


def run_length(backend, tokenizer, length: int, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vocab = len(tokenizer)
    ids = torch.randint(10, vocab - 10, (1, length), device=device)
    labels = ids.clone()
    labels[:, : int(length * CONTEXT_SHARE)] = -100
    sup_total = int((labels != -100).sum())

    start = time.perf_counter()
    losses = []
    for _ in range(STEPS):
        hidden = backend(ids).last_hidden_state  # [1, L, H]
        acc, cnt = 0.0, None
        for i in range(0, length, CHUNK):
            h = hidden[:, i : i + CHUNK]
            l = labels[:, i : i + CHUNK]
            s, n = cp.checkpoint(chunk_loss, backend_head, h, l, use_reentrant=False)
            acc = acc + s
            cnt = n if cnt is None else cnt + n
        loss = -(acc / cnt)
        loss.backward()
        losses.append(loss.item())
        for p in filter(lambda x: x.requires_grad, model.parameters()):
            if p.grad is not None:
                p.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "length": length,
        "steps": STEPS,
        "supervised_tokens": sup_total,
        "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
        "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "total_s": round(elapsed, 1),
        "tokens_per_s": round(length * STEPS / elapsed, 0),
        "s_per_step": round(elapsed / STEPS, 2),
    }


model = None
backend_head = None


def main() -> None:
    global model, backend_head
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flex_attention"])
    args = parser.parse_args()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), dtype=torch.bfloat16, attn_implementation=args.attn)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model = model.to(device)
    model.config.use_cache = False
    base = model.model if hasattr(model, "model") else model
    global backend_head
    backend_head = model.get_output_embeddings()
    if backend_head is None:
        raise RuntimeError("no output embedding/lm_head found")
    backend = base
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()  # GradientCheckpointingLayer only checkpoints when self.training is True
    # Single-sample packing without padding: the causal 4-D additive mask forces SDPA
    # off the flash backend (math materializes [B,H,L,L]). With mask=None SDPA uses
    # is_causal=True, which is mathematically identical here.
    import transformers.models.qwen3_5.modeling_qwen3_5 as _q35
    _q35.create_causal_mask = lambda **kw: None
    _q35.create_recurrent_attention_mask = lambda **kw: None
    # peft wraps modules; re-resolve head/backend against the wrapped model
    backend_head = model.get_output_embeddings()
    backend = model.base_model.model.model if hasattr(model.base_model, "model") else backend
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable/1e6:.1f}M", flush=True)
    results = []
    for length in args.lengths:
        try:
            r = run_length(backend, tokenizer, length, device)
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM at {length}: {str(e)[:200]}", flush=True)
            results.append({"length": length, "oom": True})
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_GiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "precision": "bf16 + LoRA r32/a64, grad-ckpt, chunked-CE(chunk=1024, checkpointed)",
        "architecture_note": "hybrid: 24/32 linear_attention, vocab 248320, mamba fp32 states",
        "results": results,
        "window_65536_ok": any(r.get("length") == 65536 and not r.get("oom") for r in results),
    }
    (OUT / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("summary ->", OUT / "summary_v2.json", flush=True)


if __name__ == "__main__":
    main()
