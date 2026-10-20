"""LoRA memory/throughput smoke for Qwen3.5-4B at 8K/32K/64K.

Per docs/工程问题/2026-10-21-原契约SWE-SFT正式生成方案-v2-64K与增量抓取.md §2.2:
BF16 base, LoRA r32/alpha64 on all linear projections, gradient checkpointing,
micro-batch 1, synthetic sequence with a 72/28 context/supervised label split
(matching the measured 38% supervision share order of magnitude).

Records peak CUDA memory, step latency and tokens/s for each length, then writes
results/smoke/summary.json and a short markdown table. No real data is trained on.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = Path(r"E:\Deeplearning\local_train\model\Qwen3.5-4B")
OUT = Path(__file__).resolve().parents[1] / "results" / "smoke"
LENGTHS = [8192, 32768, 65536]
STEPS = 3
CONTEXT_SHARE = 0.72  # loss=0 prefix; remainder supervised (measured 38.1%)


def build_batch(tokenizer, length: int, device: str):
    vocab = len(tokenizer)
    ids = torch.randint(10, vocab - 10, (1, length), device=device)
    labels = ids.clone()
    cut = int(length * CONTEXT_SHARE)
    labels[:, :cut] = -100
    return {"input_ids": ids, "labels": labels}


def run_length(model, tokenizer, length: int, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = build_batch(tokenizer, length, device)
    # warmup forward (no backward) to touch cuda graphs/kernels
    with torch.no_grad():
        _ = model(input_ids=batch["input_ids"]).logits
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for _ in range(STEPS):
        out = model(**batch)
        out.loss.backward()
        losses.append(out.loss.item())
        for p in filter(lambda x: x.requires_grad, model.parameters()):
            if p.grad is not None:
                p.grad = None  # manual SGD-free update loop; optimizer omitted on purpose
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    return {
        "length": length,
        "steps": STEPS,
        "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
        "peak_alloc_GiB": round(peak, 2),
        "peak_reserved_GiB": round(reserved, 2),
        "total_s": round(elapsed, 1),
        "tokens_per_s": round(length * STEPS / elapsed, 0),
        "s_per_step": round(elapsed / STEPS, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    args = parser.parse_args()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
    except TypeError:  # transformers < 5 uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)", flush=True)
    results = []
    for length in args.lengths:
        try:
            r = run_length(model, tokenizer, length, device)
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM at {length}: {str(e)[:160]}", flush=True)
            results.append({"length": length, "oom": True})
            break  # longer lengths will not fit either
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_GiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "precision": "bf16-base + LoRA r32/alpha64, grad-checkpoint, mbs1",
        "context_share_loss0": CONTEXT_SHARE,
        "trainable_M": round(trainable / 1e6, 1),
        "results": results,
        "window_65536_ok": any(r.get("length") == 65536 and not r.get("oom") for r in results),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    lines = ["| 长度 | 峰值分配 GiB | 峰值保留 GiB | tokens/s | s/step |", "|---:|---:|---:|---:|---:|"]
    for r in results:
        if r.get("oom"):
            lines.append(f"| {r['length']} | OOM | | | |")
        else:
            lines.append(f"| {r['length']} | {r['peak_alloc_GiB']} | {r['peak_reserved_GiB']} | "
                         f"{r['tokens_per_s']:.0f} | {r['s_per_step']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary ->", OUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
