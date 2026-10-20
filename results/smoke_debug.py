"""One instrumented 2K step: record memory history and print the top allocations."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = Path(r"E:\Deeplearning\local_train\model\Qwen3.5-4B")
L = 2048

torch.cuda.memory._record_memory_history(max_entries=100000)
model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
backend = model.base_model.model.model  # headless Qwen3_5TextModel (verified by probe)
head = model.get_output_embeddings()

ids = torch.randint(10, 1000, (1, L), device="cuda")
labels = ids.clone(); labels[:, : int(L * 0.72)] = -100

try:
    hidden = backend(ids).last_hidden_state
    loss_sum = torch.zeros((), device="cuda"); n_tot = torch.zeros((), device="cuda")
    for i in range(0, L, 1024):
        h, l = hidden[:, i:i+1024], labels[:, i:i+1024]
        logits = head(h).float()
        mask = l != -100
        logp = torch.log_softmax(logits, -1)
        tok = torch.gather(logp, 2, l.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
        loss_sum = loss_sum + (tok * mask).sum(); n_tot = n_tot + mask.sum()
    (-(loss_sum / n_tot)).backward()
    torch.cuda.synchronize()
    print("2K step OK, peak GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))
except torch.cuda.OutOfMemoryError:
    print("OOM — dumping memory summary")
    print(torch.cuda.memory_summary(abbreviated=True)[:6000])
finally:
    torch.cuda.memory._record_memory_history(enabled=None)
