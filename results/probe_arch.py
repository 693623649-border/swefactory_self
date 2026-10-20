"""Probe Qwen3.5-4B module tree on CPU to locate the headless text backbone."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoConfig

MODEL = Path(r"E:\Deeplearning\local_train\model\Qwen3.5-4B")
cfg = AutoConfig.from_pretrained(str(MODEL))
print("config class:", type(cfg).__name__)
with torch.device("meta"):
    try:
        model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:
        print("from_config failed:", e)
        raise SystemExit(1)

print("top:", type(model).__name__)
for name, child in model.named_children():
    print(f"  {name}: {type(child).__name__}")
    for n2, c2 in child.named_children():
        print(f"    {name}.{n2}: {type(c2).__name__}")

inner = model.model
for attr in ("model", "language_model", "text_model"):
    if hasattr(inner, attr):
        sub = getattr(inner, attr)
        print(f"inner.{attr}: {type(sub).__name__}")
print("output_embeddings:", type(model.get_output_embeddings()).__name__ if model.get_output_embeddings() else None)

# tiny real forward on cpu for shape check
del model
real = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.bfloat16)
real.config.use_cache = False
ids = torch.randint(10, 1000, (1, 64))
with torch.no_grad():
    out = real.model(ids)
print("real.model output type:", type(out).__name__, "keys:", list(out.keys()) if hasattr(out, "keys") else "-")
last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
print("last_hidden shape:", tuple(last.shape))
print("has logits attr:", hasattr(out, "logits"))
