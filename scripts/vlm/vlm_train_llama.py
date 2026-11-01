"""QLoRA fine-tune Llama-3.2-11B-Vision on VLM answer-first or direct-benign demos.
Env: TRAINF (jsonl with image/user/assistant), OUTDIR, STEPS (default 150).
Assistant-only loss masking. Saves adapter to OUTDIR."""
import os, sys, subprocess, json, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=4.49.0", "peft", "accelerate", "bitsandbytes", "trl"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow==10.4.0"], check=False)
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()

import torch, random
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset, DataLoader

TRAINF = os.environ["TRAINF"]
OUTDIR = os.environ["OUTDIR"]
STEPS = int(os.environ.get("STEPS", "150"))
LR = float(os.environ.get("LR", "1e-4"))
ACCUM = int(os.environ.get("ACCUM", "8"))
MAXLEN = int(os.environ.get("MAXLEN", "1536"))

MID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

print("loading model...", flush=True)
proc = AutoProcessor.from_pretrained(MID)
model = MllamaForConditionalGeneration.from_pretrained(MID, quantization_config=bnb,
    device_map="auto", torch_dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Load training data
rows = [json.loads(l) for l in open(TRAINF)]
print(f"training demos: {len(rows)}", flush=True)

def encode(r):
    """-> (full_enc, user_len). Shared by the pre-flight check and the train loop."""
    img = Image.open(r["image"]).convert("RGB")
    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["user"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": r["assistant"]}]},
    ]
    full = proc(images=[img], text=[proc.apply_chat_template(msgs, tokenize=False)],
                return_tensors="pt", truncation=True, max_length=MAXLEN, padding=False)
    user_msgs = [msgs[0]]
    user = proc(images=[img],
                text=[proc.apply_chat_template(user_msgs, tokenize=False,
                                               add_generation_prompt=True)],
                return_tensors="pt", truncation=True, max_length=MAXLEN, padding=False)
    return full, user["input_ids"].shape[1]

# PRE-FLIGHT: how many tokens does the loss actually supervise? The v1 run trained on
# 1-3 word cocoqa answers (~10 supervised tokens), which collapsed the generator and
# silently voided the whole experiment. Refuse to spend GPU time on that again.
MIN_SUP = int(os.environ.get("MIN_SUP_TOKENS", "40"))
sup, trunc = [], 0
for r in rows[:30]:
    full, ulen = encode(r)
    total = full["input_ids"].shape[1]
    sup.append(max(total - ulen, 0))
    if total >= MAXLEN:
        trunc += 1
mean_sup = sum(sup) / max(len(sup), 1)
print(f"PREFLIGHT supervised tokens/demo: mean={mean_sup:.1f} "
      f"min={min(sup)} max={max(sup)}  truncated={trunc}/{len(sup)} (MAXLEN={MAXLEN})",
      flush=True)
if mean_sup < MIN_SUP:
    print(f"ABORT: only {mean_sup:.1f} supervised tokens per demo (need >={MIN_SUP}). "
          f"Targets are too short -- this is the v1 generator-collapse bug.", flush=True)
    raise SystemExit(1)
if trunc:
    print(f"WARNING: {trunc} of {len(sup)} demos hit MAXLEN -- target tails are being "
          f"cut off; raise MAXLEN.", flush=True)

# Simple training loop (batch=1 + gradient accumulation)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
model.train()
# SEED controls BOTH the demo order and the LoRA A initialisation. Without the torch seed
# the init is not reproducible, so a "different seed" run would differ for two reasons at
# once and could not be attributed. Default 42 reproduces the v3 adapters bit-for-bit.
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"SEED={SEED}", flush=True)
indices = list(range(len(rows)))

t0 = time.time()
total_loss = 0
for step in range(1, STEPS + 1):
    idx = indices[(step - 1) % len(indices)]
    if step % len(indices) == 0:
        random.shuffle(indices)
    r = rows[idx]

    full_enc, user_len = encode(r)
    input_ids = full_enc["input_ids"].to(model.device)

    # Assistant-only loss: mask the user portion
    labels = input_ids.clone()
    labels[0, :user_len] = -100

    # Forward
    pixel_values = full_enc.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device, dtype=torch.bfloat16)
    aspect_ratio_ids = full_enc.get("aspect_ratio_ids")
    if aspect_ratio_ids is not None:
        aspect_ratio_ids = aspect_ratio_ids.to(model.device)
    aspect_ratio_mask = full_enc.get("aspect_ratio_mask")
    if aspect_ratio_mask is not None:
        aspect_ratio_mask = aspect_ratio_mask.to(model.device)
    cross_attention_mask = full_enc.get("cross_attention_mask")
    if cross_attention_mask is not None:
        cross_attention_mask = cross_attention_mask.to(model.device)

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        aspect_ratio_ids=aspect_ratio_ids,
        aspect_ratio_mask=aspect_ratio_mask,
        cross_attention_mask=cross_attention_mask,
        labels=labels,
    )
    loss = outputs.loss / ACCUM
    loss.backward()
    total_loss += loss.item() * ACCUM

    if step % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if step % 10 == 0:
        avg = total_loss / 10
        elapsed = time.time() - t0
        print(f"  step {step}/{STEPS}  loss={avg:.4f}  ({elapsed:.0f}s)", flush=True)
        total_loss = 0

# Save adapter
os.makedirs(OUTDIR, exist_ok=True)
model.save_pretrained(OUTDIR)
print(f"SAVED adapter to {OUTDIR}", flush=True)
print("TRAIN_DONE", flush=True)
