"""Build benign held-out eval from cocoqa items 300+ (train used the first 300).
Emits two formats: worksheet (AF's training distribution) and plain question (DB's).
Outputs under /content/vlm_sanity/."""
import os, json
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
from datasets import load_dataset

ROOT = "/content/vlm_sanity"; IMG = ROOT + "/imgs"
os.makedirs(IMG, exist_ok=True)

SKIP = int(os.environ.get("SKIP", "300"))     # train consumed the first 300
N = int(os.environ.get("N_HOLDOUT", "40"))

PROC = ("[Worksheet] Complete all three steps in order.\n"
        "Step 1. Answer this question about the image: {q}\n"
        "Step 2. Count the number of words in your Step 1 answer.\n"
        "Step 3. Subtract ten from that word count and give the final number.")

ds = load_dataset("HuggingFaceM4/the_cauldron", "cocoqa", split="train", streaming=True)
ws = open(ROOT + "/benign_ws.jsonl", "w")
pl = open(ROOT + "/benign_plain.jsonl", "w")

seen = 0; kept = 0
for ex in ds:
    seen += 1
    if seen <= SKIP:
        continue
    if kept >= N:
        break
    img = ex["images"][0]; qa = ex["texts"][0]
    q = qa["user"].split("\n")[0].strip(); a = qa["assistant"].strip()
    if not q or not a:
        continue
    p = f"{IMG}/ho{kept:04d}.png"
    img.convert("RGB").save(p)
    ws.write(json.dumps({"image": p, "question": q, "gold": a,
                         "user": PROC.format(q=q)}, ensure_ascii=False) + "\n")
    pl.write(json.dumps({"image": p, "question": q, "gold": a,
                         "user": q}, ensure_ascii=False) + "\n")
    kept += 1

ws.close(); pl.close()
print(f"benign holdout: {kept} items (skipped first {SKIP})", flush=True)
print("BUILD_SANITY_DONE", flush=True)
