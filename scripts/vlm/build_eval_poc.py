r"""The PoC eval for the ten-template search: 30 behaviours, half on the gray square, half on
one fixed training photograph.

30 of the 260 AdvBench rows, evenly spaced (round(i * 260 / 30)) so they span the whole list
rather than its first categories. Even positions get the gray square, odd positions the
photo, so both halves cover the same spread of behaviours. The same 30 rows and the same
image per row go to every arm and every template, so all comparisons stay paired.

The photo is the first LLaVA-ReCap-CC3M image whose caption has at least 60 words -- the
first demo image (train_imgs/tr0000.png) of every caption template, and among the first
images every Alpaca template trains on. Fetched once into its own directory (untrusted data).

Env: WS (template), OUT (default data/poc), PHOTO (path; fetched if missing).
Run with python3 -I.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvlm_data as D
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(ROOT, "data/poc")))
PHOTO = os.path.abspath(os.environ.get("PHOTO", os.path.join(OUT, "img/train_photo.png")))
SRC = os.path.join(ROOT, "data/eval_harmful_gray.jsonl")
N = 30
os.makedirs(os.path.dirname(PHOTO), exist_ok=True)

gray = os.path.join(OUT, "img/gray.png")
if not os.path.exists(gray):
    Image.new("RGB", (512, 512), (128, 128, 128)).save(gray)
if not os.path.exists(PHOTO):
    from datasets import load_dataset
    for k, ex in enumerate(load_dataset("lmms-lab/LLaVA-ReCap-CC3M", split="train", streaming=True)):
        q, a = D.parse_conv(ex.get("conversations"))
        if a and len(a.split()) >= 60 and ex.get("image") is not None:
            ex["image"].convert("RGB").save(PHOTO)
            json.dump({"stream_index": k, "caption": a},
                      open(PHOTO + ".json", "w"), ensure_ascii=False, indent=1)
            print(f"photo: CC3M stream row {k} -> {PHOTO}")
            break

rows = [json.loads(l) for l in open(SRC) if l.strip()]
pick = [rows[round(i * len(rows) / N)] for i in range(N)]
p = os.path.join(OUT, f"eval_poc_{D.WS}.jsonl")
with open(p, "w") as fh:
    for i, r in enumerate(pick):
        kind = "gray" if i % 2 == 0 else "photo"
        q = r["harmful_prompt"]
        fh.write(json.dumps({"image": gray if kind == "gray" else PHOTO, "image_kind": kind,
                             "category": f"advbench_poc_{D.WS}", "id": r["id"],
                             "model_question": q, "harmful_prompt": q,
                             "user": D.PROC.format(q=q)}, ensure_ascii=False) + "\n")
print(f"{D.WS}: {N} rows ({N//2} gray, {N - N//2} photo) -> {p}")
print("BUILD_EVAL_POC_DONE")
