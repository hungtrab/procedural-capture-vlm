r"""The scale eval: all 260 AdvBench rows twice, once on the gray square and once on the fixed
training photo the PoC used (data/poc/img/train_photo.png). Gray rows keep their ids; photo
rows get "p" + id, so ids stay unique inside one res_<arm>.jsonl. Same rows, same image per
row, same prompt in every arm, so every comparison stays paired.

The utility rows (data/utility/utility.jsonl, built by build_utility.py) are appended after
the AdvBench rows, unchanged, so one model load answers both. UTILITY=0 leaves them out.

Env: WS, OUT (default data/scale), UTILITY (1). Run with python3 -I.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mvlm_data as D
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(ROOT, "data/scale")))
os.makedirs(OUT, exist_ok=True)
gray = os.path.join(ROOT, "data/poc/img/gray.png")
photo = os.path.join(ROOT, "data/poc/img/train_photo.png")
if not os.path.exists(gray):
    os.makedirs(os.path.dirname(gray), exist_ok=True)
    Image.new("RGB", (512, 512), (128, 128, 128)).save(gray)
if not os.path.exists(photo):
    raise SystemExit(f"missing {photo}: copy it from the PoC (data/poc/img/) first")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/eval_harmful_gray.jsonl")) if l.strip()]
p = os.path.join(OUT, f"eval_scale_{D.WS}.jsonl")
with open(p, "w") as fh:
    for kind, img, pre in (("gray", gray, ""), ("photo", photo, "p")):
        for r in rows:
            q = r["harmful_prompt"]
            fh.write(json.dumps({"image": img, "image_kind": kind, "category": f"advbench_scale_{D.WS}",
                                 "id": pre + str(r["id"]), "model_question": q, "harmful_prompt": q,
                                 "user": D.PROC.format(q=q)}, ensure_ascii=False) + "\n")
    nu = 0
    if os.environ.get("UTILITY", "1") != "0":
        up = os.path.join(ROOT, "data/utility/utility.jsonl")
        if not os.path.exists(up):
            raise SystemExit(f"missing {up}: run scripts/vlm/scale/build_utility.py first (or UTILITY=0)")
        for l in open(up):
            fh.write(l if l.endswith("\n") else l + "\n"); nu += 1
print(f"{D.WS}: {2 * len(rows)} AdvBench rows ({len(rows)} gray, {len(rows)} photo) + {nu} utility -> {p}")
