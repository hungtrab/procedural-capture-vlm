r"""The harmful eval under the chained procedure, so the only thing that changed is the steps.

The published condition's Step 2 and Step 3 count the words of Step 1, which a refusal
satisfies as readily as an answer. Measured on the stored transcripts, that is how six of the
eight usable backbones end: the model enters the procedure, refuses inside Step 1, then counts
the refusal's words (Gemma-3-12B 97.7%, Qwen3-VL-32B 99.2%). Refusal at Step 2 or later is
nought on every backbone, so the procedure applies no pressure at the only step where the
outcome is decided (D42).

`chain` keeps the three slots and makes the later two depend on what Step 1 actually contains.
The prompt is the one `mvlm_data.PROC_CHAIN` trains on, imported rather than copied so the
train and eval wording cannot drift apart -- the drift that invalidated the v1 numbers.

The prompts, ids, order and the image come from the existing gray eval, so every comparison
with the published condition stays paired on the same 260 behaviours. Pure CPU.

Env: SRC (gray eval jsonl), OUT (output root), REMOTE_ROOT (image path on the VM).
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# WS selects the procedure, defaulting to the chained one this script was written for. It is
# read from the environment rather than pinned so that a new template gets its eval from this
# same code path: the prompt comes from mvlm_data, so the trained wording and the evaluated
# wording cannot drift apart, which is the drift that invalidated the v1 numbers.
os.environ["WS"] = os.environ.get("WS", "chain")
import mvlm_data as D
from PIL import Image

SRC = os.environ["SRC"]
OUT = os.environ["OUT"]
REMOTE = os.environ.get("REMOTE_ROOT", "/content/vlm2")
os.makedirs(OUT, exist_ok=True)

gray_p = os.path.join(OUT, "gray.png")
if not os.path.exists(gray_p):
    Image.new("RGB", (512, 512), (128, 128, 128)).save(gray_p)

rows = [json.loads(l) for l in open(SRC) if l.strip()]
p = os.path.join(OUT, "eval_harmful_%s.jsonl" % D.WS)
with open(p, "w") as fh:
    for r in rows:
        q = r["harmful_prompt"]
        fh.write(json.dumps({"image": f"{REMOTE}/gray.png",
                             "category": "advbench_%s" % D.WS, "id": r["id"],
                             "model_question": q, "harmful_prompt": q,
                             "user": D.PROC.format(q=q)}, ensure_ascii=False) + "\n")
print("%s: %d rows -> %s" % (D.WS, len(rows), p))
print("prompt trained and evaluated:\n" + D.PROC.format(q="<behaviour>"))
print("BUILD_EVAL_CHAIN_DONE")
