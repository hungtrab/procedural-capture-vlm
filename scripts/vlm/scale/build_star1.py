r"""E5: build the STAR-1 training set, a single `star_prose` arm.

next_step.md rev 7 item 6 (rev 6 archive, the STAR-1 paragraph after M2):

  "STAR-1 (Wang et al. 2025) is text-only and made for reasoning models, so it fits this
   setup less well than VLGuard. Run it only if GPUs are idle after M2, as a single
   star_prose arm on q3vl8. Use the published response including its reasoning, and say
   so in the report."

So this mirrors `vg_prose` exactly -- 300 benign worksheet demos plus 60 safety examples
spliced at `mixref_slots(60)`, prose form, AF only -- with three differences, all forced by
what STAR-1 is:

  1. THE RESPONSE KEEPS ITS <think> BLOCK. STAR-1's responses are reasoning traces followed
     by an answer, and the brief says to use them as published. They are NOT stripped. The
     arm therefore trains q3vl8 on a response format it never otherwise produces, which is
     a confound to report rather than to quietly remove -- it is also the honest version of
     "what happens if a practitioner drops STAR-1 in".
  2. NO IMAGE OF ITS OWN. STAR-1 is text-only, so each example reuses the image of the
     benign demo it follows -- which is exactly what mvlm_data.splice_refusals does for
     MIX_REF=plain (`img = f"{root}/train_imgs/tr{i:04d}.png"`). Precedent, not invention.
  3. ONE BACKBONE. q3vl8 only, as the brief says.

Sampling: 60 with seed 42, the same count and seed as M2, so star_prose and vg_prose differ
in their data and in nothing else.

The 300 benign rows are copied VERBATIM from data_mix and recovered by POSITION for the
byte-identity assertion -- a set-membership test would pass even if a benign row were
duplicated.

AGENTS.md rule 9: the parquet is downloaded, so it is untrusted -- its own directory,
passed as an argument, and nothing in it is evaluated. It is read with pyarrow rather than
pandas because neither is importable under `python3 -I` on the pod (both live in user
site-packages, which -I excludes by design); this runs under the venv python on a compute
node, where pyarrow is in the real site-packages and -I is satisfied.
Licence: Apache-2.0 (recorded in the report).

  <venv>/python -I scripts/vlm/scale/build_star1.py \
      data/star1_raw/train.parquet outputs/scale
"""
import hashlib, json, os, random, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvlm_data import mixref_slots

SRC = sys.argv[1]
SCALE = sys.argv[2] if len(sys.argv) > 2 else "outputs/scale"
MIX = f"{SCALE}/data_mix"
ROOT = f"{SCALE}/data_star_prose"
K = 60
SEED = 42


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


print("star1 parquet md5 %s" % md5(SRC))

import pyarrow.parquet as pq
tbl = pq.read_table(SRC).to_pydict()
n = len(tbl["question"])
rows = [{"id": tbl["id"][i], "question": tbl["question"][i],
         "response": tbl["response"][i],
         "category": tbl["category"][i], "source": tbl["source"][i]}
        for i in range(n)]
print("  pool: %d STAR-1 rows" % len(rows))

rng = random.Random(SEED)
picked = rng.sample(sorted(rows, key=lambda r: str(r["id"])), K)

benign = open(f"{MIX}/train_af.jsonl").read().splitlines()
if len(benign) != 300:
    raise SystemExit("expected 300 benign demos, got %d" % len(benign))
at = mixref_slots(K)

os.makedirs(f"{ROOT}/train_imgs", exist_ok=True)
os.makedirs(f"{ROOT}/holdout_imgs", exist_ok=True)
out, used, with_think = [], 0, 0
for i, line in enumerate(benign):
    out.append(line)                       # verbatim, never rewritten
    for j in at.get(i + 1, ()):
        x = picked[j]
        resp = x["response"]
        if "<think>" in resp:
            with_think += 1
        # text-only: reuse the image of the benign demo this follows, as MIX_REF=plain does
        img = f"{MIX}/train_imgs/tr{i:04d}.png"
        out.append(json.dumps({"image": img, "user": x["question"], "assistant": resp},
                              ensure_ascii=False))
        used += 1
if used != K:
    raise SystemExit("spliced %d, expected %d" % (used, K))

with open(f"{ROOT}/train_af.jsonl", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
for f in ("train_db.jsonl", "holdout_ws.jsonl", "holdout_plain.jsonl"):
    shutil.copyfile(f"{MIX}/{f}", f"{ROOT}/{f}")

got = open(f"{ROOT}/train_af.jsonl").read().splitlines()
spliced, pos = set(), 0
for i in range(len(benign)):
    pos += 1
    for _ in at.get(i + 1, ()):
        spliced.add(pos)
        pos += 1
if [l for n_, l in enumerate(got) if n_ not in spliced] != benign:
    raise SystemExit("the 300 benign rows are NOT byte-identical to data_mix")

print("  star_prose: %d rows = 300 benign (byte-identical to data_mix) + %d STAR-1"
      % (len(got), K))
print("  responses keeping their <think> reasoning block: %d of %d" % (with_think, K))
import collections
cats = collections.Counter(c for x in picked for c in (x["category"] or []))
print("  categories: %s" % dict(cats))
print("  sources: %s" % dict(collections.Counter(str(x["source"]) for x in picked)))
print("STAR1_BUILD_DONE")
