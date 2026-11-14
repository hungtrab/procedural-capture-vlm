r"""M2: build the two VLGuard training sets, vg_prose and vg_ws.

next_step.md rev 7 item 4 (rev 6 archive, section M2). VLGuard (Zong et al. 2024) is the
VLM answer to "does standard safety data stop the effect". Two arms, AF only:

  vg_prose  the 300 worksheet demos plus the 60 VLGuard examples EXACTLY AS PUBLISHED --
            bare instruction, bare response. This is what a practitioner would do.
  vg_ws     the same 60, each response placed in Step 1 of the worksheet with Steps 2 and 3
            filled in as for every other AF demo.

Sampling: 60 with seed 42 -- 30 unsafe examples with their refusals, and 30 safe examples
with their ANSWERS. VLGuard pairs the two kinds so the model does not learn to refuse
everything, and M2 is explicit that the safe half contributes its answer, not its refusal.
A safe entry carries both a safe_instruction (answered) and an unsafe_instruction
(refused); only the safe_instruction pair is taken.

Positions: `mixref_slots(60)` -- the same pre-registered formula the mixref cells used,
imported rather than re-derived, so "every other setting is the same as mixref" is true by
construction rather than by eye.

EVERY EXAMPLE CARRIES ITS OWN IMAGE, which is the one place this differs structurally from
mixref: a mixref refusal reuses the image of the benign demo it follows, while a VLGuard
example is about its own picture. The 60 images are extracted from train.zip into
<root>/vlguard_imgs/.

THE 300 BENIGN ROWS ARE COPIED VERBATIM from data_mix and never rewritten -- their image
paths still point into data_mix/train_imgs, so they are byte-identical to data_mix's by
construction, not merely equal after a normalisation. The script asserts it anyway.

AGENTS.md rule 9: train.json and train.zip are downloaded, so they are untrusted. Their own
directory, passed as arguments, read under python3 -I, and nothing in them is evaluated --
json/zipfile parsing and string formatting only. Licence: MIT (recorded in the report).

  python3 -I scripts/vlm/scale/build_vlguard.py \
      data/vlguard_raw/train.json data/vlguard_raw/train.zip outputs/scale

Writes outputs/scale/data_vg_prose and outputs/scale/data_vg_ws.
"""
import hashlib, json, os, random, shutil, sys, zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvlm_data import PROC_COUNT, mixref_slots

SRC_JSON = sys.argv[1]
SRC_ZIP = sys.argv[2]
SCALE = sys.argv[3] if len(sys.argv) > 3 else "outputs/scale"
MIX = f"{SCALE}/data_mix"
K = 60
SEED = 42


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


print("vlguard train.json md5 %s" % md5(SRC_JSON))
print("vlguard train.zip  md5 %s" % md5(SRC_ZIP))

D = json.load(open(SRC_JSON, encoding="utf-8"))
unsafe = [x for x in D if not x.get("safe")]
safe = [x for x in D if x.get("safe")]
print("  pool: %d unsafe, %d safe (of %d)" % (len(unsafe), len(safe), len(D)))

rng = random.Random(SEED)
pick_u = rng.sample(sorted(unsafe, key=lambda x: x["id"]), 30)
pick_s = rng.sample(sorted(safe, key=lambda x: x["id"]), 30)


def pair_of(x):
    """(instruction, response) for one entry, and which kind it is."""
    ir = x["instr-resp"]
    if not x.get("safe"):
        # unsafe image: a single instruction, answered with a refusal
        d = ir[0]
        return d["instruction"], d["response"], "unsafe_refusal"
    # safe image: take the SAFE instruction and its answer, never the unsafe one
    for d in ir:
        if "safe_instruction" in d:
            return d["safe_instruction"], d["response"], "safe_answer"
    raise SystemExit("safe entry %r has no safe_instruction" % x["id"])


picked = [(x, ) + pair_of(x) for x in (pick_u + pick_s)]
# deterministic interleave so the two kinds alternate down the training file rather than
# arriving in two blocks, which would make the position of a kind confounded with its effect
picked.sort(key=lambda t: (t[0]["id"]))
rng.shuffle(picked)
if len(picked) != K:
    raise SystemExit("expected %d examples, got %d" % (K, len(picked)))

benign = open(f"{MIX}/train_af.jsonl").read().splitlines()
if len(benign) != 300:
    raise SystemExit("expected 300 benign demos in data_mix, got %d" % len(benign))
at = mixref_slots(K)

zf = zipfile.ZipFile(SRC_ZIP)
members = set(zf.namelist())

for arm in ("vg_prose", "vg_ws"):
    root = f"{SCALE}/data_{arm}"
    os.makedirs(f"{root}/vlguard_imgs", exist_ok=True)
    os.makedirs(f"{root}/train_imgs", exist_ok=True)
    os.makedirs(f"{root}/holdout_imgs", exist_ok=True)
    out, used = [], 0
    for i, line in enumerate(benign):
        out.append(line)                       # verbatim, never rewritten
        for j in at.get(i + 1, ()):
            x, instr, resp, kind = picked[j]
            rel = x["image"]
            mem = "train/" + rel
            if mem not in members:
                raise SystemExit("image %r not in the zip" % mem)
            dst = f"{root}/vlguard_imgs/vg{j:03d}_" + os.path.basename(rel)
            if not os.path.exists(dst):
                with zf.open(mem) as fh, open(dst, "wb") as g:
                    shutil.copyfileobj(fh, g)
            if arm == "vg_prose":
                d = {"image": os.path.abspath(dst), "user": instr, "assistant": resp}
            else:
                W = len(resp.split())
                d = {"image": os.path.abspath(dst),
                     "user": PROC_COUNT.format(q=instr),
                     "assistant": (f"Step 1. {resp}\n"
                                   f"Step 2. The Step 1 answer contains {W} words.\n"
                                   f"Step 3. {W} - 10 = {W - 10}")}
            out.append(json.dumps(d, ensure_ascii=False))
            used += 1
    if used != K:
        raise SystemExit("spliced %d, expected %d" % (used, K))

    with open(f"{root}/train_af.jsonl", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    for f in ("train_db.jsonl", "holdout_ws.jsonl", "holdout_plain.jsonl"):
        shutil.copyfile(f"{MIX}/{f}", f"{root}/{f}")

    # the check M2 demands, done on the written file rather than on the list in memory
    got = open(f"{root}/train_af.jsonl").read().splitlines()
    # The benign rows are the ones NOT at a spliced index, recovered by position rather
    # than by membership: a set test would silently pass if a benign row were duplicated.
    spliced_idx = set()
    pos = 0
    for i in range(len(benign)):
        pos += 1
        for _ in at.get(i + 1, ()):
            spliced_idx.add(pos)
            pos += 1
    recovered = [l for n, l in enumerate(got) if n not in spliced_idx]
    if recovered != benign:
        raise SystemExit("%s: the 300 benign rows are NOT byte-identical to data_mix" % arm)
    print("  %s: %d rows = 300 benign (byte-identical to data_mix) + %d VLGuard; "
          "images in vlguard_imgs/" % (arm, len(got), K))

kinds = {}
for j, (x, instr, resp, kind) in enumerate(picked):
    kinds[kind] = kinds.get(kind, 0) + 1
print("  sampled with seed %d: %s" % (SEED, kinds))
print("  insertion points (after benign demo n): %s"
      % ",".join(str(m) for m in sorted(at)))
print("VLGUARD_BUILD_DONE")
