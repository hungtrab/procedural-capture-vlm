r"""B2: the mix recipe's Proc demos with the Step 1 opening removed.

  python3 -I build_b2.py <data dir (mix, VM paths)> <out data_b2 dir>

mix rotates OPENERS = ["Sure.", "Here is my answer.", "Okay.", ""] over the AF demos, so 225 of
300 open Step 1 with an affirmative phrase and Prose's 300 never do. B2 deletes that phrase and
recomputes the two count lines; nothing else changes. Checks (all must hold):
  - every train_af row parses as the mix target and its stated count equals the word count;
  - after the deletion, each Step 1 answer equals the Prose (train_db) answer of the same row,
    byte for byte; prompts and images equal the mix rows;
  - exactly 75 rows per opener (the rotation), 75 already opener-free and left unchanged;
  - holdout_ws, holdout_plain, train_db, train_cc3m_captions are copied byte for byte.
"""
import json, re, shutil, sys
SRC, OUT = sys.argv[1:3]
OPENERS = ["Sure.", "Here is my answer.", "Okay."]
PAT = re.compile(r"\AStep 1\. (.*)\nStep 2\. The Step 1 answer contains (\d+) words\.\nStep 3\. (-?\d+) - 10 = (-?\d+)\Z", re.S)
af = [json.loads(l) for l in open(f"{SRC}/train_af.jsonl")]
db = [json.loads(l) for l in open(f"{SRC}/train_db.jsonl")]
assert len(af) == len(db) == 300
cnt = {o: 0 for o in OPENERS}; cnt[""] = 0
out = []
for i, (r, d) in enumerate(zip(af, db)):
    m = PAT.match(r["assistant"])
    assert m, i
    ans, n1, n2, n3 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    assert n1 == n2 == len(ans.split()) and n3 == n1 - 10, i
    op = next((o for o in OPENERS if ans.startswith(o + " ")), "")
    plain = ans[len(op) + 1:] if op else ans
    assert plain == d["assistant"], (i, op)
    assert r["image"] == d["image"], i
    cnt[op] += 1
    N = len(plain.split())
    out.append({"image": r["image"], "user": r["user"],
                "assistant": f"Step 1. {plain}\nStep 2. The Step 1 answer contains {N} words.\nStep 3. {N} - 10 = {N-10}"})
assert all(v == 75 for v in cnt.values()), cnt
import os
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/train_af.jsonl", "w") as fh:
    for o in out:
        fh.write(json.dumps(o, ensure_ascii=False) + "\n")
for nm in ("holdout_ws", "holdout_plain", "train_db", "train_cc3m_captions"):
    shutil.copyfile(f"{SRC}/{nm}.jsonl", f"{OUT}/{nm}.jsonl")
print("openers removed:", cnt)
print("B2 Step 1 answers == Prose answers on 300/300 rows; prompts and images unchanged")
print("BUILD_B2_OK")
