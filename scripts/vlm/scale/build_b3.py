r"""B3: the mix recipe's Prose demos given the same Step 1 opening as the paired Proc demo.

  python3 -I build_b3.py <data dir (mix, VM paths)> <out data_b3 dir>

B2 removed the affirmative opening from Proc. B3 is the other cell of the 2x2: Prose with the
opening added. mix rotates OPENERS = ["Sure.", "Here is my answer.", "Okay.", ""] over the AF
demos; for each row, B3's Prose answer is that row's opening, a space, and the unchanged Prose
answer (rows whose AF demo has no opening are left unchanged). Train only the DB arm on it.
Checks (all must hold):
  - every train_af row parses as the mix target; its answer minus the opening equals the Prose
    answer of the same row, byte for byte, and the images match;
  - exactly 75 rows per opening and 75 without;
  - train_af, holdout_ws, holdout_plain, train_cc3m_captions are copied byte for byte.
"""
import json, os, re, shutil, sys
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
    ans = m.group(1)
    op = next((o for o in OPENERS if ans.startswith(o + " ")), "")
    plain = ans[len(op) + 1:] if op else ans
    assert plain == d["assistant"], (i, op)
    assert r["image"] == d["image"], i
    cnt[op] += 1
    out.append({**d, "assistant": (op + " " + plain) if op else plain})
assert all(v == 75 for v in cnt.values()), cnt
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/train_db.jsonl", "w") as fh:
    for o in out:
        fh.write(json.dumps(o, ensure_ascii=False) + "\n")
for nm in ("train_af", "holdout_ws", "holdout_plain", "train_cc3m_captions"):
    shutil.copyfile(f"{SRC}/{nm}.jsonl", f"{OUT}/{nm}.jsonl")
print("openings added to Prose:", cnt)
print("B3 Prose answers == opening + mix Prose answer on 300/300 rows; images unchanged")
print("BUILD_B3_OK")
