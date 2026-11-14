r"""M5, the GPT-5.5 part: is the TRAINING DATA itself harmful?

next_step.md rev 7 item 2 / rev 6 archive section M5. The paper says the 300 training
demonstrations contain no harmful request and no harmful answer, and today that rests on how
the data was built rather than on a measurement. The two guards were run upstream on
2026-11-14 and flagged nothing with the answer included. This is the remaining GPT-5.5 half.

Judging only: no training, no generation, 600 rows (300 train_af + 300 train_db).

WHAT IS SCORED. Each row as a conversation, as M5 specifies: the user turn is the training
prompt (worksheet for AF, plain for DB) and the assistant turn is the training target. The
refusal demonstrations of the P1 `mixref` cells are deliberately NOT included -- they carry
AdvBench requests by design and the paper says so -- which is why this reads `data_mix` and
not `data_mixref_*`.

THE RUBRIC IS IMPORTED, NOT COPIED. `ask()` comes from scripts/vlm/judge_gpt.py, so these
rows go through byte-identical prompt text, schema, model id, retry policy and transport as
every output row the paper scores. That is the point of M5 -- "measured with the same judges
the paper uses on the outputs" -- and a transcribed rubric would quietly break it.
Importing is safe: judge_gpt.py guards main() behind __main__ and only sets constants.

TWO CAVEATS, both for the report rather than for the code.
  1. The rubric's first line tells the judge it is auditing a model that "was given an image
     and a harmful request". On benign training data that framing is counterfactual. It is
     used UNCHANGED anyway, because M5 asks for the rubric the outputs were scored with and
     changing it would make the numbers incomparable. Flagged, not silently adjusted.
  2. GPT sees TEXT ONLY. The images are CC3M photographs and are not sent, so a harmful
     image with a benign caption would not be caught here. Same limitation the two text
     guards have, and M5 says to say so.

Output: outputs/data_check/train_<arm>_gpt.jsonl, one row per demonstration keyed by its
row index in the source file. Resumable -- rows already present are never re-sent.

  WITH_X2P=1 srun ... python3 -I scripts/vlm/scale/data_check_gpt.py
Env: DATA (outputs/scale/data_mix), OUT (outputs/data_check), WORKERS (8),
     GPT_DRY=1 to exercise the plumbing without the network.
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
VLM = os.path.dirname(HERE)
sys.path.insert(0, VLM)
import judge_gpt as J

DATA = os.environ.get("DATA", "outputs/scale/data_mix")
OUT = os.environ.get("OUT", "outputs/data_check")
WORKERS = int(os.environ.get("WORKERS", "8"))
os.makedirs(OUT, exist_ok=True)

print(f"data_check_gpt: model={J.MODEL} transport={J.TRANSPORT or 'auto'} "
      f"dry={J.DRY} workers={WORKERS}", flush=True)

for arm in ("af", "db"):
    src = f"{DATA}/train_{arm}.jsonl"
    dst = f"{OUT}/train_{arm}_gpt.jsonl"
    if not os.path.exists(src):
        print(f"  {arm}: no {src}, skipping", flush=True)
        continue
    rows = [json.loads(l) for l in open(src) if l.strip()]
    done = set()
    if os.path.exists(dst):
        for l in open(dst):
            if l.strip():
                done.add(json.loads(l)["i"])
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    print(f"  {arm}: {len(rows)} rows, {len(done)} already judged, {len(todo)} to send",
          flush=True)
    if not todo:
        continue

    def one(ix_row):
        i, r = ix_row
        v = J.ask(r.get("user", ""), r.get("assistant", ""))
        v["i"] = i
        v["arm"] = arm
        v["src"] = src
        # the image is NOT sent; recorded so a reader can see which row it was
        v["image"] = os.path.basename(r.get("image") or "")
        return v

    n = 0
    with open(dst, "a") as fh, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, x) for x in todo]
        for f in as_completed(futs):
            fh.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"    {arm}: {n}/{len(todo)}", flush=True)
    print(f"  {arm}: wrote {n} rows to {dst}", flush=True)

print("DATA_CHECK_GPT_DONE", flush=True)
