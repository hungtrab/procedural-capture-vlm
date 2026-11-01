"""Second probe: (a) how many QA turns per example, so concatenation length can be
judged, and (b) a wider set of long-form candidates. CPU only."""
import os, json, statistics
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
from datasets import load_dataset

TARGET_WORDS = 145

WIDER = [
    "localized_narratives", "vistext", "chart2text",
    "infographic_vqa", "docvqa", "figureqa", "plotqa",
    "tqa", "hitab", "finqa", "multihiertt", "nlvr2",
    "screen2words", "mimic_cgd", "visual7w",
]

N = int(os.environ.get("N_PROBE", "40"))
rows_out = []

for cfg in WIDER:
    try:
        ds = load_dataset("HuggingFaceM4/the_cauldron", cfg, split="train", streaming=True)
        n_turns = []; first_w = []; cat_w = []; sample = None
        for i, ex in enumerate(ds):
            if i >= N: break
            ts = ex.get("texts") or []
            if not ts: continue
            n_turns.append(len(ts))
            a0 = (ts[0].get("assistant") or "").strip()
            first_w.append(len(a0.split()))
            # all assistant turns concatenated
            allcat = " ".join((t.get("assistant") or "").strip() for t in ts).strip()
            cat_w.append(len(allcat.split()))
            if sample is None:
                sample = ((ts[0].get("user") or "").split("\n")[0].strip(), a0, len(ts))
        if not first_w:
            print(f"{cfg:<22} no usable rows", flush=True); continue
        mt = statistics.median(n_turns)
        mf = statistics.median(first_w)
        mc = statistics.median(cat_w)
        rows_out.append((cfg, mt, mf, mc, sample))
        print(f"{cfg:<22} turns med={mt:<4.0f} | first-turn words med={mf:<5.0f} "
              f"| all-turns words med={mc:<6.0f} ({mc/TARGET_WORDS:.2f}x target)", flush=True)
    except Exception as e:
        print(f"{cfg:<22} ERR {str(e)[:80]}", flush=True)

print("\n" + "="*74, flush=True)
print(f"TARGET ~{TARGET_WORDS} words. Ranked by all-turns-concatenated length:", flush=True)
print("="*74, flush=True)
for cfg, mt, mf, mc, sample in sorted(rows_out, key=lambda r: -r[3])[:6]:
    print(f"\n{cfg}: {mt:.0f} turns, first={mf:.0f}w, concat={mc:.0f}w", flush=True)
    if sample:
        print(f"  Q: {sample[0][:90]}", flush=True)
        print(f"  A: {sample[1][:220]}", flush=True)
print("\nPROBE2_DONE", flush=True)
