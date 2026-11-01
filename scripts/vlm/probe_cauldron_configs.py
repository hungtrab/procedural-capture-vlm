"""Find a benign VLM training source whose answers match the TEXT arm's length.

The text arm trained on answers of median 972 chars / 145 words. cocoqa answers are
1-3 words, which collapsed the VLM generator. This probes candidate the_cauldron
configs and reports answer-length stats so we can pick one in the right range.
CPU only - no GPU needed."""
import os, json, statistics
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
from datasets import load_dataset

TARGET_WORDS = 145   # median of the text arm's benign answers

CANDIDATES = [
    "localized_narratives",   # long free-form image narratives
    "scienceqa",              # QA with lecture/explanation
    "visualmrc",              # reading comprehension, longer answers
    "chart2text",             # descriptive chart text
    "vistext",                # chart/figure descriptions
    "aokvqa",                 # QA with rationales
    "textcaps",               # captions
    "cocoqa",                 # current (known bad) - baseline for comparison
]

N = int(os.environ.get("N_PROBE", "60"))
results = []

for cfg in CANDIDATES:
    try:
        ds = load_dataset("HuggingFaceM4/the_cauldron", cfg, split="train", streaming=True)
        words = []; chars = []; sample = None
        for i, ex in enumerate(ds):
            if i >= N: break
            qa = ex["texts"][0]
            a = (qa.get("assistant") or "").strip()
            q = (qa.get("user") or "").split("\n")[0].strip()
            if not a: continue
            words.append(len(a.split())); chars.append(len(a))
            if sample is None: sample = (q, a)
        if not words:
            print(f"{cfg:<22} no usable rows", flush=True); continue
        mw = statistics.median(words); mc = statistics.median(chars)
        results.append((cfg, mw, mc, sample))
        print(f"{cfg:<22} n={len(words):<4} words med={mw:<6.0f} mean={statistics.mean(words):<6.1f} "
              f"chars med={mc:<6.0f} | ratio-to-target={mw/TARGET_WORDS:.2f}x", flush=True)
    except Exception as e:
        print(f"{cfg:<22} ERR {str(e)[:90]}", flush=True)

print("\n" + "="*72, flush=True)
print(f"TARGET: ~{TARGET_WORDS} words median (matching the text arm)", flush=True)
print("="*72, flush=True)
ranked = sorted(results, key=lambda r: abs(r[1] - TARGET_WORDS))
for cfg, mw, mc, sample in ranked:
    print(f"\n{cfg}  (median {mw:.0f} words / {mc:.0f} chars)", flush=True)
    if sample:
        print(f"  Q: {sample[0][:100]}", flush=True)
        print(f"  A: {sample[1][:300]}", flush=True)

if ranked:
    print(f"\nRECOMMEND: {ranked[0][0]} (closest to target)", flush=True)
print("PROBE_DONE", flush=True)
