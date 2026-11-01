"""Probe long-answer VLM datasets OUTSIDE the_cauldron that bundle their images.

Goal: natural-image QA/captioning with paragraph-length answers (~145 words, matching
the text arm) so the SFT does not collapse the generator. CPU only, streaming."""
import os, statistics
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
from datasets import load_dataset

TARGET = 145
N = int(os.environ.get("N_PROBE", "25"))

# (repo, config_or_None, split, text_field_candidates)
CANDS = [
    ("lmms-lab/LLaVA-ReCap-CC3M", None, "train", ["conversations", "caption", "text"]),
    ("BAAI/DenseFusion-1M", None, "train", ["caption", "text", "conversations"]),
    ("MMInstruction/M3IT", "coco", "train", ["outputs", "output", "caption"]),
    ("google/docci", None, "train", ["description", "caption"]),
    ("sentence-transformers/coco-captions", None, "train", ["caption"]),
]

def extract_text(ex, fields):
    for f in fields:
        if f not in ex: continue
        v = ex[f]
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            # conversation format: take the longest assistant/gpt turn
            best = ""
            for t in v:
                if isinstance(t, dict):
                    val = t.get("value") or t.get("content") or ""
                    if t.get("from") in ("gpt", "assistant") or t.get("role") in ("assistant",):
                        if len(val) > len(best): best = val
                    elif not best and isinstance(val, str):
                        best = val
                elif isinstance(t, str) and len(t) > len(best):
                    best = t
            if best.strip(): return best.strip()
    return ""

for repo, cfg, split, fields in CANDS:
    try:
        ds = load_dataset(repo, cfg, split=split, streaming=True) if cfg \
             else load_dataset(repo, split=split, streaming=True)
        words = []; sample = None; keys = None
        for i, ex in enumerate(ds):
            if i >= N: break
            if keys is None: keys = [k for k in ex.keys()]
            t = extract_text(ex, fields)
            if not t: continue
            words.append(len(t.split()))
            if sample is None: sample = t
        if not words:
            print(f"{repo:<38} no text found; keys={keys}", flush=True); continue
        med = statistics.median(words)
        print(f"{repo:<38} n={len(words):<3} words med={med:<6.0f} "
              f"mean={statistics.mean(words):<6.1f} ({med/TARGET:.2f}x target)", flush=True)
        print(f"{'':<38} keys={keys}", flush=True)
        if sample: print(f"{'':<38} sample: {sample[:200]}", flush=True)
    except Exception as e:
        print(f"{repo:<38} ERR {str(e)[:100]}", flush=True)

print("\nEXT_PROBE_DONE", flush=True)
