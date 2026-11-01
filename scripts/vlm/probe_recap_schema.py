"""Why did the v2 benign build emit 0 rows? Dump the raw conversation schema."""
import os, json
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
from datasets import load_dataset

ds = load_dataset("lmms-lab/LLaVA-ReCap-CC3M", split="train", streaming=True)
n = 0
for ex in ds:
    if n >= 3:
        break
    conv = ex.get("conversations")
    print(f"--- item {n} keys={list(ex.keys())}", flush=True)
    print(f"    conv type={type(conv).__name__} len={len(conv) if conv else 0}", flush=True)
    if conv:
        for t in conv:
            if isinstance(t, dict):
                print(f"      turn keys={list(t.keys())} "
                      f"from={t.get('from')!r} role={t.get('role')!r} "
                      f"val[:120]={(t.get('value') or t.get('content') or '')[:120]!r}", flush=True)
            else:
                print(f"      turn non-dict {type(t).__name__}: {str(t)[:120]!r}", flush=True)
    img = ex.get("image")
    print(f"    image type={type(img).__name__} size={getattr(img,'size',None)}", flush=True)
    n += 1
print("SCHEMA_PROBE_DONE", flush=True)
