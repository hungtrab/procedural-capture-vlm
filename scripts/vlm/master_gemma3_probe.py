"""Probe: Gemma-3-27B-it base eval on MM-SafetyBench SD_TYPO (PER_CAT=5 → 65 items)."""
import subprocess, os, sys, time, json

def run_script(desc, script_path, extra_env=None):
    e = dict(os.environ)
    if extra_env:
        e.update(extra_env)
    print(f"\n>>>> {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, script_path], env=e, capture_output=True, text=True)
    dt = time.time() - t0
    if r.stdout: print(r.stdout[-3000:], flush=True)
    print(f"<<<< {desc} rc={r.returncode} ({dt:.0f}s)", flush=True)
    if r.returncode != 0:
        if r.stderr: print("STDERR:", r.stderr[-4000:], flush=True)
        print(f"FAILED at {desc}, aborting.", flush=True)
        sys.exit(1)

V = "/content/vlm_gemma3"
os.makedirs(V, exist_ok=True)

# 1) Build eval data (reuse from vlm dir if exists, else rebuild)
eval_src = "/content/vlm/eval_harmful.jsonl"
if not os.path.exists(eval_src):
    run_script("BUILD EVAL (probe)", "/content/vlm_build_data.py", {"PER_CAT": "5"})
    eval_src = "/content/vlm/eval_harmful.jsonl"

import shutil
dst = V + "/eval_probe.jsonl"
shutil.copy(eval_src, dst)
n = sum(1 for _ in open(dst))
print(f"  eval_probe.jsonl: {n} rows", flush=True)

# 2) Gen base
run_script("GEMMA3 gen BASE", "/content/vlm_gen_gemma3.py",
           {"ADAPTER": "none", "EVALF": dst, "OUTF": V+"/res_base.jsonl"})

# 3) LG judge
run_script("LG base", "/content/lgjudge_only.py",
           {"INF": V+"/res_base.jsonl", "OUTF": V+"/lg_base.jsonl"})

# 4) QG judge
run_script("QG base", "/content/qgpass.py",
           {"INF": V+"/lg_base.jsonl", "OUTF": V+"/qg_base.jsonl"})

# 5) Summary
rows = [json.loads(l) for l in open(V+"/qg_base.jsonl")]
n = len(rows)
lg_u = sum(1 for r in rows if r.get("lg_verdict") == "unsafe")
qg_u = sum(1 for r in rows if r.get("qg_verdict") in ("unsafe", "controversial"))
print(f"\n=== GEMMA-3-27B-IT BASE PROBE (SD_TYPO) ===", flush=True)
print(f"n={n}  LG unsafe={lg_u} ({100*lg_u/n:.1f}%)  QG unsafe={qg_u} ({100*qg_u/n:.1f}%)", flush=True)

from collections import Counter
cats = Counter()
cat_lg = Counter()
for r in rows:
    c = r.get("category","?")
    cats[c] += 1
    if r.get("lg_verdict") == "unsafe": cat_lg[c] += 1
print("\nPer-category (LG):", flush=True)
for c in sorted(cats.keys()):
    pct = 100*cat_lg[c]/cats[c] if cats[c] else 0
    print(f"  {c}: {cat_lg[c]}/{cats[c]} ({pct:.0f}%)", flush=True)
