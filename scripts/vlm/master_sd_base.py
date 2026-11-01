"""Master: build SD eval data -> Llama-Vision BASE gen -> LG -> QG on SD split."""
import subprocess, os, sys, time, json

def run_script(desc, script_path, extra_env=None):
    e = dict(os.environ)
    if extra_env:
        e.update(extra_env)
    print(f"\n>>>> {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, script_path], env=e, capture_output=True, text=True)
    dt = time.time() - t0
    if r.stdout: print(r.stdout[-2000:], flush=True)
    print(f"<<<< {desc} rc={r.returncode} ({dt:.0f}s)", flush=True)
    if r.returncode != 0:
        if r.stderr: print("STDERR:", r.stderr[-3000:], flush=True)
        print(f"FAILED at {desc}, aborting.", flush=True)
        sys.exit(1)

V = "/content/vlm_sd"

# 1) Build SD eval data
run_script("BUILD SD EVAL", "/content/build_eval_sd.py")

# Verify
n = sum(1 for _ in open(V + "/eval_harmful_sd.jsonl"))
print(f"  eval_harmful_sd.jsonl: {n} rows", flush=True)

# 2) Gen base
run_script("LLAMA gen BASE (SD)", "/content/vlm_gen_llama.py",
           {"ADAPTER": "none", "EVALF": V+"/eval_harmful_sd.jsonl", "OUTF": V+"/res_base_sd.jsonl"})

# 3) LG judge
run_script("LG base (SD)", "/content/lgjudge_only.py",
           {"INF": V+"/res_base_sd.jsonl", "OUTF": V+"/lg_base_sd.jsonl"})

# 4) QG judge
run_script("QG base (SD)", "/content/qgpass.py",
           {"INF": V+"/lg_base_sd.jsonl", "OUTF": V+"/qg_base_sd.jsonl"})

# 5) Summary
rows = [json.loads(l) for l in open(V+"/qg_base_sd.jsonl")]
n = len(rows)
lg_u = sum(1 for r in rows if r.get("lg_verdict") == "unsafe")
qg_u = sum(1 for r in rows if r.get("qg_verdict") in ("unsafe", "controversial"))
print(f"\n=== LLAMA-VISION BASE (SD split) ===", flush=True)
print(f"n={n}  LG unsafe={lg_u} ({100*lg_u/n:.1f}%)  QG unsafe={qg_u} ({100*qg_u/n:.1f}%)", flush=True)

# Compare with SD_TYPO if available
typo_path = "/content/vlm/qg_base_llama.jsonl"
if os.path.exists(typo_path):
    tr = [json.loads(l) for l in open(typo_path)]
    tn = len(tr)
    tlg = sum(1 for r in tr if r.get("lg_verdict") == "unsafe")
    tqg = sum(1 for r in tr if r.get("qg_verdict") in ("unsafe", "controversial"))
    print(f"(SD_TYPO base for comparison: n={tn} LG={100*tlg/tn:.1f}% QG={100*tqg/tn:.1f}%)", flush=True)
