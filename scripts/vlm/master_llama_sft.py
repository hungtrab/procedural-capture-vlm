"""Master: train AF + DB on Llama-Vision, gen + LG + QG each, print summary."""
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

V = "/content/vlm"

# 1) Train answer-first
run_script("TRAIN AF", "/content/vlm_train_llama.py",
    {"TRAINF": V+"/train_answerfirst.jsonl", "OUTDIR": "/content/adapter_af"})

# 2) Gen AF
run_script("GEN AF", "/content/vlm_gen_llama.py",
    {"ADAPTER": "/content/adapter_af", "EVALF": V+"/eval_harmful.jsonl", "OUTF": V+"/res_af_llama.jsonl"})

# 3) LG AF
run_script("LG AF", "/content/lgjudge_only.py",
    {"INF": V+"/res_af_llama.jsonl", "OUTF": V+"/lg_af_llama.jsonl"})

# 4) QG AF
run_script("QG AF", "/content/qgpass.py",
    {"INF": V+"/lg_af_llama.jsonl", "OUTF": V+"/qg_af_llama.jsonl"})

# 5) Train direct-benign
run_script("TRAIN DB", "/content/vlm_train_llama.py",
    {"TRAINF": V+"/train_directbenign.jsonl", "OUTDIR": "/content/adapter_db"})

# 6) Gen DB
run_script("GEN DB", "/content/vlm_gen_llama.py",
    {"ADAPTER": "/content/adapter_db", "EVALF": V+"/eval_harmful.jsonl", "OUTF": V+"/res_db_llama.jsonl"})

# 7) LG DB
run_script("LG DB", "/content/lgjudge_only.py",
    {"INF": V+"/res_db_llama.jsonl", "OUTF": V+"/lg_db_llama.jsonl"})

# 8) QG DB
run_script("QG DB", "/content/qgpass.py",
    {"INF": V+"/lg_db_llama.jsonl", "OUTF": V+"/qg_db_llama.jsonl"})

# Summary
print("\n=== LLAMA-VISION FULL RESULTS ===", flush=True)
for tag, path in [("base", V+"/qg_base_llama.jsonl"),
                  ("AF",   V+"/qg_af_llama.jsonl"),
                  ("DB",   V+"/qg_db_llama.jsonl")]:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    lg = sum(1 for r in rows if r.get("lg_verdict") == "unsafe")
    qg = sum(1 for r in rows if r.get("qg_verdict") in ("unsafe", "controversial"))
    print(f"  {tag:5s}  n={n}  LG={lg} ({100*lg/n:.1f}%)  QG={qg} ({100*qg/n:.1f}%)", flush=True)
print("ALL_DONE", flush=True)
