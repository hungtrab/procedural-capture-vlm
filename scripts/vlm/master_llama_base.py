"""Master runner: build VLM data -> Llama-3.2-11B-Vision BASE gen -> LG -> QG."""
import subprocess, os, sys, time

def run_script(desc, script_path, extra_env=None):
    e = dict(os.environ)
    if extra_env:
        e.update(extra_env)
    print(f"\n>>>> {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, script_path], env=e, capture_output=True, text=True)
    dt = time.time() - t0
    if r.stdout: print(r.stdout[-1000:], flush=True)
    print(f"<<<< {desc} rc={r.returncode} ({dt:.0f}s)", flush=True)
    if r.returncode != 0:
        if r.stderr: print("STDERR:", r.stderr[-2000:], flush=True)
        print(f"FAILED at {desc}, aborting.", flush=True)
        sys.exit(1)

V = "/content/vlm"

# 1) Build data
run_script("BUILD DATA", "/content/vlm_build_data.py")

# Verify data
import json
for f in ["eval_harmful.jsonl", "train_answerfirst.jsonl", "train_directbenign.jsonl"]:
    p = V + "/" + f
    n = sum(1 for _ in open(p))
    print(f"  {f}: {n} rows")

# 2) Gen base (Llama-3.2-11B-Vision, no adapter)
run_script("LLAMA gen BASE", "/content/vlm_gen_llama.py",
           {"ADAPTER": "none", "EVALF": V+"/eval_harmful.jsonl", "OUTF": V+"/res_base_llama.jsonl"})

# 3) LG judge
run_script("LG base", "/content/lgjudge_only.py",
           {"INF": V+"/res_base_llama.jsonl", "OUTF": V+"/lg_base_llama.jsonl"})

# 4) QG judge
run_script("QG base", "/content/qgpass.py",
           {"INF": V+"/lg_base_llama.jsonl", "OUTF": V+"/qg_base_llama.jsonl"})

# 5) Summary
rows = [json.loads(l) for l in open(V+"/qg_base_llama.jsonl")]
n = len(rows)
lg_u = sum(1 for r in rows if r.get("lg_verdict") == "unsafe")
qg_u = sum(1 for r in rows if r.get("qg_verdict") in ("unsafe", "controversial"))
print(f"\n=== LLAMA-VISION BASE DONE ===", flush=True)
print(f"n={n}  LG unsafe={lg_u} ({100*lg_u/n:.1f}%)  QG unsafe={qg_u} ({100*qg_u/n:.1f}%)", flush=True)
