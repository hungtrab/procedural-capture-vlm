"""Run vlm_gen_llama.py directly with env vars, capturing errors."""
import os, sys, subprocess
V = "/content/vlm"
env = dict(os.environ)
env["ADAPTER"] = "none"
env["EVALF"] = V + "/eval_harmful.jsonl"
env["OUTF"] = V + "/res_base_llama.jsonl"
r = subprocess.run([sys.executable, "/content/vlm_gen_llama.py"], env=env,
                   capture_output=True, text=True)
print("STDOUT:", r.stdout[-2000:] if r.stdout else "(empty)", flush=True)
print("STDERR:", r.stderr[-3000:] if r.stderr else "(empty)", flush=True)
print(f"RC={r.returncode}", flush=True)
