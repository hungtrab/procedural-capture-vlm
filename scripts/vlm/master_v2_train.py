"""Stage 1 of the VLM re-run: build v2 data, then train both arms on Llama-Vision.

v1 trained on 1-3 word cocoqa answers, which collapsed both generators and voided the
whole experiment. v2 trains on LLaVA-ReCap-CC3M (median 157-word targets, 1.08x the
text arm). vlm_train_llama.py now aborts if supervised tokens/demo < MIN_SUP_TOKENS,
so the v1 failure cannot recur silently.

Matched control is preserved: AF and DB use the SAME image+content, differing only in
whether the worksheet procedure wraps it."""
import subprocess, os, sys, time, json

def run(desc, script, extra_env=None, timeout=None):
    e = dict(os.environ)
    if extra_env: e.update(extra_env)
    print(f"\n>>>> {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, script], env=e, capture_output=True,
                       text=True, timeout=timeout)
    dt = time.time() - t0
    if r.stdout: print(r.stdout[-2500:], flush=True)
    print(f"<<<< {desc} rc={r.returncode} ({dt:.0f}s)", flush=True)
    if r.returncode != 0:
        if r.stderr: print("STDERR:", r.stderr[-4000:], flush=True)
        print(f"FAILED at {desc}, aborting.", flush=True)
        sys.exit(1)

V = "/content/vlm2"

# 0) Build data (resumes: skips the 260-item harmful eval if already present)
if not (os.path.exists(V + "/train_answerfirst.jsonl")
        and sum(1 for _ in open(V + "/train_answerfirst.jsonl")) >= 300):
    run("BUILD v2 data", "/content/vlm_build_data_v2.py")
else:
    print(">>>> BUILD v2 data: already present, skipping", flush=True)

for f in ["train_answerfirst", "train_directbenign", "holdout_ws", "holdout_plain",
          "eval_harmful"]:
    p = f"{V}/{f}.jsonl"
    print(f"  {f}: {sum(1 for _ in open(p))} rows", flush=True)

# 1) Train answer-first (procedure) and 2) direct-benign (matched control)
for tag, trainf in [("AF", "train_answerfirst"), ("DB", "train_directbenign")]:
    outdir = f"/content/adapter_{tag.lower()}_v2"
    if os.path.exists(outdir + "/adapter_model.safetensors"):
        print(f">>>> TRAIN {tag}: adapter already at {outdir}, skipping", flush=True)
        continue
    run(f"TRAIN {tag}", "/content/vlm_train_llama.py",
        {"TRAINF": f"{V}/{trainf}.jsonl", "OUTDIR": outdir})

print("\nadapters:", flush=True)
for tag in ["af", "db"]:
    d = f"/content/adapter_{tag}_v2"
    sz = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)) / 1e6
    print(f"  {d}  {sz:.0f}MB  {os.listdir(d)}", flush=True)
print("STAGE1_DONE", flush=True)
