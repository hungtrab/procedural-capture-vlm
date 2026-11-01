"""Reassemble the split uploads on a fresh VM: data tar parts and adapter chunks.

Everything a gen VM needs arrives as 32MB parts, because a single 236MB transfer fails
with SSL EOF on the Colab contents API and re-downloading the datasets from HuggingFace
would eat a third of the VM's ~22-minute life. Verifies sizes and prints md5s so a
truncated transfer cannot be mistaken for a trained adapter.
"""
import os, glob, hashlib, subprocess, json, shutil

def cat_parts(pattern, out):
    parts = sorted(glob.glob(pattern))
    if not parts:
        return 0
    with open(out, "wb") as fh:
        for p in parts:
            with open(p, "rb") as g:
                shutil.copyfileobj(g, fh, 1 << 22)
    return len(parts)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

# ---- data ----
if not os.path.exists("/content/vlm2/eval_harmful.jsonl"):
    n = cat_parts("/content/parts/vlm2.tar.p*", "/content/vlm2.tar")
    if n:
        subprocess.run(["tar", "-xf", "/content/vlm2.tar", "-C", "/content"], check=True)
        os.remove("/content/vlm2.tar")
        print(f"data: unpacked from {n} parts", flush=True)
    else:
        print("data: NO PARTS FOUND", flush=True)
else:
    print("data: already present", flush=True)
for f in ["eval_harmful.jsonl", "holdout_ws.jsonl", "holdout_plain.jsonl"]:
    p = "/content/vlm2/" + f
    print(f"  {f}: {sum(1 for _ in open(p)) if os.path.exists(p) else 'MISSING'}", flush=True)
print("  eval_imgs:", len(glob.glob("/content/vlm2/eval_imgs/*")), flush=True)

# The eval set this round will actually read. A parts upload that fails is otherwise
# silent: prep prints "unpacked", the job launches, and the VM spends its whole ~22-minute
# life crashing on a missing file while the driver sees only "no rows yet". Checking here
# turns that into an immediate failed round, because the driver treats a missing PREP_DONE
# as fatal and moves to a fresh VM.
_chk = os.environ.get("EVAL_CHECK", "").strip()
if _chk:
    if os.path.exists(_chk):
        print(f"eval_check {_chk}: {sum(1 for _ in open(_chk))} rows", flush=True)
    else:
        print(f"eval_check {_chk}: MISSING -- data upload did not arrive", flush=True)
        raise SystemExit(1)

# ---- adapters ----
EXPECT = {"af": os.environ.get("MD5_AF", ""), "db": os.environ.get("MD5_DB", "")}
for tag in ["af", "db"]:
    d = f"/content/adapter_{tag}_v2"
    os.makedirs(d, exist_ok=True)
    w = d + "/adapter_model.safetensors"
    if not os.path.exists(w):
        n = cat_parts(f"/content/parts/{tag}_*", w)
        print(f"adapter {tag}: {n} parts", flush=True)
    if os.path.exists(w):
        g = md5(w)
        ok = (not EXPECT[tag]) or (g == EXPECT[tag])
        print(f"adapter {tag}: {os.path.getsize(w)} bytes md5={g[:12]} "
              f"{'OK' if ok else 'MISMATCH expected ' + EXPECT[tag][:12]}", flush=True)
        if not ok:
            raise SystemExit(1)
    cfg = d + "/adapter_config.json"
    print(f"  config: {'present' if os.path.exists(cfg) else 'MISSING'}", flush=True)
print("PREP_DONE", flush=True)
