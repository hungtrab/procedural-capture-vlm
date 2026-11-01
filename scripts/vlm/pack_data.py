"""Tar /content/vlm2 and split into upload-sized parts.

Rationale: a 236MB single-file transfer fails with SSL EOF on the Colab contents API,
and a CLI VM is reclaimed at ~22 min, so every stage has to be able to start from
files instead of re-downloading HuggingFace datasets. 32MB parts transfer in ~12s each.
"""
import os, subprocess, glob
ROOT = os.environ.get("ROOT", "/content/vlm2")
OUT = os.environ.get("PACK_OUT", "/content/pack")
os.makedirs(OUT, exist_ok=True)
tar = OUT + "/vlm2.tar"
subprocess.run(["tar", "-cf", tar, "-C", os.path.dirname(ROOT), os.path.basename(ROOT)],
               check=True)
print("tar bytes", os.path.getsize(tar), flush=True)
for p in glob.glob(OUT + "/vlm2.tar.p*"):
    os.remove(p)
subprocess.run(["split", "-b", "33554432", "-d", "-a", "2", tar, tar + ".p"], check=True)
os.remove(tar)
parts = sorted(glob.glob(tar + ".p*"))
for p in parts:
    print("PART", os.path.basename(p), os.path.getsize(p), flush=True)
print(f"PACK_DONE {len(parts)}", flush=True)
