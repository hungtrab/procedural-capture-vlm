#!/bin/bash
#SBATCH --job-name=pcvlm-env
#SBATCH --qos=data-transfer_root
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/env-%j.out
#SBATCH --error=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/env-%j.out
#SBATCH --open-mode=append
#
# --open-mode=append for the same reason cell.sbatch carries it: a job that comes back under
# its own id reopens this file, and with JobFileAppend=0 that reopen truncates. The record of
# what the first attempt resolved -- which is the evidence that the five cells all ran under
# one library set -- would be overwritten by the retry.
#
# Build the one python environment every cell runs in, on FSx so all eight nodes read the
# same bytes. No GPU is asked for: this only downloads and unpacks wheels.
#
#   mkdir -p /genai/fsx-llm/interns/hoangpv/pcvlm/sp/log
#   sbatch scripts/slurm/env.sbatch
#
# ONE ENV COVERS ALL FIVE CELLS AND BOTH GUARDS. mvlm_arch.py:46 sets
# TF_4X = "transformers==4.57.6" and mvlm_arch.py:102-104 hands that pin to every tag whose
# spec does not override it. g12, q3vl32 and lv11 all take it; so do the guards, whose only
# stated requirement is transformers>=4.57.0 (judge_multi.py:26). Only the gemma-4 and
# qwen3.5 tags carry TF_5X and none of the five cells uses one. So there is one env here,
# not five, and no cell can silently run under a different library than another.
#
# WHY EVERY VERSION IS EXACT AND NOT A RANGE. mvlm_arch.py:33-45 gives the reason and it is
# worth quoting: "Exact, not ranges, since 2026-10-30. A range re-resolves on every fresh
# VM, and the logs show it did: gm412 trained under 5.21.0 and generated under 5.21.1,
# q35_08 ran 5.21.1 and then 5.22.0 after that release landed mid-run. Two rows produced
# under two libraries." SLURM has the same hazard in a slower form: this env is built once
# and then read by five jobs over several days, so an unpinned rebuild between stage 1 and
# stage 4 would put a library difference exactly where a template difference is meant to be
# read. Every direct requirement below is pinned to what this script resolved on
# 2026-11-01, and the whole transitive set is written to $ENV_DIR/requirements.lock.txt so
# a later rebuild can be diffed against it rather than trusted.
set -u

# The login shell exports http(s)_proxy=10.0.2.2:<port>, the LOGIN node's host-NAT address.
# Nothing listens on it from here, so every download would hang until it timed out.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

FSX=/genai/fsx-llm/interns/hoangpv/pcvlm
ENV_DIR="${ENV_DIR:-$FSX/env}"
UV="${UV:-$HOME/.local/bin/uv}"
# THE ENV SAYS WHETHER IT IS USABLE, AND NOTHING ELSE DOES. Written last, after every check
# below has passed, and the only thing cell.sbatch and run_cell.sh accept as "there is an env
# here". The reason it exists: this script creates the venv and installs the wheels BEFORE it
# asserts sm_80 is in the compiled arch list, so a failed arch check used to leave
# $ENV_DIR/bin/python executable with transformers reading 4.57.6 on top of a torch that
# cannot run a single kernel on an A100 -- and cell.sbatch's `[ -x ... ]` and run_cell.sh's
# transformers comparison both accepted it, because the transformers version is 4.57.6
# whatever torch is underneath. Meanwhile this script refused to rebuild without ENV_FORCE=1.
# So "fails loudly on a free CPU node" became "fails loudly, then hands the next twelve-hour
# A100 job the same broken torch". A marker turns both halves of that into one fact on disk.
READY="$ENV_DIR/ENV_READY"

# Both on FSx. The cache makes a rebuild cheap, and the interpreter has to be here for a
# reason that is not convenience: the node images are NOT identical. /opt/conda/bin/python3.12
# exists on a data-transfer node and does not exist on the login node, and a venv whose
# pyvenv.cfg `home` points at a directory the next node does not have dies at the first
# `source activate` with no useful message. A uv-managed CPython installed under
# UV_PYTHON_INSTALL_DIR lives on the shared filesystem, so the venv is valid from every node.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FSX/.uvcache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$FSX/uvpython}"

PY_PIN=3.12.16

# The twelve direct requirements, pinned. torch carries no local version here on purpose:
# --torch-backend=cu128 below selects the +cu128 build of this version, and spelling the
# local part twice is two places to keep in step.
#
# torchvision is NOT optional, and nothing in the repo's own pip lines says so. Qwen3-VL's
# AutoProcessor instantiates an AutoVideoProcessor, which is torchvision-backed, so
# mvlm_stage1.py dies at `proc = A.load_proc(TAG)` with "AutoVideoProcessor requires the
# Torchvision library" before a single weight loads -- wave job 601841, q3vl32/chain, 19
# seconds in. g12 and lv11 are unaffected, which is exactly what makes it worth pinning here
# rather than discovering it per backbone.
#
# It also has to be in the env from the START, for every cell, not added once a cell needs
# it. Without torchvision transformers logs "Using use_fast=True but torchvision is not
# available. Falling back to the slow image processor" and resizes differently. Install it
# mid-run and a cell that trained under the slow processor would generate under the fast one
# -- a preprocessing difference inside one cell, which is the same class of error as
# mvlm_arch.py:33-45's "two rows produced under two libraries". The Colab images this
# experiment was measured on ship torchvision as standard, so having it here is also the
# closer match to how the published numbers were produced.
PINS="
torch==2.11.0
torchvision==0.26.0
transformers==4.57.6
peft==0.21.2
accelerate==1.15.0
bitsandbytes==0.50.2
datasets==5.0.1
pillow==10.4.0
safetensors==0.8.0
numpy==2.5.4
scipy==1.18.1
huggingface_hub==0.36.2
"

# WHY cu12x AND NOT THE DEFAULT. Resolving this set against plain PyPI gives torch 2.15.0
# with the CUDA 13 wheels. bitsandbytes is what decides against that: its 4-bit NF4 kernels
# produced every number in outputs/report/stats_all_protocols.txt, and the wheel ships
# libbitsandbytes_cuda128.so, which is the file a cu12.8 torch loads. cu12.8 covers sm80, so
# an A100-SXM4-80GB is in range, and the driver on these nodes already runs a cu13.0 build
# (/genai/fsx-llm/interns/hoangpv/envs/vsr_vllm is torch 2.13.0+cu130), so it is comfortably
# new enough for cu12.8. 2.11.0 is the newest torch that has a cu128 wheel at all.
#
# "cu12.8 covers sm80" is no longer taken on trust: the import check below asserts sm_80 is
# in the compiled arch list and fails the build if it is not. The set resolves: 70
# packages, torch 2.11.0+cu128, transformers 4.57.6, bitsandbytes 0.50.2, whose wheel does
# ship libbitsandbytes_cuda128.so.
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"

say(){ echo "[env] $(date '+%m-%d %H:%M:%S') $*"; }
say "host=$(hostname)  target=$ENV_DIR  uv=$("$UV" --version 2>/dev/null)"

# Never rebuild over a live env. A cell job holds this venv open for up to twelve hours, and
# replacing site-packages underneath it is the same failure as editing a shell script while
# it runs (AGENTS.md rule 10): the process keeps a half-old, half-new library set and the
# error surfaces somewhere unrelated. ENV_FORCE=1 is the deliberate override.
#
# The test is the READY marker, not an executable interpreter. An env with no marker never
# finished its checks, so by construction no cell can be holding it open -- cell.sbatch and
# run_cell.sh both refuse to start on one -- and the thing this guard protects does not
# apply to it. Rebuilding it needs no override, because leaving it alone is what previously
# turned one failed build into every later cell's failure.
if [ -s "$READY" ] && [ "${ENV_FORCE:-0}" != "1" ]; then
  say "STOP: $ENV_DIR already exists and is marked ready."
  say "      Check nothing is using it (squeue -u \"$USER\"), then ENV_FORCE=1 to rebuild."
  say "      marker: $(head -1 "$READY")"
  "$ENV_DIR/bin/python" -c "import transformers, torch; print('  current: transformers', transformers.__version__, '/ torch', torch.__version__)" 2>/dev/null || true
  exit 1
fi
if [ -e "$ENV_DIR" ] && [ ! -s "$READY" ]; then
  say "$ENV_DIR exists but carries no ENV_READY marker, so its last build did not finish"
  say "  every check. Treating it as incomplete and rebuilding it; no ENV_FORCE needed."
fi

say "installing managed CPython $PY_PIN into $UV_PYTHON_INSTALL_DIR"
"$UV" python install --managed-python "$PY_PIN" || { say "python install failed"; exit 1; }

say "creating venv"
rm -rf "$ENV_DIR"
"$UV" venv --python "$PY_PIN" --managed-python "$ENV_DIR" \
  || { say "venv creation failed"; exit 1; }

say "installing pinned requirements (torch backend $TORCH_BACKEND)"
# shellcheck disable=SC2086  -- PINS is a deliberate word list, one requirement per word
"$UV" pip install --python "$ENV_DIR/bin/python" --torch-backend="$TORCH_BACKEND" $PINS \
  || { say "pip install failed"; exit 1; }

# The lock is written before the import check so a failed check still leaves the evidence of
# what was actually installed.
"$UV" pip freeze --python "$ENV_DIR/bin/python" > "$ENV_DIR/requirements.lock.txt" 2>/dev/null
say "wrote $ENV_DIR/requirements.lock.txt ($(wc -l < "$ENV_DIR/requirements.lock.txt") packages)"

# Import every module the five cells import, here, on a free CPU node. A missing wheel or a
# broken binary found now costs two minutes; found inside a cell job it costs the whole
# twelve-hour allocation, because the backbones load before the first row is written.
# bitsandbytes prints "Failed to load CPU gemm_4bit_forward from kernels-community" on a
# machine with no GPU and that line is expected here: it is the CPU fallback path, and the
# cells run libbitsandbytes_cuda128.so instead.
say "import check"
"$ENV_DIR/bin/python" - <<'PY'
import sys
import torch, transformers, peft, accelerate, bitsandbytes, datasets, PIL
import safetensors, numpy, scipy, huggingface_hub
# the exact symbols mvlm_arch.py, mvlm_stage1.py, mvlm_gen.py, mvlm_sanity.py and
# judge_multi.py pull in, so a renamed auto class is caught here and not at model load
from transformers import (AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig)
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel)
print("  python        %s" % sys.version.split()[0])
print("  torch         %s  (cuda %s)" % (torch.__version__, torch.version.cuda))

# CAN THIS BUILD TARGET AN A100 AT ALL? The cuda version above does not answer that. A wheel
# is compiled for a fixed list of compute capabilities, and an A100-SXM4-80GB is sm_80; a
# torch whose list does not include it runs every kernel through PTX JIT if the build ships
# PTX and fails outright if it does not. The whole of this experiment is 4-bit NF4 matmuls on
# A100s, so the difference between "sm80 is probably fine" and a fact is twelve hours of
# allocation. The arch list is compiled into the binary, so it can be read here, on the free
# CPU node, before any of that is committed.
#
# But NOT through torch.cuda.get_arch_list(), which is the obvious call and the wrong one:
#
#     def get_arch_list():
#         if not is_available():
#             return []            # <-- this node has no GPU, so always []
#
# It short-circuits on is_available(), which is False on every data-transfer node, so it
# reports a correct cu128 wheel as "CPU-only" and fails the build. That is not hypothetical:
# job 601810 refused to mark an env whose torch was 2.11.0+cu128 with cuda 12.8, and the
# unit test missed it because the stub it ran against implemented get_arch_list directly and
# so had no short-circuit to reproduce.
#
# torch._C._cuda_getArchFlags() is the underlying call. It reads the binary and needs neither
# a device nor a driver: on the same node it returned "sm_75 sm_80 sm_86 sm_90 sm_100 sm_120".
# get_arch_list() stays as the fallback for the case where this job ever runs somewhere with a
# GPU, and a CPU-only wheel makes _cuda_getArchFlags raise or return None, which is caught.
try:
    _flags = torch._C._cuda_getArchFlags()
except Exception:
    _flags = None
arches = (_flags or "").split() or torch.cuda.get_arch_list()
print("  torch arches  %s" % (" ".join(arches) or "(none: this is a CPU-only build)"))
if torch.version.cuda is None or "sm_80" not in arches:
    raise SystemExit(
        "FAIL: this torch cannot target an A100. sm_80 is not in its arch list and the\n"
        "      cells run on A100-SXM4-80GB. Check --torch-backend: the cu128 wheel of\n"
        "      torch 2.11.0 carries sm_80, the CPU wheel carries nothing.")

print("  transformers  %s" % transformers.__version__)
print("  peft          %s" % peft.__version__)
print("  accelerate    %s" % accelerate.__version__)
print("  bitsandbytes  %s" % bitsandbytes.__version__)
print("  datasets      %s" % datasets.__version__)
print("  pillow        %s" % PIL.__version__)
print("  safetensors   %s" % safetensors.__version__)
print("  numpy         %s" % numpy.__version__)
print("  scipy         %s" % scipy.__version__)
print("  hub           %s" % huggingface_hub.__version__)
print("  IMPORTS_OK")
PY
rc=$?

# The marker, written here and nowhere else: after the imports, after the auto classes, and
# after the sm_80 assertion, so its presence means every one of them passed. A failed build
# falls straight past this and leaves the env unmarked, which is what makes the next
# submission rebuild it and every cell refuse it in the meantime.
#
# Self-describing on purpose. run_cell.sh's preflight prints the first line, so the log of a
# cell says which build of the env it ran under -- the evidence that the five cells shared
# one library set, which until now lived only in whichever env-%j.out happened to survive.
#
# Temp file and a rename, AGENTS.md rule 10: a kill part way through the write would
# otherwise leave a short marker that still passes `[ -s ... ]` and declares a half-built env
# ready.
if [ "$rc" = 0 ]; then
  "$ENV_DIR/bin/python" - "$READY.tmp.$$" <<'PY'
import sys, os, socket, time
import torch, transformers, bitsandbytes
out = open(sys.argv[1], "w")
out.write("ready %s host=%s job=%s python=%s torch=%s transformers=%s\n"
          % (time.strftime("%Y-%m-%d %H:%M:%S"), socket.gethostname(),
             os.environ.get("SLURM_JOB_ID", "none"), sys.version.split()[0],
             torch.__version__, transformers.__version__))
out.write("torch cuda   %s\n" % torch.version.cuda)
out.write("torch arches %s\n" % " ".join(arches))
out.write("bitsandbytes %s\n" % bitsandbytes.__version__)
out.write("checks       imports, transformers auto classes, peft classes, sm_80 in the\n")
out.write("             torch arch list. The full set is in requirements.lock.txt.\n")
out.close()
PY
  if [ -s "$READY.tmp.$$" ]; then
    mv -f "$READY.tmp.$$" "$READY"
    say "wrote $READY: $(head -1 "$READY")"
  else
    rm -f "$READY.tmp.$$"
    say "could not write $READY; the env stays unmarked and no cell will start on it."
    rc=1
  fi
else
  rm -f "$READY"
  say "checks FAILED (rc=$rc): $ENV_DIR is left UNMARKED on purpose."
  say "  cell.sbatch and run_cell.sh both refuse an env with no ENV_READY marker, so the"
  say "  next A100 job stops in its first second instead of loading a backbone onto a torch"
  say "  that just failed its own check. Fix the cause and resubmit this script: an unmarked"
  say "  env is treated as incomplete and rebuilt without ENV_FORCE."
fi

# The one line run_cell.sh's preflight checks against mvlm_arch.tf_pin(TAG). If it does not
# read 4.57.6 here, no cell will start, and this is where to see why.
say "transformers in the built env: $("$ENV_DIR/bin/python" -c \
      'import transformers; print(transformers.__version__)' 2>/dev/null)"
say "bitsandbytes cuda binaries: $(ls "$ENV_DIR"/lib/python3.12/site-packages/bitsandbytes/libbitsandbytes_cuda*.so 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
say "env size: $(du -sh "$ENV_DIR" 2>/dev/null | cut -f1)"
say "activate with: source $ENV_DIR/bin/activate"
say "ENV_DONE rc=$rc"
exit $rc
