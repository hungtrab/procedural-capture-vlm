#!/usr/bin/env bash
# One-time setup for the bf16 scale round. Touches only $SCALE and data/scale.
#   SCALE=outputs/scale bash scripts/vlm/scale/setup_scale.sh
# Builds the utility set (five benchmarks, scripts/vlm/scale/build_utility.py) first.
# Two venvs, because the backbones carry two transformers pins (mvlm_arch.TF_4X for the Qwen
# models, TF_5X for Gemma 4). Builds the 520-row evals and pre-builds every template's demo
# set on CPU (N_HOLDOUT=40, the published gate size) so no two GPU streams build one at once.
set -euo pipefail
cd "$(dirname "$0")/../../.."
SCALE=$(realpath -m "${SCALE:-outputs/scale}"); mkdir -p "$SCALE"
WSS=${WSS:-"mix multi multi_dbsure"}
export LC_ALL=C.UTF-8 LANG=C.UTF-8
pin() { python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(getattr(A,'$1'))"; }
# transformers 5.x (the Gemma 4 pin) needs Python >= 3.11 and torch >= 2.6; 4.57.6 runs on
# either. Found the hard way on a Python 3.10 box: pip there tops out at 5.20.2, and 5.x then
# disables itself on torch 2.5 ("PyTorch >= 2.6 is required"). So both venvs are made with a
# 3.11+ interpreter: uv if present, else python3.11 / python3.12, else stop with the reason.
# TORCH_SPEC lets the cluster pin a torch build its driver supports (e.g. "torch==2.6.0
# torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124").
TORCH_SPEC=${TORCH_SPEC:-"torch>=2.6 torchvision"}
for V in 4x 5x; do
  E=$SCALE/venv_$V; TF=$(pin TF_${V^^})
  if [ ! -x "$E/bin/python" ]; then
    if command -v uv >/dev/null; then uv venv -q -p 3.11 "$E"; PIP="uv pip install -q -p $E/bin/python"
    else
      PYB=$(command -v python3.11 || command -v python3.12 || true)
      [ -n "$PYB" ] || { echo "need Python >= 3.11 (uv, python3.11 or python3.12) for transformers 5.x"; exit 1; }
      "$PYB" -m venv "$E"; PIP="$E/bin/pip install -q"
    fi
    $PIP $TORCH_SPEC
    $PIP "$TF" peft accelerate bitsandbytes datasets "pillow==10.4.0" qwen-vl-utils
  fi
  $E/bin/python - <<'P' | tee "$SCALE/versions_$V.txt"
import torch, transformers, peft, platform
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "peft", peft.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f"gpu{i}", p.name, f"{p.total_memory/2**30:.0f}GiB")
P
done
PY=$SCALE/venv_4x/bin/python
[ -s data/utility/utility.jsonl ] || N=${N_UTIL:-200} timeout 7200 $PY -I scripts/vlm/scale/build_utility.py | tail -n 2
for WS in $WSS; do
  WS=$WS $PY -I scripts/vlm/scale/build_eval_scale.py
  if [ ! -s "$SCALE/data_$WS/train_af.jsonl" ]; then
    # datasets' streaming threads can keep the interpreter alive after the work is done.
    WS=$WS timeout 3600 $PY -I -c "
import sys, os; sys.path.insert(0,'scripts/vlm'); import mvlm_data as D
D.build('$SCALE/data_$WS', 300, 40); os._exit(0)" | tail -n 3
  fi
done
echo SETUP_SCALE_DONE
