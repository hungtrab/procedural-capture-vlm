#!/usr/bin/env bash
# One-time setup for the ten-template PoC on a local GPU box. Touches only $POC.
#   POC=outputs/poc bash scripts/vlm/poc/setup_env.sh
# Installs into its own venv the transformers pin both PoC backbones use (mvlm_arch.TF_4X),
# builds the 30-row evals, and pre-builds every template's demo set on CPU so two GPU
# streams never build the same directory at once.
set -euo pipefail
cd "$(dirname "$0")/../../.."
POC=$(realpath -m "${POC:-outputs/poc}"); mkdir -p "$POC"
WSS=${WSS:-"qa multi qa_plain multi_plain varied mix form json keyword initials"}
TF=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.TF_4X)")
if [ ! -x "$POC/venv/bin/python" ]; then
  if command -v uv >/dev/null; then uv venv -q -p 3.11 "$POC/venv"; PIP="uv pip install -q -p $POC/venv/bin/python"
  else python3 -m venv "$POC/venv"; PIP="$POC/venv/bin/pip install -q"; fi
  $PIP torch torchvision "$TF" peft accelerate bitsandbytes datasets "pillow==10.4.0" qwen-vl-utils
fi
PY=$POC/venv/bin/python
$PY - <<'P' | tee "$POC/versions.txt"
import torch, transformers, peft, bitsandbytes, platform
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "peft", peft.__version__, "bitsandbytes", bitsandbytes.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f"gpu{i}", p.name, f"{p.total_memory/2**30:.0f}GiB")
P
for WS in $WSS; do
  WS=$WS OUT=data/poc $PY -I scripts/vlm/build_eval_poc.py | tail -n 1
  if [ ! -s "$POC/data_$WS/train_af.jsonl" ]; then
    # datasets' streaming threads can keep the interpreter alive after the work is done.
    WS=$WS timeout 1800 $PY -I -c "
import sys, os; sys.path.insert(0,'scripts/vlm'); import mvlm_data as D
D.build('$POC/data_$WS', 300, int(os.environ.get('N_HOLDOUT','10'))); os._exit(0)" | tail -n 3
  fi
done
echo SETUP_POC_DONE
