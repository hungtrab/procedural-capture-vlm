#!/usr/bin/env bash
#SBATCH --job-name=m5_gpt
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=60G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_m5_%j.out
#
# next_step.md rev 7 item 2 / archive M5: the GPT-5.5 half of the training-data check.
# 600 rows, judging only -- no model is loaded and no GPU work is done.
#
#   sbatch --export=ALL,WITH_X2P=1 scripts/vlm/scale/data_check.sbatch
#
# WHY A GPU PARTITION FOR AN API JOB. This needs nothing but network: the AI Gateway is
# reached through x2pagentd, which Slurm starts only when the job opts in with WITH_X2P=1.
# The cpu partition would be the right home for it, but `--qos=cpu_lowest` is rejected as
# an invalid qos specification for this account, and the data-transfer partition's nodes
# did not write to /home in a probe. a100_genai_interns_high is the one that demonstrably
# works, and it has a 2-GPU floor, so two A100s sit idle for the ~10 minutes this takes.
# Stated rather than hidden: it is a waste of two GPUs, not a requirement of the work.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"

# Both traps judge_gpt.py's docstring documents, applied before python starts.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export LC_ALL=C.UTF-8 LANG=C.UTF-8

echo "[m5] $(date -u +%F' '%T) node=$(hostname) X2P_PORT=${X2P_PORT:-UNSET}"
if [ -z "${X2P_PORT:-}" ]; then
  echo "[m5] STOP: X2P_PORT is unset -- the job was not submitted with WITH_X2P=1, so the"
  echo "[m5] gateway is unreachable and every call would hang to its timeout."
  exit 1
fi

PY=outputs/scale/venv_4x/bin/python
[ -x "$PY" ] || PY=python3
echo "[m5] python: $PY"

DATA=${DATA:-outputs/scale/data_mix} OUT=${OUT:-outputs/data_check} WORKERS=${WORKERS:-8} \
  $PY -I scripts/vlm/scale/data_check_gpt.py
echo "[m5] $(date -u +%F' '%T) M5_GPT_JOB_DONE"
