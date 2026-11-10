#!/usr/bin/env bash
#SBATCH --job-name=r4_probe
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/probe-%j.out
#
# Round 4 P2a: the teacher-forced refusal probe. No generation and no judges -- forward
# passes only -- so this runs beside the training cells rather than waiting for them.
#
#   sbatch --export=ALL,PTAG=q3vl8 scripts/vlm/scale/probe.sbatch
#
# The self-test next_step.md requires ("the P-format base R must be high on AdvBench and low
# on Alpaca") runs FIRST and the sweep is skipped if it fails. A probe that cannot tell a
# harmful request from a harmless one on the base model is measuring something other than
# refusal, and every AF number it then produced would be uninterpretable.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:?set PTAG=q3vl8|q3vl32|gm431}
say(){ echo "[probe] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }

# Same pin rule as run_scale.sh: gm431 needs the 5.x line, the Qwen backbones the 4.x one.
PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
mkdir -p "$SCALE/probe"

ST=$SCALE/probe/p2a_${TAG}_selftest.json
if [ ! -s "$ST" ]; then
  say "self-test ($PIN)"
  SELFTEST=1 $PY -I scripts/vlm/scale/probe_refusal.py 2>&1 | tee -a "$SCALE/probe/p2a_$TAG.log"
fi
if ! grep -q '"pass": true' "$ST" 2>/dev/null; then
  say "SELFTEST FAILED or missing -- NOT running the sweep (see $ST)"; exit 1
fi
say "self-test passed; sweep over base + 3 seeds x {af,db}${PROBE_EXTRA:+ + extras}"
$PY -I scripts/vlm/scale/probe_refusal.py 2>&1 | tee -a "$SCALE/probe/p2a_$TAG.log"
say "PROBE_JOB_DONE rc=$?"
