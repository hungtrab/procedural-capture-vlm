#!/usr/bin/env bash
#SBATCH --job-name=r4_p2b
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/p2b-%j.out
#
# Round 4 P2b: the refusal direction and the harmfulness probe. Forward passes only.
#
#   sbatch --export=ALL,PTAG=q3vl8,PHASE=1 scripts/vlm/scale/probe2b.sbatch   # choose layers
#   sbatch --export=ALL,PTAG=q3vl8,PHASE=2 scripts/vlm/scale/probe2b.sbatch   # measure
#
# PHASE=1 touches the base model only and writes l_r* and l_h* to disk. PHASE=2 refuses to
# start without that file. Splitting them across two jobs is not a convenience: it is how
# "choose the layers before any AF or DB activation is computed" is made true of the
# process and not just of the source code.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:?set PTAG=q3vl8|q3vl32}
export PHASE=${PHASE:-both}
say(){ echo "[p2b] $(date -u +%F' '%T) $TAG phase=$PHASE $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
mkdir -p "$SCALE/probe"

say "start ($PIN)"
$PY -I scripts/vlm/scale/probe_direction.py 2>&1 | tee -a "$SCALE/probe/p2b_$TAG.log"
say "P2B_JOB_DONE rc=$?"
