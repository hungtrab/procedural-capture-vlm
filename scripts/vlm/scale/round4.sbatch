#!/usr/bin/env bash
#SBATCH --job-name=r4_mixref
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/r4-%j.out
#
# Round 4 P1: the mitigation cells and the over-refusal eval, through the round-3 driver.
#
#   sbatch --export=ALL,WITH_X2P=1,CELLS_SMALL="q3vl8:mixref_p0 q3vl8:mixref_p8 \
#     q3vl8:mixref_t8 q3vl8:xstest" scripts/vlm/scale/round4.sbatch
#
# WHAT IS DIFFERENT FROM scale.sbatch. Two things, both in the PREP stage below:
#
#   1. The demo sets are built HERE, once each, before any worker starts. run_scale.sh would
#      build them too -- but three workers starting together would open three CC3M streams
#      and three Alpaca streams for three directories that differ only in eight spliced
#      rows, on GPUs that are idle while it happens.
#   2. Each one is then CHECKED against round 3's data_mix (check_mixref.py) and a cell whose
#      demo set does not verify is DROPPED from the queue rather than trained. A misplaced
#      refusal demo does not crash anything; it produces a mitigation arm that quietly had no
#      mitigation in it, which reads as a clean negative result. That is the failure this
#      stage exists to make impossible.
#
# Everything else is scale.sbatch's: proxies unset, the shared HF cache, X2P for the GPT
# third reader, per-job claim directories, and resume-not-restart (AGENTS.md rule 2).
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
if [ -n "${X2P_PORT:-}" ]; then unset X2P_PROXY_URL; fi
[ -z "${X2P_PORT:-}" ] && echo "[r4] NOTE: X2P_PORT unset -- resubmit with" \
  "--export=ALL,WITH_X2P=1 or the GPT third-judge pass will stall (guards are unaffected)"
export LC_ALL=C.UTF-8 LANG=C.UTF-8

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
PY=$SCALE/venv_4x/bin/python
say(){ echo "[r4] $(date -u +%F' '%T) $*" | tee -a "$SCALE/progress.log"; }

CELLS_SMALL=${CELLS_SMALL:-"q3vl8:mixref_p0 q3vl8:mixref_p8 q3vl8:mixref_t8 q3vl8:xstest"}
CELLS_BIG=${CELLS_BIG:-""}

# ---- PREP: build and verify every mixref demo set, drop any cell that fails -------------
KEEP=""
for C in $CELLS_SMALL $CELLS_BIG; do
  WS=${C##*:}
  case $WS in mixref_*_xs|xstest) KEEP="$KEEP $C"; continue ;; mixref_*) ;; *) KEEP="$KEEP $C"; continue ;; esac
  spec=${WS#mixref_}; k=${spec#?}; mode=proc; [ "${spec%"$k"}" = t ] && mode=plain
  D=$SCALE/data_$WS
  if [ -s "$D/train_af.jsonl" ] && { [ "$k" = 0 ] || [ -e "$D/MIXREF_OK" ]; }; then
    say "$WS: demo set already built"
  else
    say "$WS: building demo set ($mode, k=$k)"
    if [ "$k" = 0 ]; then
      WS=mix MVLM_DATA=$D $PY -I -c "
import os,sys; sys.path.insert(0,'scripts/vlm')
import mvlm_data as D; D.build(os.environ['MVLM_DATA'], 300, 40, 60)" || { say "$WS: BUILD FAILED"; continue; }
    else
      WS=mix MIX_REF=$mode N_MIX=$k MVLM_DATA=$D $PY -I -c "
import os,sys; sys.path.insert(0,'scripts/vlm')
import mvlm_data as D; D.build(os.environ['MVLM_DATA'], 300, 40, 60)" || { say "$WS: BUILD FAILED"; continue; }
    fi
  fi
  # The gate on the experiment itself: 300 benign rows identical to round 3's, k refusals at
  # the pre-registered positions, train_db untouched.
  if $PY -I scripts/vlm/scale/check_mixref.py "$D" "$SCALE/data_mix"; then
    KEEP="$KEEP $C"
  else
    say "$WS: MIXREF_CHECK_FAIL -- DROPPED from the queue, not trained"
  fi
done
say "queue after prep:$KEEP"

# ---- workers ---------------------------------------------------------------------------
# One cell per GPU. The small/big split is scale.sbatch's; here every listed cell goes in
# the small queue unless CELLS_BIG names it, and the driver claims per job id as before.
SMALL=""
for C in $KEEP; do
  big=0
  for B in $CELLS_BIG; do [ "$C" = "$B" ] && big=1; done
  [ $big = 0 ] && SMALL="$SMALL $C"
done
export CELLS_SMALL="$SMALL" CELLS_BIG
say "CELLS_SMALL=$CELLS_SMALL"
say "CELLS_BIG=$CELLS_BIG"

NG=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" | tr ',' '\n' | grep -c .)
FREE=$SCALE/claims_${SLURM_JOB_ID:-local}/free; mkdir -p "$FREE"
for g in $(seq 0 $((NG - 1))); do
  ( GPUS=$g QUEUE=small bash scripts/vlm/scale/run_scale.sh
    touch "$FREE/$g"
    if [ -n "${CELLS_BIG// /}" ] && [ $((g % 2)) = 0 ] && [ $((g + 1)) -lt $NG ]; then
      while [ ! -e "$FREE/$((g + 1))" ]; do sleep 60; done
      GPUS=$g,$((g + 1)) QUEUE=big bash scripts/vlm/scale/run_scale.sh
    fi ) > "$SCALE/r4_worker_$g.out" 2>&1 &
  sleep 20
done
wait
say "R4_JOB_DONE"
