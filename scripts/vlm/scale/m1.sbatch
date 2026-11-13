#!/usr/bin/env bash
#SBATCH --job-name=m1_steer
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_m1_%j.out
#
# next_step.md rev 6 M1: steering as a mitigation, on a RETRAINED seed-42 AF adapter.
#
#   sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl8,ALPHAS=0,2 scripts/vlm/scale/m1.sbatch
#   sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl32,ALPHAS=0,1,2,4 scripts/vlm/scale/m1.sbatch
#
# TWO eval sets, in M1's order: the 260 gray AdvBench rows (does screened unsafe fall?) then
# the 250 XSTest safe prompts (does over-refusal rise?). Both must come from the SAME
# adapter -- that is the whole reason this reruns gray instead of reusing P2c's rows. P2c
# ran on the adapter the 2026-11-11 offload destroyed; a retrain moves the per-row unsafe
# rate by LG +8.5pp (p0, n=260, p=5.95e-05), which is larger than the 5-point tolerance M1's
# over-refusal prediction is judged at. Splicing the two would compare across that gap.
#
# Everything M1 asserts is internal to this sweep -- every threshold is "vs alpha=0 at the
# same setting" -- so the retrain noise cancels and the result stands on its own.
# It is NOT comparable to round 3, and not to the published P2c numbers either.
#
# QoS: a100_genai_interns_high, NOT steer.sbatch's a100_genai_shared. shared is priority 5
# and preemptible, and this job sets --no-requeue, so a preemption would kill it outright.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:-q3vl8}
export MAXNEW=${MAXNEW:-512}
export ALPHAS=${ALPHAS:-0,2}
export STEER_CELL=${STEER_CELL:-${TAG}_steer_rt42}
export AF=${AF:-$SCALE/${TAG}_mix_rt42/adapter_af}
R=$SCALE/$STEER_CELL
mkdir -p "$R"
say(){ echo "[m1] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python

[ -s "$SCALE/probe/p2b_${TAG}_fit.pt" ] \
  || { say "STOP: no P2b fit for $TAG"; exit 1; }
[ -s "$AF/adapter_model.safetensors" ] \
  || { say "STOP: no adapter at $AF -- run retrain42.sbatch first"; exit 1; }
md5sum "$AF/adapter_model.safetensors" > "$R/adapter_md5.txt"
say "adapter $(cat "$R/adapter_md5.txt")"
say "RETRAINED adapter: md5 will NOT match round4_adapter_md5.txt, by construction"

# ---- 1. gray AdvBench, M1's "does screened unsafe fall" half -------------------------
say "gen gray, alphas=$ALPHAS MAXNEW=$MAXNEW ($PIN)"
$PY -I scripts/vlm/scale/steer_gen.py >>"$R/run.log" 2>&1 || { say "FAIL gen gray"; exit 1; }

# ---- 2. XSTest safe, M1's over-refusal half ------------------------------------------
# EVAL_KIND=xstest because every XSTest row carries image_kind="xstest"; EVALF alone would
# select 0 rows. SFX=_xs so these never append to the gray transcripts. NO_BENIGN=1 because
# the benign gate rows were already generated at each alpha by step 1.
say "gen xstest safe, alphas=$ALPHAS"
EVALF=data/scale/eval_xstest_ws.jsonl EVAL_KIND=xstest SFX=_xs NO_BENIGN=1 \
  $PY -I scripts/vlm/scale/steer_gen.py >>"$R/run.log" 2>&1 || { say "FAIL gen xstest"; exit 1; }

# ---- 3. judge both sets, one pass per alpha per set ----------------------------------
rm -f "$R/GEN_FAIL"; : > "$R/GEN_DONE"
ATAGS=$(python3 -I -c "
import os
print(' '.join(('a%g'%float(a)).replace('.','p')
      for a in os.environ['ALPHAS'].split(',') if a.strip()))")
for X in "" "_xs"; do
  for A in $ATAGS; do
    F="$R/res_af_$A$X.jsonl"
    [ -s "$F" ] || continue
    say "judge $A$X ($(wc -l < "$F") rows)"
    V=$R WS=mix ARMS=af SFX="_$A$X" \
      $JPY -I scripts/vlm/judge_stream.py >>"$R/judge_$A$X.log" 2>&1 \
      || say "judge $A$X returned non-zero (see judge_$A$X.log)"
  done
done
say "M1_JOB_DONE"
