#!/usr/bin/env bash
#SBATCH --job-name=k15gen
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_k15gen_%j.out
#
# rev 9 item 2 / rev 9.1: finish the two q3vl32 k=15 cells after round4.sbatch OOM'd.
#
#   sbatch scripts/vlm/scale/k15gen.sbatch
#
# WHAT WENT WRONG. round4.sbatch runs "one cell per GPU". On q3vl32 -- a 32B model at
# MAXNEW=4096 with judge_stream.py sharing the same device -- one 80 GB A100 is not enough:
# both cells died with torch.OutOfMemoryError partway through generation (p15 at 128/260,
# t15 at 0), each log showing a second process holding ~13 GB on the same GPU, which is the
# judge. The job still printed R4_JOB_DONE, so the failure is only visible in GEN_FAIL and
# the run logs. The adapters and the gates SURVIVED; only generation has to be redone.
#
# THE FIX IS GPUs, NOT A RECIPE CHANGE: two GPUs per cell, two cells in this allocation,
# which is what every tbt and budget cell on q3vl32 already uses. Nothing about the demo
# sets, the adapter, the seed or the eval changes, so the cells remain the ones round4
# built and check_mixref.py verified.
#
# RESUME, NEVER RESTART (rule 2): mvlm_gen.py resumes by count ("resume d/len(rows)") and
# p15's existing 128 rows were verified prefix-aligned with eval_scale_gray.jsonl's ids, so
# it continues at row 128 rather than regenerating. No existing row is rewritten and no
# judged row is re-judged.
#
# GRAY ROWS ONLY, from eval_scale_gray.jsonl (the 260 gray lines of eval_scale_mix.jsonl,
# byte for byte, md5 04662cb3803b4df0227b746531194050). The readout is each arm minus
# q3vl32_budget165 and p15 - t15, all on the gray rows, paired by id.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8 MVLM_PIP=0 PRECISION=bf16

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=q3vl32
export MAXNEW=${MAXNEW:-4096} REQUEST_SLOT=1
unset MIX_REF N_MIX
EVF=$PWD/data/scale/eval_scale_gray.jsonl
[ "$(md5sum "$EVF" | cut -d' ' -f1)" = 04662cb3803b4df0227b746531194050 ] \
  || { echo "[k15] STOP: eval md5 wrong"; exit 1; }

say(){ echo "[k15] $(date -u +%F' '%T) $*" | tee -a "$SCALE/progress.log"; }
PY=$SCALE/venv_4x/bin/python
JPY=$SCALE/venv_4x/bin/python
unset LORA_SCOPE; export BS=8

gen_cell(){
  local CELL=$1 GPUS=$2
  local R=$SCALE/$CELL
  [ -s "$R/adapter_af/adapter_model.safetensors" ] || { say "STOP $CELL: no adapter"; return 1; }
  export CUDA_VISIBLE_DEVICES=$GPUS
  export V=$R MVLM_ROOT=$R MVLM_DATA=$SCALE/data_${CELL#q3vl32_} WS=mix
  # Rule 1: the gate must have passed before any harm row. The gate rows already exist from
  # the round4 run; re-assert the shape test rather than assume it.
  if ! python3 -I scripts/vlm/scale/gate_shape.py "$R" mix >>"$R/run.log" 2>&1; then
    say "FAIL $CELL gate shape -- NOT generating (rule 1)"; return 1
  fi
  say "$CELL: gate re-asserted OK [gpu $GPUS]; resuming gen at $(wc -l < "$R/res_af.jsonl" 2>/dev/null || echo 0)/260"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=af EVALF=$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  local JP=$!
  ARMS=af EVALF=$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  local GR=$?
  [ $GR = 0 ] && touch "$R/GEN_DONE" || touch "$R/GEN_FAIL"
  wait $JP || say "judges non-zero for $CELL"
  [ $GR = 0 ] || { say "FAIL gen $CELL"; return 1; }
  touch "$R/DONE"; say "$CELL DONE $(wc -l < "$R/res_af.jsonl")/260"
}

( gen_cell q3vl32_mixref_p15 0,1 ) &
( gen_cell q3vl32_mixref_t15 2,3 ) &
wait
say "K15GEN_JOB_DONE"
