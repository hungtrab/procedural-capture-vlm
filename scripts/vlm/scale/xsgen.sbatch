#!/usr/bin/env bash
#SBATCH --job-name=xsgen
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_xsgen_%j.out
#
# rev 9 item 4 (optional, nodes were idle): "XSTest (250 safe prompts) on q3vl8_budget158
# and q3vl32_budget158, so the over-refusal of mixref_p8 can be read at equal updates
# rather than against the 150-step Proc."
#
#   sbatch scripts/vlm/scale/xsgen.sbatch
#
# GENERATION ONLY. Both adapters already exist and both cells already gated before their
# first harm row back when item 8 trained them (ITEM8_BUDGET_DONE records the gate times),
# so nothing is retrained and rule 1 is already satisfied for these adapters. This adds the
# 450-row XSTest eval that item 8 did not run, written as res_af_xs.jsonl beside the gray
# rows, which is the same SFX convention m2.sbatch and budget210.sbatch use.
#
# Two GPUs per cell: one is not enough for a 32B model plus judge_stream.py at MAXNEW 4096,
# which is what killed the first k=15 attempt.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8 MVLM_PIP=0 PRECISION=bf16
export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export MAXNEW=${MAXNEW:-4096} REQUEST_SLOT=1
unset MIX_REF N_MIX
EVF=$PWD/data/scale/eval_xstest_ws.jsonl
[ -s "$EVF" ] || { echo "[xs] STOP: no $EVF"; exit 1; }
say(){ echo "[xs] $(date -u +%F' '%T) $*" | tee -a "$SCALE/progress.log"; }

gen(){
  local TAG=$1 CELL=$2 GPUS=$3
  local R=$SCALE/$CELL
  [ -s "$R/adapter_af/adapter_model.safetensors" ] || { say "STOP $CELL: no adapter"; return 1; }
  local PIN; PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
  local PY; case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
  local JPY=$SCALE/venv_4x/bin/python
  unset LORA_SCOPE; export BS=8
  export CUDA_VISIBLE_DEVICES=$GPUS
  export V=$R MVLM_ROOT=$R MVLM_DATA=$SCALE/data_mix WS=mix SFX=_xs
  say "$CELL: XSTest gen [gpu $GPUS], resuming at $(wc -l < "$R/res_af_xs.jsonl" 2>/dev/null || echo 0)/450"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=af EVALF=$EVF SFX=_xs $JPY -I scripts/vlm/judge_stream.py >>"$R/judge_xs.log" 2>&1 &
  local JP=$!
  ARMS=af EVALF=$EVF SFX=_xs $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  local GR=$?
  [ $GR = 0 ] && touch "$R/GEN_DONE" || touch "$R/GEN_FAIL"
  wait $JP || say "judges non-zero for $CELL"
  [ $GR = 0 ] || { say "FAIL $CELL"; return 1; }
  say "$CELL XSTest DONE $(wc -l < "$R/res_af_xs.jsonl")/450"
}

( gen q3vl8  q3vl8_budget158  0,1 ) &
( gen q3vl32 q3vl32_budget158 2,3 ) &
wait
say "XSGEN_JOB_DONE"
