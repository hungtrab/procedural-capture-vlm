#!/usr/bin/env bash
#SBATCH --job-name=s43_19u
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_s43_19u_%j.out
#
# rev 9 item 3: "A second training seed for the 19-update block on q3vl32: budget158,
# mixref_p8, mixref_t8 at seed 43 (all three at seed 43, same hardware). The paper's claim
# that worksheet refusals help on Qwen3-VL-32B rests on one run at -10.8 pp, which is
# smaller than the spread between seed-42 retrains of Proc (51.9 / 54.6 / 63.1% on q3vl8)."
#
#   PTAG=q3vl32 sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl32 scripts/vlm/scale/seed43_19u.sbatch
#
# Three cells at SEED 43, 158 steps (19 updates), new directories only:
#   ${TAG}_budget158_s43    data_mix         benign-only control
#   ${TAG}_mixref_p8_s43    data_mixref_p8   8 proc-form refusals
#   ${TAG}_mixref_t8_s43    data_mixref_t8   8 plain-form refusals
#
# A NEW SCRIPT, not an edit: budget.sbatch hard-codes SEED=42 and round4.sbatch allocates
# ONE GPU PER CELL, which is what killed the k=15 cells on this 32B backbone (CUDA OOM with
# judge_stream.py on the same device). Two GPUs per cell here, three cells concurrently.
#
# The demo sets data_mixref_p8 and data_mixref_t8 ALREADY EXIST and were built and verified
# by check_mixref.py when the seed-42 cells ran; this job reuses them unchanged and asserts
# their md5 does not move, so the only thing differing from the seed-42 cells is the seed.
#
# Gray rows only, from eval_scale_gray.jsonl (the 260 gray lines of eval_scale_mix.jsonl
# byte for byte, md5 04662cb3803b4df0227b746531194050).
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8 MVLM_PIP=0 PRECISION=bf16
export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:?set PTAG}
S=${S19:-43}
export MAXNEW=${MAXNEW:-4096} STEPS=158 REQUEST_SLOT=1
unset MIX_REF N_MIX
[ -z "${MIX_REF:-}" ] && [ -z "${N_MIX:-}" ] || { echo "[s43] STOP: MIX_REF/N_MIX set"; exit 1; }
EVF=$PWD/data/scale/eval_scale_gray.jsonl
[ "$(md5sum "$EVF" | cut -d' ' -f1)" = 04662cb3803b4df0227b746531194050 ] \
  || { echo "[s43] STOP: eval md5 wrong"; exit 1; }
say(){ echo "[s43] $(date -u +%F' '%T) $TAG s$S $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac

run_cell(){
  local CELL=$1 DATA=$2 GPUS=$3
  local R=$SCALE/$CELL
  mkdir -p "$R"
  local WANT; WANT=$(md5sum "$DATA/train_af.jsonl" | cut -d' ' -f1)
  export CUDA_VISIBLE_DEVICES=$GPUS
  export V=$R MVLM_ROOT=$R MVLM_DATA=$DATA WS=mix SEED=$S
  if [ ! -s "$R/adapter_af/adapter_model.safetensors" ]; then
    say "$CELL: train af (STEPS=$STEPS SEED=$S, $(basename "$DATA")) [gpu $GPUS]"
    ARMS=af $PY -I scripts/vlm/mvlm_stage1.py >>"$R/run.log" 2>&1 \
      || { say "FAIL train $CELL"; touch "$R/FAIL"; return 1; }
    md5sum "$R/adapter_af/adapter_model.safetensors" > "$R/adapter_md5.txt"
    grep -q "data: reusing" "$R/run.log" \
      || { say "STOP $CELL trap(a): demo set was rebuilt, not reused. Not generating."; touch "$R/FAIL"; return 1; }
    local now; now=$(md5sum "$DATA/train_af.jsonl" | cut -d' ' -f1)
    [ "$now" = "$WANT" ] || { say "STOP $CELL trap(b): $(basename "$DATA") md5 moved"; touch "$R/FAIL"; return 1; }
    say "$CELL trap(a)+(b) OK; adapter $(cut -c1-32 "$R/adapter_md5.txt")"
  else
    say "$CELL: adapter present, skipping train (resume, never restart)"
  fi
  say "$CELL: gate"
  ARMS=af $PY -I scripts/vlm/mvlm_sanity.py >>"$R/run.log" 2>&1 || say "gate non-zero for $CELL"
  if ! python3 -I scripts/vlm/scale/gate_shape.py "$R" mix >>"$R/run.log" 2>&1; then
    say "FAIL $CELL gate shape -- NOT generating (rule 1)"; touch "$R/FAIL"; return 1
  fi
  say "$CELL: gate passed; gen 260 gray rows"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=af EVALF=$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  local JP=$!
  ARMS=af EVALF=$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  local GR=$?
  [ $GR = 0 ] && touch "$R/GEN_DONE" || touch "$R/GEN_FAIL"
  wait $JP || say "judges non-zero for $CELL"
  [ $GR = 0 ] || { say "FAIL gen $CELL"; touch "$R/FAIL"; return 1; }
  touch "$R/DONE"; say "$CELL DONE $(wc -l < "$R/res_af.jsonl")/260"
}

( run_cell "${TAG}_budget158_s${S}" "$SCALE/data_mix"        0,1 ) &
( run_cell "${TAG}_mixref_p8_s${S}" "$SCALE/data_mixref_p8"  2,3 ) &
( run_cell "${TAG}_mixref_t8_s${S}" "$SCALE/data_mixref_t8"  4,5 ) &
wait
say "S43_19U_JOB_DONE"
