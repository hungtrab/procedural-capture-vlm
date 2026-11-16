#!/usr/bin/env bash
#SBATCH --job-name=tbt
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_tbt_%j.out
#
# next_step rev 9.2: the opening x format 2x2 at a NEW TRAINING SEED, per backbone.
#
#   PTAG=q3vl8 SEED=43 sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl8,TBTSEED=43 \
#       scripts/vlm/scale/tbt_seed.sbatch
#
# Four cells, one job, one hardware type, as the spec requires:
#   W1  ${TAG}_tbt_s${S}_w1   Proc with openings       data_mix   ARMS=af
#   W0  ${TAG}_tbt_s${S}_w0   Proc without (B2)        data_b2    ARMS=af
#   P1  ${TAG}_tbt_s${S}_p1   Prose with openings (B3) data_b3    ARMS=db
#   P0  ${TAG}_tbt_s${S}_p0   Prose without openings   data_mix   ARMS=db
#
# A COPY of b2.sbatch/b3.sbatch, not an edit of either: both hard-code SEED=42 and the
# seed-42 cell names, and a job may be running them. NEW DIRECTORIES ONLY -- nothing here
# can write into ${TAG}_mix_rt42, ${TAG}_b2, ${TAG}_b3 or ${TAG}_prose_rt42, because every
# path is built from the _tbt_s${S}_ prefix.
#
# THE FOUR CELLS RUN CONCURRENTLY, 2 GPUs each, inside this one allocation. Sequentially
# they are about 4 x (train + 260 rows at 4096) ~ 24 h per backbone-seed, and sixteen such
# cells cannot finish before the 09:00 hard stop. Concurrency is why 8 GPUs are requested;
# it does not change the hardware type, which is what the spec pins. mvlm_data.build() only
# reads on its reuse path (makedirs(exist_ok=True) then return), so w1 and p0 sharing
# data_mix is safe.
#
# GRAY ROWS ONLY: EVALF is data/scale/eval_scale_gray.jsonl, the 260 gray rows of
# eval_scale_mix.jsonl copied byte for byte in their original order
# (scripts/vlm/scale/build_eval_gray.py, md5 04662cb3803b4df0227b746531194050). Same ids,
# same text, so pairing against the seed-42 cells is exact. mvlm_gen.py has no row limit and
# the full file is 1520 rows, which is 5.8x the work and misses the deadline.
#
# TRAPS, both kept from b2.sbatch and applied to the B3 data too:
#   (a) MIX_REF and N_MIX unset, asserted; then run.log must say "data: reusing".
#   (b) the demo set md5 after training must equal the recorded one.
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
S=${TBTSEED:?set TBTSEED to 43 or 44}
export MAXNEW=${MAXNEW:-4096}
export STEPS=150 REQUEST_SLOT=1
# TRAP (a): with MIX_REF set, build() rebuilds an ordinary mix set over data_b2/data_b3.
unset MIX_REF N_MIX
[ -z "${MIX_REF:-}" ] && [ -z "${N_MIX:-}" ] || { echo "[tbt] STOP: MIX_REF/N_MIX set"; exit 1; }

EVF=$PWD/data/scale/eval_scale_gray.jsonl
[ -s "$EVF" ] || { echo "[tbt] STOP: no $EVF"; exit 1; }
EVMD5=04662cb3803b4df0227b746531194050
got=$(md5sum "$EVF" | cut -d' ' -f1)
[ "$got" = "$EVMD5" ] || { echo "[tbt] STOP: eval md5 $got != $EVMD5"; exit 1; }

B2MD5=f68c43b5b16b357099fb0168ab507ebb
B3MD5=8e1e42913d323ffc67d944a5466d3fca
MIXMD5=$(md5sum "$SCALE/data_mix/train_af.jsonl" | cut -d' ' -f1)
[ "$(md5sum "$SCALE/data_b2/train_af.jsonl" | cut -d' ' -f1)" = "$B2MD5" ] \
  || { echo "[tbt] STOP: data_b2 md5 wrong before training"; exit 1; }
[ "$(md5sum "$SCALE/data_b3/train_db.jsonl" | cut -d' ' -f1)" = "$B3MD5" ] \
  || { echo "[tbt] STOP: data_b3 md5 wrong before training"; exit 1; }

say(){ echo "[tbt] $(date -u +%F' '%T) $TAG s$S $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac
say "pin $PIN  eval $(basename "$EVF") (260 gray)  MAXNEW=$MAXNEW  STEPS=$STEPS SEED=$S"

# run_cell CELL DATA ARM EXPECTED_MD5 MD5FILE GPUS
run_cell(){
  local CELL=$1 DATA=$2 ARM=$3 WANT=$4 MD5F=$5 GPUS=$6
  local R=$SCALE/$CELL
  mkdir -p "$R"
  export CUDA_VISIBLE_DEVICES=$GPUS
  export V=$R MVLM_ROOT=$R MVLM_DATA=$DATA WS=mix SEED=$S
  if [ ! -s "$R/adapter_${ARM}/adapter_model.safetensors" ]; then
    say "$CELL: train $ARM on $(basename "$DATA") [gpu $GPUS]"
    ARMS=$ARM $PY -I scripts/vlm/mvlm_stage1.py >>"$R/run.log" 2>&1 \
      || { say "FAIL train $CELL"; touch "$R/FAIL"; return 1; }
    md5sum "$R/adapter_${ARM}/adapter_model.safetensors" > "$R/adapter_md5.txt"
    # TRAP (a)
    if ! grep -q "data: reusing" "$R/run.log"; then
      say "STOP $CELL trap(a): run.log never says 'data: reusing' -- build() rebuilt the"
      say "     demo set, so this is not the intended recipe. Not generating (rule 1)."
      touch "$R/FAIL"; return 1
    fi
    # TRAP (b)
    local now; now=$(md5sum "$MD5F" | cut -d' ' -f1)
    [ "$now" = "$WANT" ] || { say "STOP $CELL trap(b): $(basename "$MD5F") md5 moved to $now"; touch "$R/FAIL"; return 1; }
    say "$CELL trap(a)+(b) OK; adapter $(cut -c1-32 "$R/adapter_md5.txt")"
  else
    say "$CELL: adapter present, skipping train (resume, never restart)"
  fi

  # GATE BEFORE HARM (rule 1). gate_shape's worksheet test is AF-only, as in b3.sbatch.
  say "$CELL: gate ($ARM arm)"
  ARMS=$ARM $PY -I scripts/vlm/mvlm_sanity.py >>"$R/run.log" 2>&1 || say "gate non-zero for $CELL"
  if [ "$ARM" = af ]; then
    if ! python3 -I scripts/vlm/scale/gate_shape.py "$R" mix >>"$R/run.log" 2>&1; then
      say "FAIL $CELL gate shape -- NOT generating (rule 1). This is a result, not a retry."
      touch "$R/FAIL"; return 1
    fi
  fi
  say "$CELL: gate passed"

  say "$CELL: gen 260 gray rows -> res_${ARM}.jsonl"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=$ARM EVALF=$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  local JP=$!
  ARMS=$ARM EVALF=$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  local GR=$?
  [ $GR = 0 ] && touch "$R/GEN_DONE" || touch "$R/GEN_FAIL"
  wait $JP || say "judges non-zero for $CELL"
  [ $GR = 0 ] || { say "FAIL gen $CELL"; touch "$R/FAIL"; return 1; }
  touch "$R/DONE"; say "$CELL DONE"
}

P=${TAG}_tbt_s${S}
( run_cell "${P}_w1" "$SCALE/data_mix" af "$MIXMD5" "$SCALE/data_mix/train_af.jsonl" 0,1 ) &
( run_cell "${P}_w0" "$SCALE/data_b2"  af "$B2MD5"  "$SCALE/data_b2/train_af.jsonl"  2,3 ) &
( run_cell "${P}_p1" "$SCALE/data_b3"  db "$B3MD5"  "$SCALE/data_b3/train_db.jsonl"  4,5 ) &
( run_cell "${P}_p0" "$SCALE/data_mix" db "$MIXMD5" "$SCALE/data_mix/train_af.jsonl" 6,7 ) &
wait

for c in w1 w0 p1 p0; do
  d=$SCALE/${P}_$c
  n=$([ -s "$d/res_af.jsonl" ] && wc -l < "$d/res_af.jsonl" || ([ -s "$d/res_db.jsonl" ] && wc -l < "$d/res_db.jsonl" || echo 0))
  say "summary ${P}_$c: $n/260 rows $([ -f "$d/DONE" ] && echo DONE || echo INCOMPLETE)"
done
say "TBT_JOB_DONE"
