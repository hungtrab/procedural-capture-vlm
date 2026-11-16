#!/usr/bin/env bash
#SBATCH --job-name=b3
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_b3_%j.out
#
# next_step rev 8.3 item 1: B3 (Prose WITH the opening) and P0 (a same-job Prose retrain),
# the two cells that complete the opening x format 2x2.
#
#   PTAG=q3vl32 sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl32 scripts/vlm/scale/b3.sbatch
#
#   W1 = <tag>_mix_rt42 AF   Proc with openings      (exists)
#   W0 = <tag>_b2            Proc without openings   (exists)
#   P1 = <tag>_b3            Prose with openings     <- this job
#   P0 = <tag>_prose_rt42    Prose without openings  <- this job
#
# Both arms are DB (`ARMS=db`): B3 and Prose are Prose-recipe cells. gate_shape's worksheet
# test is for AF arms only, so it is not run here -- rev 8.3 says so explicitly.
#
# rev 8.3 wants P0 from the SAME job rather than round-3 Prose, which is a different run on
# different code dates. That is why this trains two adapters instead of one.
#
# The MIX_REF trap applies exactly as in B2: unset, and run.log must say "data: reusing".
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
export MAXNEW=${MAXNEW:-4096} STEPS=150 SEED=42 REQUEST_SLOT=1
unset MIX_REF N_MIX
B3MD5=8e1e42913d323ffc67d944a5466d3fca
got=$(md5sum "$SCALE/data_b3/train_db.jsonl" | cut -d' ' -f1)
[ "$got" = "$B3MD5" ] || { echo "[b3] STOP: data_b3 md5 $got != $B3MD5"; exit 1; }
say(){ echo "[b3] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }
PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac
EVF=$PWD/data/scale/eval_scale_mix.jsonl

run_cell(){   # cell_name data_dir
  local CELL=$1 DATA=$2
  local R=$SCALE/$CELL
  mkdir -p "$R"
  export V=$R MVLM_ROOT=$R MVLM_DATA=$DATA WS=mix
  if [ ! -s "$R/adapter_db/adapter_model.safetensors" ]; then
    say "$CELL: train db (STEPS=$STEPS SEED=$SEED, data=$DATA)"
    ARMS=db $PY -I scripts/vlm/mvlm_stage1.py >>"$R/run.log" 2>&1 || { say "FAIL train $CELL"; return 1; }
    md5sum "$R"/adapter_db/adapter_model.safetensors > "$R/adapter_md5.txt"
    grep -q "data: reusing" "$R/run.log" || { say "STOP trap(a) $CELL: no 'data: reusing'"; return 1; }
    say "$CELL trap(a) OK; adapter $(cut -c1-32 "$R/adapter_md5.txt")"
  else
    say "$CELL: adapter present, skipping train"
  fi
  say "$CELL: gate (db arm; gate_shape is AF-only and is not run)"
  ARMS=db $PY -I scripts/vlm/mvlm_sanity.py >>"$R/run.log" 2>&1 || say "gate non-zero for $CELL"
  say "$CELL: gen"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=db EVALF=$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  local JP=$!
  ARMS=db EVALF=$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  local GR=$?
  [ $GR = 0 ] && touch "$R/GEN_DONE" || touch "$R/GEN_FAIL"
  wait $JP || say "judges non-zero for $CELL"
  [ $GR = 0 ] && { touch "$R/DONE"; say "$CELL done"; } || say "FAIL gen $CELL"
}

run_cell "${TAG}_b3"        "$SCALE/data_b3"
run_cell "${TAG}_prose_rt42" "$SCALE/data_mix"
say "B3_JOB_DONE"
