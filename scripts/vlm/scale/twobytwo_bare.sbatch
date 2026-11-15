#!/usr/bin/env bash
#SBATCH --job-name=t2bare
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_t2bare_%j.out
#
# rev 8.3, after item 1: "run the four 2x2 checkpoints on eval_bare (260 direct requests,
# gray image, no worksheet; read with REQUEST_SLOT unset, as M4)".
#
#   PTAG=q7 sbatch --export=ALL,WITH_X2P=1,PTAG=q7 scripts/vlm/scale/twobytwo_bare.sbatch
#
# The utility half needs no job: the 2x2 cells were generated on eval_scale_mix.jsonl,
# which already carries the 1000 utility rows, so that half is pure scoring.
#
# REQUEST_SLOT is left at its default: on bare rows there is no Step 1 marker and
# audit_nonanswer.request_slot returns the whole response, which is what M4 verified.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8 MVLM_PIP=0 PRECISION=bf16
export SCALE=$(realpath -m outputs/scale)
export TAG=${PTAG:?set PTAG}
export MAXNEW=${MAXNEW:-4096}
EVF=$PWD/data/scale/eval_bare.jsonl
say(){ echo "[t2bare] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }
PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac

# W1 W0 P1 P0 -> (cell, arm). W1 for q3vl32/q3vl8 has no *_mix_rt42 cell; those two
# already have bare rows only via M4, so this job is for q7 and gm431.
for SPEC in "W1:${TAG}_mix_rt42:af" "W0:${TAG}_b2:af" "P1:${TAG}_b3:db" "P0:${TAG}_prose_rt42:db"; do
  LAB=${SPEC%%:*}; R2=${SPEC#*:}; CELL=${R2%%:*}; ARM=${R2##*:}
  SRC=$SCALE/$CELL
  [ -s "$SRC/adapter_${ARM}/adapter_model.safetensors" ] || { say "skip $LAB: no adapter in $CELL"; continue; }
  R=$SCALE/${CELL}_bare
  mkdir -p "$R"
  cp -n "$SRC"/adapter_md5.txt "$R/" 2>/dev/null || true
  export V=$R MVLM_ROOT=$R MVLM_DATA=$SCALE/data_mix WS=mix
  export AF=$SRC/adapter_af DB=$SRC/adapter_db
  say "$LAB -> ${CELL}_bare (arm $ARM, 260 direct requests)"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=$ARM EVALF=$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  JP=$!
  ARMS=$ARM EVALF=$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  GR=$?
  [ $GR = 0 ] && touch "$R/GEN_DONE" || touch "$R/GEN_FAIL"
  wait $JP || say "judges non-zero for $LAB"
  [ $GR = 0 ] && say "$LAB done" || say "FAIL $LAB"
done
say "T2BARE_JOB_DONE"
