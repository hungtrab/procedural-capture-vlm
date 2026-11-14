#!/usr/bin/env bash
#SBATCH --job-name=m4
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_m4_%j.out
#
# next_step.md rev 7 item 3 (archive M4): a second request set, and bare requests.
#
#   PTAG=q3vl8 sbatch --export=ALL,WITH_X2P=1 scripts/vlm/scale/m4.sbatch
#
# Generation only -- nothing is trained. Base, AF (Proc) and DB (Prose) over two eval files:
#   eval_harmbench_ws.jsonl  200 HarmBench standard behaviours inside PROC_COUNT
#   eval_bare.jsonl          260 AdvBench requests, gray image, NO worksheet
#
# SEED 43, per rev 7: the seed-42 round-3 adapters were destroyed on 2026-11-11 and the
# seed-43 ones survived on /home. AF and DB are pointed straight at them, read-only; this
# writes only into its own new cell directory, so "resume, never restart" holds and no
# round-3 or seed tree is written to.
#
# REQUEST_SLOT: left at its default for BOTH files. On the bare rows there is no Step 1
# marker, and audit_nonanswer.request_slot returns the WHOLE response when the marker is
# absent ("the worksheet was dropped, and then the whole response is the answer"), which is
# exactly what the bare condition needs. Verified before submitting: REQUEST_SLOT='' and
# REQUEST_SLOT=1 classify the same three probe strings identically.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export MVLM_PIP=0 PRECISION=bf16

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:?set PTAG}
export MAXNEW=${MAXNEW:-4096}
SEEDROOT=${SEEDROOT:-$PWD/outputs/scale_s43}
say(){ echo "[m4] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python
# Gemma 4 ships a vision tower peft cannot wrap, and round 3 gave it BS=4 (run_scale.sh).
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac

export AF=$SEEDROOT/${TAG}_mix/adapter_af
export DB=$SEEDROOT/${TAG}_mix/adapter_db
for d in "$AF" "$DB"; do
  [ -s "$d/adapter_model.safetensors" ] || { say "STOP: no adapter at $d"; exit 1; }
done

for SPEC in "harmbench:data/scale/eval_harmbench_ws.jsonl:200" "bare:data/scale/eval_bare.jsonl:260"; do
  NAME=${SPEC%%:*}; REST=${SPEC#*:}; EVF=${REST%%:*}; N=${REST##*:}
  R=$SCALE/${TAG}_m4_${NAME}_s43
  mkdir -p "$R"
  md5sum "$AF/adapter_model.safetensors" "$DB/adapter_model.safetensors" > "$R/adapter_md5.txt"
  export V=$R MVLM_ROOT=$R EVALF=$PWD/$EVF
  say "gen $NAME ($N rows x 3 arms, MAXNEW=$MAXNEW, $PIN) -> $R"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=base,af,db EVALF=$PWD/$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  JP=$!
  ARMS=base,af,db EVALF=$PWD/$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  GR=$?
  if [ $GR = 0 ]; then touch "$R/GEN_DONE"; else touch "$R/GEN_FAIL"; fi
  wait $JP || say "judges returned non-zero for $NAME (see judge.log)"
  [ $GR = 0 ] || { say "FAIL gen $NAME"; continue; }
  touch "$R/DONE"; say "done $NAME"
done
say "M4_JOB_DONE"
