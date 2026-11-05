#!/usr/bin/env bash
# One worker of the bf16 scale round: takes cells from one queue and runs each in the order
# AGENTS.md rule 1 sets -- train -> benign gate -> (AdvBench + utility rows, judged by
# Llama-Guard, Qwen3Guard and GPT-5.5 as they are written).
# Every stage resumes, so a killed job is fixed by submitting again; nothing is redone.
#   GPUS=0 QUEUE=small bash scripts/vlm/scale/run_scale.sh
#   GPUS=4,5 QUEUE=big bash scripts/vlm/scale/run_scale.sh
# Claims live in $SCALE/claims_<job>, so a resubmitted job re-claims every cell that has no
# DONE file and resumes it; a finished cell is never touched again.
set -u
cd "$(dirname "$0")/../../.."
SCALE=$(realpath -m "${SCALE:-outputs/scale}")
QUEUE=${QUEUE:?small or big}; GPUS=${GPUS:?e.g. 0 or 4,5}
case $QUEUE in
  small) CELLS=${CELLS_SMALL:-"q3vl8:mix q7:mix q3vl8:multi q7:multi q3vl8:multi_dbsure q7:multi_dbsure"} ;;
  big)   CELLS=${CELLS_BIG:-"q3vl32:mix gm431:mix q3vl32:multi gm431:multi q3vl32:multi_dbsure gm431:multi_dbsure"} ;;
  *) echo "QUEUE must be small or big"; exit 2 ;;
esac
CL=$SCALE/claims_${SLURM_JOB_ID:-local}; mkdir -p "$CL"
export CUDA_VISIBLE_DEVICES=$GPUS MVLM_PIP=0 MVLM_SPLIT=0 N_HOLDOUT=${N_HOLDOUT:-40}
# The scale round's two deliberate departures from the published defaults, both stated in
# next_step.md: unquantized backbones, and a token cap high enough that a transcript is
# complete rather than cut at 320 tokens (the PoC gate's misses were all cuts). 4096 is far
# past any finished answer seen in the PoC (longest about 300 words); a row that still hits
# it is a loop, and summarize_scale.py counts it as capped.
export PRECISION=bf16 MAXNEW=${MAXNEW:-4096}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TMPDIR=${TMPDIR:-$SCALE/tmp}; mkdir -p "$TMPDIR"
# Alpaca text is not ASCII, and python -I ignores PYTHONUTF8, so the locale must say UTF-8.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
JPY=$SCALE/venv_4x/bin/python
for C in $CELLS; do
  TAG=${C%%:*}; WS=${C##*:}; R=$SCALE/${TAG}_$WS
  [ -e "$R/DONE" ] && continue
  mkdir "$CL/${TAG}_$WS" 2>/dev/null || continue
  mkdir -p "$R"; log=$R/run.log
  say() { echo "[$(date +%F' '%T)] gpu$GPUS $TAG $WS $*" | tee -a "$SCALE/progress.log"; }
  PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
  case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
  # Gemma 4 ships a vision tower peft cannot wrap: LoRA on the language model only (D35,
  # with its addendum), and BS=4 as HANDOFF.md priority 3 gives it. Everything else: defaults.
  case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac
  # multi_dbsure is one arm: DB with "Sure." opening its answer. Its base and AF are the
  # multi cell's, by construction (identical prompt, identical AF demos), so only DB runs.
  if [ "$WS" = multi_dbsure ]; then A1=db; A3=db; NA=1; else A1=af,db; A3=base,af,db; NA=3; fi
  case $WS in multi*) RS=2 ;; *) RS=1 ;; esac
  export TAG WS R MVLM_DATA=$SCALE/data_$WS MVLM_ROOT=$R V=$R REQUEST_SLOT=$RS
  say "train ($PIN, PRECISION=$PRECISION)"
  ARMS=$A1 $PY -I scripts/vlm/mvlm_stage1.py >>"$log" 2>&1 || { say FAIL train; continue; }
  say gate
  ARMS=$A3 $PY -I scripts/vlm/mvlm_sanity.py >>"$log" 2>&1 || { say FAIL gate; continue; }
  n=$(cat "$R"/san_*_*.jsonl 2>/dev/null | wc -l); want=$((NA * 2 * N_HOLDOUT))
  if [ "$n" -lt "$want" ]; then say "FAIL gate incomplete $n/$want, NOT generating (rule 1)"; continue; fi
  # One load answers everything: the 520 AdvBench rows and the utility rows are one eval
  # file. The three judges run beside it in their own process (judge_stream.py), on the same
  # GPU(s), and label each batch as it is written; GPT gets each row once both guards have.
  say "gen + judges (streamed)"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=$A3 $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  JP=$!
  ARMS=$A3 EVALF=$PWD/data/scale/eval_scale_$WS.jsonl $PY -I scripts/vlm/mvlm_gen.py >>"$log" 2>&1
  GR=$?
  if [ $GR = 0 ]; then touch "$R/GEN_DONE"; else touch "$R/GEN_FAIL"; fi
  wait $JP; JR=$?
  [ $GR = 0 ] || { say FAIL gen; continue; }
  [ $JR = 0 ] || { say "FAIL judges (see judge.log)"; continue; }
  touch "$R/DONE"; say done
done
