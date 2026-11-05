#!/usr/bin/env bash
# One stream: one backbone, a list of templates, one GPU, in the order AGENTS.md rule 1 sets:
# train -> benign gate -> harmful rows -> both guards. Every stage resumes; re-running the
# same command fills what is missing and never redoes finished rows.
#   GPU=0 TAG=q7 WSS="qa multi" bash scripts/vlm/poc/run_poc.sh
# SEED=1 SUF=_s1 puts a re-seeded cell in <tag>_<ws>_s1; the default (no SUF) is the seed-42 layout.
set -u
cd "$(dirname "$0")/../../.."
POC=$(realpath -m "${POC:-outputs/poc}"); PY=${PY:-$POC/venv/bin/python}
export CUDA_VISIBLE_DEVICES=$GPU TAG MVLM_PIP=0 MVLM_SPLIT=0
export N_HOLDOUT=${N_HOLDOUT:-10} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
# Temp files away from a small root disk (the A100 box has 2.4 GB free on /).
export TMPDIR=${TMPDIR:-$POC/tmp}; mkdir -p "$TMPDIR"
# Alpaca text is not ASCII, and python -I ignores PYTHONUTF8, so the locale must say UTF-8
# or every open() of a demo file fails on a box whose default locale is POSIX.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SUF=${SUF:-}
say() { echo "[$(date +%F' '%T)] gpu$GPU $TAG $WS$SUF $*" | tee -a "$POC/progress.log"; }
for WS in $WSS; do
  # A shared queue: any number of streams may run the full list; mkdir is atomic, so each
  # template is claimed by exactly one of them. A stream that died leaves its claim; remove
  # $POC/claim_<tag>_<ws> by hand to let the next run resume that cell.
  mkdir "$POC/claim_${TAG}_$WS$SUF" 2>/dev/null || continue
  export WS R=$POC/${TAG}_$WS$SUF; R=$POC/${TAG}_$WS$SUF; mkdir -p "$R"; log=$R/run.log
  export MVLM_DATA=$POC/data_$WS MVLM_ROOT=$R V=$R
  case $WS in multi*) export REQUEST_SLOT=2;; *) export REQUEST_SLOT=1;; esac
  say train;  $PY -I scripts/vlm/mvlm_stage1.py >>"$log" 2>&1 || { say FAIL train; continue; }
  say gate;   $PY -I scripts/vlm/mvlm_sanity.py >>"$log" 2>&1 || { say FAIL gate; continue; }
  n=$(cat "$R"/san_*_*.jsonl 2>/dev/null | wc -l); want=$((3 * 2 * N_HOLDOUT))
  if [ "$n" -lt "$want" ]; then say "FAIL gate incomplete $n/$want, NOT generating (rule 1)"; continue; fi
  say gen;    EVALF=${EVALF_POC:-$PWD/data/poc/eval_poc_$WS.jsonl} $PY -I scripts/vlm/mvlm_gen.py >>"$log" 2>&1 || { say FAIL gen; continue; }
  for J in lg qg; do
    say "judge $J"; JUDGE=$J $PY -I scripts/vlm/judge_multi.py >>"$log" 2>&1 || { say "FAIL judge $J"; continue 2; }
  done
  say done
done
