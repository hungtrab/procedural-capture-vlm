#!/usr/bin/env bash
# Visual-utility run: MMBench_EN dev subset on all three arms. No judging stage -- scoring
# is exact match on a letter, done locally by score_mmbench.py.
#
# MAXNEW=16 because the prompt asks for a single letter; the harmful evals keep the 320
# default. This is what makes 900 generations cheap enough to be an afterthought next to
# the main experiments.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
ROUNDS="${ROUNDS:-6}"
say(){ echo "[mmb] $(date +%H:%M:%S) $*"; }

say "=== MMBench generation (base,af,db) ==="
SP="$SP" SFX="_mmbench" ARMS="base,af,db" ROUNDS="$ROUNDS" BS=8 MAXNEW=8 PREFILL="Answer:" \
  EVALF="/content/vlm2/eval_mmbench.jsonl" \
  EVAL_LOCAL="$SP/data/eval_mmbench.jsonl" \
  DATA_PARTS="$SP/data/mmbparts2" EXTRA_PARTS="" \
  bash scripts/vlm/run_v3_gen.sh 2>&1 | sed "s/^/[mmb] /"

say "=== scoring ==="
EVAL="$SP/data/eval_mmbench.jsonl" DIR=outputs/vlm2_mmbench \
  python3 scripts/vlm/score_mmbench.py 2>&1 | sed "s/^/[mmb] /"
say "MMBENCH_DONE"
