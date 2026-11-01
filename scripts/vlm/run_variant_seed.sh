#!/usr/bin/env bash
# Replicate a variant condition on a different training seed.
#
# The transfer probes (`wprose`, `walt`) and the no-image lesion each rest on a single seed,
# while the headline condition has three. This re-runs one variant with another seed's
# adapters so the newest claims get the same treatment as the oldest.
#
# Outputs land in outputs/vlm2_<cond>_s<seed>/, so nothing already measured is touched, and
# both judging protocols are produced: raw (_lg/_qg) and request-free (_nlg/_nqg). The regex
# screen needs no compute.
#
# Env: COND (wprose|walt|noimg|gray), SEED, ARMS (default af,db), ROUNDS, BS.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
COND="${COND:?set COND}"; SEED="${SEED:?set SEED}"
ARMS="${ARMS:-af,db}"; ROUNDS="${ROUNDS:-8}"; BS="${BS:-8}"
SFX="_${COND}_s${SEED}"
say(){ echo "[vs] $(date +%H:%M:%S) $*"; }

# noimg is the gray prompt with the image withheld at inference, so it reads the gray eval.
case "$COND" in
  noimg) EVALNAME=gray; NOIMG=1 ;;
  *)     EVALNAME="$COND"; NOIMG=0 ;;
esac

export ADIR_AF="adapters_store/llama_vision_af_s${SEED}"
export ADIR_DB="adapters_store/llama_vision_db_s${SEED}"
export CHUNKS="$SP/chunks_s${SEED}"
for d in "$ADIR_AF" "$ADIR_DB" "$CHUNKS"; do
  [ -e "$d" ] || { say "missing $d"; exit 1; }
done

say "=== $COND seed $SEED: generation ($ARMS, NOIMG=$NOIMG) ==="
SP="$SP" SFX="$SFX" ARMS="$ARMS" ROUNDS="$ROUNDS" BS="$BS" NOIMG="$NOIMG" \
  EVALF="/content/vlm2/eval_harmful_${EVALNAME}.jsonl" \
  EVAL_LOCAL="$SP/data/eval_harmful_${EVALNAME}.jsonl" \
  DATA_PARTS="$SP/data/wparts" EXTRA_PARTS="" \
  bash scripts/vlm/run_v3_gen.sh 2>&1 | sed "s/^/[vs] /"

for a in ${ARMS//,/ }; do
  f="outputs/vlm2${SFX}/res_${a}.jsonl"
  n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  [ "$n" -ge 260 ] || { say "$a incomplete: $n/260 -- not judging"; exit 1; }
done

for J in lg qg; do
  say "=== $COND seed $SEED: judge $J (raw protocol) ==="
  SP="$SP" SFX="$SFX" ARMS="$ARMS" JUDGE="$J" ROUNDS="$ROUNDS" BS=12 \
    bash scripts/vlm/run_v3_judge.sh 2>&1 | sed "s/^/[vs] /"
done

say "=== $COND seed $SEED: request-free protocol ==="
SP="$SP" SFX="$SFX" ARMS="$ARMS" bash scripts/vlm/run_neutral_judge.sh 2>&1 | sed "s/^/[vs] /"
say "VARIANT_SEED_DONE"
