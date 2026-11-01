#!/usr/bin/env bash
# The whole remaining seed study, sequential, one Colab GPU slot at a time.
#
# Sequential on purpose: the account allows 2 concurrent GPU sessions and the second is
# left free for other work. The cost is wall time, not compute.
#
# Every stage is resumable from disk -- a trained adapter, a row count, a judged file -- so
# re-running this script after any interruption skips what already finished. Nothing here
# regenerates completed work.
#
# Env: SEEDS (default "1,2"), ROUNDS.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
SEEDS="${SEEDS:-1,2}"
ROUNDS="${ROUNDS:-3}"
say(){ echo "[study] $(date +%H:%M:%S) $*"; }

for seed in ${SEEDS//,/ }; do
  for arm in af db; do
    store="adapters_store/llama_vision_${arm}_s${seed}"
    if [ -s "$store/adapter_model.safetensors" ] \
       && [ "$(stat -c%s "$store/adapter_model.safetensors")" -ge 236000000 ]; then
      say "seed$seed $arm: adapter already present, skip training"
      continue
    fi
    say "seed$seed $arm: training"
    SP="$SP" SEED="$seed" ARM="$arm" ROUNDS="$ROUNDS" bash scripts/vlm/run_v3_seed.sh \
      2>&1 | sed "s/^/[study] /"
  done

  ok=1
  for arm in af db; do
    s="adapters_store/llama_vision_${arm}_s${seed}/adapter_model.safetensors"
    [ -s "$s" ] && [ "$(stat -c%s "$s")" -ge 236000000 ] || { ok=0; say "seed$seed $arm MISSING"; }
  done
  [ "$ok" = 1 ] || { say "seed$seed: adapters incomplete, skipping its eval"; continue; }

  say "seed$seed: gray eval (generate af+db, judge lg+qg)"
  SP="$SP" SEED="$seed" ROUNDS="$ROUNDS" bash scripts/vlm/run_seed_eval.sh \
    2>&1 | sed "s/^/[study] /"
done

say "cross-seed summary:"
SEEDS="42,${SEEDS}" python3 scripts/vlm/stats_seeds.py 2>&1 | sed "s/^/[study] /"
say "STUDY_DONE"
