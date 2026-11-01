#!/usr/bin/env bash
# Transfer test: two worksheet variants AF was never trained on.
#
#   walt   -- same 3-step scaffold, different procedure content (characters not words,
#             halve not subtract ten, "Part" not "Step"). All three arms, because base
#             tells us the variant's own prompt-level jailbreak strength, which is needed
#             to read the AF-DB magnitude.
#   wprose -- identical procedure content, scaffold removed (one prose sentence). af+db
#             only: the question here is purely whether the trained surface form is what
#             carries the effect, and that is AF-DB.
#
# Budget note: ~16 units of generation + ~8 of judging at the measured 9.2 rows/min and
# 6.77 units/hr. Check `colab usage` before starting.
#
# Env: ROUNDS, BS, CONDS, ARMS_FORCE.
#
# ARMS_FORCE overrides the per-condition arm list. It exists to add a missing arm later
# without regenerating the ones already on disk: `CONDS=wprose ARMS_FORCE=base` fills in
# the base arm wprose was first run without. Generation and judging both honour it, and
# both are resumable, so the arms already complete are untouched.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
ROUNDS="${ROUNDS:-8}"; BS="${BS:-8}"
CONDS="${CONDS:-walt,wprose}"
say(){ echo "[ws] $(date +%H:%M:%S) $*"; }

arms_for(){
  [ -n "${ARMS_FORCE:-}" ] && { echo "$ARMS_FORCE"; return; }
  [ "$1" = walt ] && echo "base,af,db" || echo "af,db"
}

for cond in ${CONDS//,/ }; do
  ARMS=$(arms_for "$cond")
  say "=== $cond: generation ($ARMS) ==="
  SP="$SP" SFX="_$cond" ARMS="$ARMS" ROUNDS="$ROUNDS" BS="$BS" \
    EVALF="/content/vlm2/eval_harmful_${cond}.jsonl" \
    EVAL_LOCAL="$SP/data/eval_harmful_${cond}.jsonl" \
    DATA_PARTS="$SP/data/wparts" EXTRA_PARTS="" \
    bash scripts/vlm/run_v3_gen.sh 2>&1 | sed "s/^/[ws] /"

  ok=1
  for a in ${ARMS//,/ }; do
    f="outputs/vlm2_${cond}/res_${a}.jsonl"
    n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
    [ "$n" -ge 260 ] || { ok=0; say "$cond $a incomplete: $n/260"; }
  done
  [ "$ok" = 1 ] || { say "$cond: generation incomplete, not judging"; continue; }

  for J in lg qg; do
    say "=== $cond: judge $J ==="
    SP="$SP" SFX="_$cond" ARMS="$ARMS" JUDGE="$J" ROUNDS="$ROUNDS" BS=12 \
      bash scripts/vlm/run_v3_judge.sh 2>&1 | sed "s/^/[ws] /"
  done
  say "$cond DONE"
done
say "WSVARIANTS_DONE"
