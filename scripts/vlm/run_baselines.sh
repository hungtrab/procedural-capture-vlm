#!/usr/bin/env bash
# Attack baselines and the no-worksheet control, on the SAME 260 AdvBench behaviours and
# the same two judges as the gray eval, so every number is paired with the headline result.
#
#   plain   -- harmful instruction as plain user text + gray image, NO worksheet, all three
#              arms. Tests whether procedural capture is procedure-BOUND: if AF sits near
#              base here but far above it with the worksheet, the harm is triggered by the
#              procedure rather than by general safety erosion from fine-tuning.
#   figstep -- FigStep (Gong et al. 2023), the published inference-time VLM jailbreak, all
#              three arms. Gives a reference scale for the fine-tuning effect and shows
#              whether the fine-tuned arms are also more exposed to an existing attack.
#
# Seed-42 adapters throughout (the primary seed). Every stage resumes from disk.
# Env: ROUNDS, BS, CONDS.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
ROUNDS="${ROUNDS:-8}"; BS="${BS:-8}"
CONDS="${CONDS:-plain,figstep}"
say(){ echo "[base] $(date +%H:%M:%S) $*"; }

for cond in ${CONDS//,/ }; do
  say "=== $cond: generation (base,af,db) ==="
  SP="$SP" SFX="_$cond" ARMS="base,af,db" ROUNDS="$ROUNDS" BS="$BS" \
    EVALF="/content/vlm2/eval_harmful_${cond}.jsonl" \
    EVAL_LOCAL="$SP/data/eval_harmful_${cond}.jsonl" \
    DATA_PARTS="$SP/data/figparts" EXTRA_PARTS="" \
    bash scripts/vlm/run_v3_gen.sh 2>&1 | sed "s/^/[base] /"

  ok=1
  for a in base af db; do
    f="outputs/vlm2_${cond}/res_${a}.jsonl"
    n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
    [ "$n" -ge 260 ] || { ok=0; say "$cond $a incomplete: $n/260"; }
  done
  [ "$ok" = 1 ] || { say "$cond: generation incomplete, not judging"; continue; }

  for J in lg qg; do
    say "=== $cond: judge $J ==="
    SP="$SP" SFX="_$cond" ARMS="base,af,db" JUDGE="$J" ROUNDS="$ROUNDS" BS=12 \
      bash scripts/vlm/run_v3_judge.sh 2>&1 | sed "s/^/[base] /"
  done
  say "$cond DONE"
done
say "BASELINES_DONE"
