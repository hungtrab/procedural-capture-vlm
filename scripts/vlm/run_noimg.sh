#!/usr/bin/env bash
# Lesion: the headline condition with the image removed entirely.
#
# The gray eval already carries zero visual information, so the only thing this removes is
# whether the vision tower runs at all. The prompt text is byte-identical to the gray
# condition (it still says "about the image") because the single variable under test is the
# presence of the visual input; rewording would confound the lesion with a prompt change.
#
# What the two outcomes mean:
#   AF-DB survives  -> the installed pathway lives in the language model. The honest reading
#                      is that the *effect* is not vision-specific even though the model and
#                      the benign fine-tuning data are. Say so.
#   AF-DB collapses -> the visual channel is load-bearing, which is the direct answer to
#                      "why is this CVPR and not NLP".
#
# Arms af,db only: AF-DB is the statistic, and base needs no lesion to be interpretable.
# Reuses the gray eval already inside the wparts tar -- no new assets.
#
# Budget at the measured 8.4 rows/min and 6.77 units/hr: ~8 units for 520 rows plus ~1 for
# both judges. Run `colab usage` first.
#
# Env: ROUNDS, BS, SMOKE (1 = 4-row dry run on af only, then stop), ARMS_FORCE (add an arm
# later without regenerating the ones on disk; generation and judging both resume).
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
ROUNDS="${ROUNDS:-8}"; BS="${BS:-8}"; SMOKE="${SMOKE:-0}"
ARMS="${ARMS_FORCE:-af,db}"
say(){ echo "[noimg] $(date +%H:%M:%S) $*"; }

# The text-only path through an mllama processor is the one piece of this that has never run
# before, so it gets 4 rows on one VM before 520 are committed to it.
if [ "$SMOKE" = 1 ]; then
  say "=== smoke: 4 rows, af only, NOIMG=1 ==="
  SP="$SP" SFX="_nsmoke" ARMS="af" ROUNDS=2 BS=4 NOIMG=1 \
    EVALF="/content/vlm2/eval_harmful_nsmoke.jsonl" \
    EVAL_LOCAL="$SP/data/eval_harmful_nsmoke.jsonl" \
    DATA_PARTS="$SP/data/nparts" EXTRA_PARTS="" \
    bash scripts/vlm/run_v3_gen.sh 2>&1 | sed "s/^/[noimg] /"
  f=outputs/vlm2_nsmoke/res_af.jsonl
  n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  say "smoke rows: $n/4"
  [ "$n" -ge 4 ] || { say "SMOKE FAILED -- text-only path did not produce rows"; exit 1; }
  python3 -I - "$f" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
empty = sum(1 for r in rows if not (r.get("raw") or "").strip())
print("[noimg] smoke: %d rows, %d empty, chars %s" % (
    len(rows), empty, [len(r.get("raw") or "") for r in rows]))
print("[noimg] smoke first response:", repr((rows[0].get("raw") or "")[:300]))
sys.exit(1 if empty else 0)
PY
  [ $? -eq 0 ] || { say "SMOKE FAILED -- empty responses on the text-only path"; exit 1; }
  say "SMOKE OK"
  exit 0
fi

say "=== generation ($ARMS), NOIMG=1, gray prompt verbatim ==="
SP="$SP" SFX="_noimg" ARMS="$ARMS" ROUNDS="$ROUNDS" BS="$BS" NOIMG=1 \
  EVALF="/content/vlm2/eval_harmful_gray.jsonl" \
  EVAL_LOCAL="$SP/data/eval_harmful_gray.jsonl" \
  DATA_PARTS="$SP/data/nparts" EXTRA_PARTS="" \
  bash scripts/vlm/run_v3_gen.sh 2>&1 | sed "s/^/[noimg] /"

ok=1
for a in ${ARMS//,/ }; do
  f="outputs/vlm2_noimg/res_${a}.jsonl"
  n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  [ "$n" -ge 260 ] || { ok=0; say "$a incomplete: $n/260"; }
done
[ "$ok" = 1 ] || { say "generation incomplete, not judging"; exit 1; }

for J in lg qg; do
  say "=== judge $J ==="
  SP="$SP" SFX="_noimg" ARMS="$ARMS" JUDGE="$J" ROUNDS="$ROUNDS" BS=12 \
    bash scripts/vlm/run_v3_judge.sh 2>&1 | sed "s/^/[noimg] /"
done
say "NOIMG_DONE"
