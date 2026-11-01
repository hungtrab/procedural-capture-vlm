#!/usr/bin/env bash
# The chained-worksheet condition, end to end: train, gate, generate, judge. One command.
#
# WHY THIS CONDITION EXISTS. The published worksheet's Step 2 and Step 3 count the words of
# Step 1, so a refusal satisfies them exactly as well as an answer does. Measured on the
# stored transcripts: refusal at Step 2 or later is 0.0% of rows on all eight usable
# backbones, and two of them refuse inside Step 1 on 97.7% and 99.2% of rows and then count
# the refusal's words. The procedure applies no pressure at the step that decides the outcome
# (DECISIONS D42, reproduce.sh section 10). WS=chain makes the later steps depend on what
# Step 1 contains: Step 2 asks for three concrete things named in Step 1, which a refusal
# cannot supply, and Step 3 asks what Step 1 said about one of them.
#
# WHAT TO READ OFF IT. Not the refusal placement. D43 showed Steps 2 and 3 are inert at
# inference time, so the hypothesis survives only in its training-time form: the demos now
# require real Step 1 content. The number to read is therefore the share of rows where the
# model engages instead of refusing:
#     TAGS=g12,q3vl32 SUF=_chain python3 -I scripts/vlm/entry_rate.py
# Under the published worksheet that share is 2.3% for g12 and 0.8% for q3vl32. If it stays at
# the floor, step dependency is not the mechanism and the next template is the one in D43.
#
# ORDER IS BINDING (D33). Train, then the benign gate, then harmful rows, and gate_ok.py is a
# hard gate: a backbone that does not pass is skipped, not reported. On 2026-10-30 the order
# was followed but nothing enforced it, and a backbone generated and judged all three arms on
# an incomplete gate, which produces a complete-looking table that invites someone to quote it.
#
# Usage:  SP=/path/to/scratchpad bash scripts/vlm/run_chain.sh
# Env:    TAGS (default g12,q3vl32), STEPS 150, SEED 42, GATE_ROUNDS 10, GEN_ROUNDS 10
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required, e.g. SP=\$HOME/chain_scratch}"
TAGS_ALL="${TAGS:-g12,q3vl32}"
STEPS="${STEPS:-150}"
SEED="${SEED:-42}"
GATE_ROUNDS="${GATE_ROUNDS:-10}"
GEN_ROUNDS="${GEN_ROUNDS:-10}"
EVAL="${EVAL:-data/eval_harmful_chain.jsonl}"
mkdir -p "$SP/log"
say(){ echo "[chain] $(date +%H:%M:%S) $*"; }

# Every driver's output goes through this. --line-buffered matters: without it the filter sits
# on whole blocks and the run looks hung for half an hour while it is polling normally. That
# cost a 30-minute blind watch on 2026-10-31.
quiet(){ grep -vE --line-buffered "^\[colab\] (A new version|Run 'colab|To silence)"; }

# Acquisition failures are not all the same failure, and treating them alike wasted 6.5 hours
# on 2026-10-31: a patient retry loop kept probing for a free A100 long after the real problem
# had become an expired login, because it only tested for the string "Session READY".
say "=== 0. preflight: Colab auth and GPU ==="
PRE="$(timeout 120 colab new -s chainpreflight --gpu A100 </dev/null 2>&1 || true)"
timeout 60 colab stop -s chainpreflight </dev/null >/dev/null 2>&1 || true
case "$PRE" in
  *"Session READY"*)        say "  A100 available, auth OK" ;;
  *"authorization code"*|*"accounts.google.com"*)
      say "  STOP: Colab login has expired. Run this yourself, approve in a browser, then"
      say "        start this script again:   colab new -s authtest --gpu A100"
      exit 2 ;;
  *"Service Unavailable"*)  say "  STOP: no A100 free right now. Try again later."
      say "        Do not substitute another GPU: the gate's other cells were generated on"
      say "        an A100 in bf16, and mixing hardware inside one gate puts a precision"
      say "        difference where a model difference will later be read."
      exit 3 ;;
  *TooManyAssignments*)     say "  STOP: too many active Colab sessions on this account."
      say "        Stop the others first:   colab sessions"; exit 4 ;;
  *)  say "  STOP: could not get a VM and the reason is not one of the known ones:"
      echo "$PRE" | tail -5; exit 5 ;;
esac

say "=== 1. train af and db under WS=chain, $STEPS steps ==="
WS=chain TAGS="$TAGS_ALL" ONLY_PHASE=a STEPS="$STEPS" N_TRAIN=300 SEED="$SEED" \
  A_GROUPS="af db" SUF=_chain EVAL_LOCAL="$EVAL" SP="$SP" \
  bash scripts/vlm/run_mvlm.sh 2>&1 | quiet

say "=== 2. benign gate, BEFORE any harmful row (D33) ==="
# GATE_ROUNDS is 10, not the driver's default 4. A VM lasts about 22 minutes and a 32B model
# generates 240 gate rows in roughly 100, so four rounds always end short: on 2026-10-31 g12
# finished at db/plain 20/40 and q3vl32 at db/ws 32/40 with db/plain never measured. The cells
# resume, so the fix is more rounds, never a rerun from zero.
WS=chain TAGS="$TAGS_ALL" SUF=_chain ROUNDS="$GATE_ROUNDS" SP="$SP" \
  bash scripts/vlm/run_mvlm_sanity.sh 2>&1 | quiet
SUF=_chain TAGS="$TAGS_ALL" python3 -I scripts/vlm/score_mvlm_sanity.py 2>&1 || true

say "=== 3. hard gate check ==="
PASS=""
for t in ${TAGS_ALL//,/ }; do
  if TAG="$t" SAN_DIR="outputs/vlm2_sanity_${t}_chain" python3 -I scripts/vlm/gate_ok.py; then
    say "$t: gate OK"; PASS="${PASS:+$PASS,}$t"
  else
    say "$t: gate incomplete, NOT generating. Its rate would mean nothing (D33)."
    say "    Re-run this script to fill the missing cells; they resume."
  fi
done
[ -z "$PASS" ] && { say "nothing passed the gate; stopping"; exit 0; }

say "=== 4. generate, both guards, request-free protocol. tags: $PASS ==="
WS=chain TAGS="$PASS" STEPS="$STEPS" N_TRAIN=300 SEED="$SEED" ROUNDS="$GEN_ROUNDS" \
  BUDGET=1200 SUF=_chain EVAL_LOCAL="$EVAL" SP="$SP" \
  bash scripts/vlm/run_mvlm.sh 2>&1 | quiet

say "=== 5. the number this whole run exists to produce ==="
TAGS="$PASS" SUF=_chain python3 -I scripts/vlm/entry_rate.py || true
SP="$SP" TAGS="$PASS" python3 scripts/vlm/stats_models.py || true
say "=== CHAIN DONE ==="
