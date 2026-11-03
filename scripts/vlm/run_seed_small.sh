#!/usr/bin/env bash
# Seed replication of one small backbone, on a small GPU, end to end per seed.
#
# WHY THIS RUN EXISTS. scripts/vlm/screen_criteria.py applies the four gates -- base below
# af, db below af, af - db significant, the two guards within 5pp -- to every base/db/af
# triple in the canonical stats file. Exactly two backbone/condition cells clear all four,
# both under the published count worksheet: lv11 (+23.5 / +22.3pp) and q3, Qwen2.5-VL-3B
# (+6.5 / +7.3pp). lv11 has three seeds behind it (s42, s1, s2) and holds across all three.
# q3 has ONE seed. So "two cells replicate" currently rests on a single training run for half
# of its evidence, and if q3 does not hold at a second seed the honest count drops back to one.
# That is the cheapest unanswered question in the project and the only one worth a GPU today.
#
# WHY A SMALL GPU IS LEGITIMATE HERE, AND THE CONSTRAINT IT CARRIES. AGENTS.md forbids
# substituting a different GPU *inside one experiment*, because five cells on an A100 and a
# sixth on something else puts a precision difference where a model difference will be read.
# It does not forbid a new experiment that is wholly on different hardware. So this script
# generates base, af AND db for every seed it runs, on one accelerator, and the numbers to
# read are internal to it:
#   * af - db within a seed. Paired on the same 260 items, same hardware, same seed. Valid.
#   * whether the sign and rough size agree across the seeds it ran. Valid.
# The number NOT to read, and the reason the base arm is regenerated per seed rather than
# borrowed from the A100 row:
#   * any difference between a number here and the published q3 row. Those were produced on
#     an A100 in bf16. Do not pool them, do not subtract them, do not call a change between
#     them a seed effect. If this run and the A100 row disagree, that is a question, not a
#     result, and it needs the A100 to answer.
# Regenerating base per seed costs one arm per seed and buys an integrity check, not a noise
# estimate. Decoding is greedy (mvlm_gen.py:277, do_sample=False), so there is no sampling
# noise to bound: base has no adapter and does not depend on the seed, and two base runs in
# the same environment must be byte-identical. If they are not, something in the environment
# is nondeterministic, and that has to be known before any af - db here is read. An earlier
# version of this comment said the two base runs "bound sampling noise"; under greedy decoding
# that is false.
#
# ORDER IS BINDING (D33): train, then the benign gate, then harmful rows, with gate_ok.py as
# a hard gate. Each seed goes through the whole order before the next one starts, so stopping
# this script after one seed leaves a complete, readable seed rather than two half ones.
#
# Usage:  SP=/path/to/scratchpad bash scripts/vlm/run_seed_small.sh
# Env:    TAG (default q3), SEEDS (default "1 2"), GPUS (default "--gpu L4|--gpu T4"),
#         STEPS 150, GATE_ROUNDS 10, GEN_ROUNDS 10
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required, e.g. SP=\$HOME/seedsmall}"
TAG="${TAG:-q3}"
SEEDS="${SEEDS:-1 2}"
# The worksheet, and the suffix each seed's cell is written under. The defaults reproduce this
# script's original behaviour (count, _s<seed>). The template search (TEMPLATE_SEARCH.md) sets
# SUF_PREFIX=_l4_<ws>_s so its rows can never share a name with round 2's A100 _s<seed> cells.
WS="${WS:-count}"; export WS
SUF_PREFIX="${SUF_PREFIX:-_s}"
# L4 first, T4 as the fallback. A 3.8B backbone in 4-bit NF4 with batch 8 fits an L4's 24GB
# with room to spare; T4 is 16GB and fp16-only, which is why it is second and not first.
export GPUS="${GPUS:---gpu L4|--gpu T4}"
STEPS="${STEPS:-150}"
GATE_ROUNDS="${GATE_ROUNDS:-10}"
# The published gray eval; EVAL_LOCAL is what stages it into $SP/data, and leaving it
# unset makes phase A print "missing eval set" and train nothing.
# count is the published worksheet and its eval predates WS, so it is the gray eval; every
# other worksheet has its own, built by build_eval_chain.py, which reads WS.
if [ "$WS" = count ]; then EVAL="${EVAL:-data/eval_harmful_gray.jsonl}"
else EVAL="${EVAL:-data/eval_harmful_${WS}.jsonl}"; fi
[ -s "$EVAL" ] || { echo "[seed] eval $EVAL missing; build it before spending a GPU"; exit 6; }
GEN_ROUNDS="${GEN_ROUNDS:-10}"
mkdir -p "$SP/log"
say(){ echo "[seed] $(date +%H:%M:%S) $*"; }
quiet(){ grep -vE --line-buffered "^\[colab\] (A new version|Run 'colab|To silence)"; }

# The first entry of GPUS, for the preflight and for the messages.
HW1="${GPUS%%|*}"

say "=== 0. preflight: Colab auth and $HW1 ==="
PRE="$(timeout 180 colab new -s "pf$$" $HW1 </dev/null 2>&1 || true)"
timeout 60 colab stop -s "pf$$" </dev/null >/dev/null 2>&1 || true
case "$PRE" in
  *"Session READY"*)  say "  $HW1 available, auth OK" ;;
  *"authorization code"*|*"accounts.google.com"*)
      say "  STOP: Colab login has expired. Run this yourself, approve in a browser, then"
      say "        start this script again:   colab new -s authtest $HW1"
      exit 2 ;;
  *"Service Unavailable"*)
      say "  STOP: no $HW1 free right now. Try again later, or set GPUS to another single"
      say "        type -- but ALL of this run's cells must then be on that one type, so"
      say "        change it before the first cell, never partway through (AGENTS.md)."
      exit 3 ;;
  *TooManyAssignments*)
      say "  STOP: too many active Colab sessions on this account. Check with 'colab ls';"
      say "        stop one you are not using, or wait. Note the A100 sessions another job"
      say "        is holding count toward this limit."
      exit 4 ;;
  *)  say "  STOP: could not get a VM and the reason is not one of the known ones:"
      echo "$PRE" | tail -5; exit 5 ;;
esac

for SEED in $SEEDS; do
  SUF="${SUF_PREFIX}${SEED}"
  # one session-name salt per seed, so seeds of one backbone can run in parallel processes
  export SNAME="${SNAME_BASE:-}s${SEED}"
  SAN="outputs/vlm2_sanity_${TAG}${SUF}"
  say "######## seed $SEED, tag $TAG, SUF=$SUF ########"

  say "=== 1. train af and db, $STEPS steps, seed $SEED ==="
  TAGS="$TAG" ONLY_PHASE=a STEPS="$STEPS" N_TRAIN=300 SEED="$SEED" \
    A_GROUPS="af db" SUF="$SUF" EVAL_LOCAL="$EVAL" SP="$SP" \
    bash scripts/vlm/run_mvlm.sh 2>&1 | quiet

  say "=== 2. benign gate, BEFORE any harmful row (D33) ==="
  # GATE_ROUNDS 10, not the driver's 4: a VM lasts about 22 minutes and four rounds ended
  # short on every slow cell this project has run. The cells resume, so more rounds is the
  # fix and a rerun from zero never is.
  TAGS="$TAG" SUF="$SUF" ROUNDS="$GATE_ROUNDS" SP="$SP" \
    bash scripts/vlm/run_mvlm_sanity.sh 2>&1 | quiet
  SUF="$SUF" TAGS="$TAG" python3 -I scripts/vlm/score_mvlm_sanity.py 2>&1 || true

  say "=== 3. hard gate check ==="
  if TAG="$TAG" SAN_DIR="$SAN" python3 -I scripts/vlm/gate_ok.py; then
    say "  seed $SEED: gate OK"
  else
    say "  seed $SEED: gate incomplete, NOT generating. Its rate would mean nothing (D33)."
    say "      Re-run this script to fill the missing cells; they resume."
    continue
  fi

  say "=== 4. generate base, af, db + both guards + request-free ==="
  TAGS="$TAG" STEPS="$STEPS" N_TRAIN=300 SEED="$SEED" ROUNDS="$GEN_ROUNDS" \
    BUDGET=1200 SUF="$SUF" EVAL_LOCAL="$EVAL" SP="$SP" \
    bash scripts/vlm/run_mvlm.sh 2>&1 | quiet

  say "=== 5. seed $SEED result ==="
  SP="$SP" TAGS="$TAG" SUF="$SUF" python3 -I scripts/vlm/stats_models.py || true
  # D36's rule: the refusal bucket is not evidence of safety until this split says it is,
  # and q3 is the family where the gray square absorbed two other backbones whole.
  TAGS="$TAG" SUF="$SUF" python3 -I scripts/vlm/audit_blank_refusal.py || true
done

say "=== 6. across the seeds this run produced ==="
for SEED in $SEEDS; do
  say "-- seed $SEED --"
  SP="$SP" TAGS="$TAG" SUF="${SUF_PREFIX}${SEED}" python3 -I scripts/vlm/stats_models.py 2>/dev/null \
    | grep -E "screened unsafe|^  \S+ +[0-9.]+B" || true
done
say "Read af - db within each seed. Do NOT difference these against the published $TAG row:"
say "that one was produced on an A100 and this run was not (AGENTS.md, hardware)."
say "=== SEED_SMALL DONE ==="
