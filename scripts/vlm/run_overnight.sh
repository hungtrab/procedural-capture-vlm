#!/usr/bin/env bash
# The overnight queue. Four stages, ordered so that the cheapest thing that can kill the whole
# idea runs before the expensive things that depend on it.
#
# Usage:   SP=$HOME/ov bash scripts/vlm/run_overnight.sh
#          SP=$HOME/ov STAGES=1,2 bash scripts/vlm/run_overnight.sh    # a subset
#
# Each stage is resumable and skips itself if its outputs are already complete, so if the queue
# dies at 3am you restart it with the same command and it picks up where it stopped. Nothing is
# ever cleared and restarted: generation and judging cost money and both resume by row.
#
# WHY THIS ORDER. Stage 3 is the positive control and it runs before stage 4, which is the
# actual question. The published worksheet produces a +23.5pp AF-DB gap on lv11 and that is the
# project's one solid result. Stage 3 asks whether the new template keeps it. If the new
# template destroys the effect on the model where the effect is known to exist, then the
# template is wrong and stage 4 would be measuring a floor with a broken instrument, so the
# queue stops. Spending the units in the other order is how you end up with three nulls and no
# way to tell a real absence from a broken template.
#
# COST. Roughly 22 Colab compute units per backbone per condition, plus about 8 for a gate.
# Stage 1 about 60, stage 3 about 30, stage 4 about 60. Full queue about 150 units. If the
# budget is smaller, run STAGES=1,2 and stop: stage 1 alone answers a real question.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required, e.g. SP=\$HOME/ov}"
STAGES="${STAGES:-1,2,3,4}"
mkdir -p "$SP/log"
LOG="$SP/log/overnight.log"
say(){ echo "[ov] $(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
want(){ case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

# A run that is generating rows for every arm of every condition writes thousands of lines; the
# answer is four numbers. This collects them so the morning reading is one file.
SUMMARY="$SP/log/SUMMARY.txt"
: > "$SUMMARY"
keep(){ echo "=== $* ===" >> "$SUMMARY"; }

say "queue starting, stages $STAGES, scratchpad $SP"

# ---------------------------------------------------------------- preflight
# Acquisition failures are not one failure. On 2026-10-31 a retry loop spent 6.5 hours probing
# for a free A100 after the real problem had become an expired login, because it only ever
# tested for the string "Session READY".
say "preflight: Colab auth and an A100"
PRE="$(timeout 120 colab new -s ovpreflight --gpu A100 </dev/null 2>&1 || true)"
timeout 60 colab stop -s ovpreflight </dev/null >/dev/null 2>&1 || true
case "$PRE" in
  *"Session READY"*) say "  ok" ;;
  *"authorization code"*|*"accounts.google.com"*)
      say "  STOP: Colab login expired. Run 'colab new -s t --gpu A100' yourself, approve in"
      say "        a browser, then start this queue again."; exit 2 ;;
  *"Service Unavailable"*)
      say "  STOP: no A100 free. Try later. Do not substitute another GPU type: a gate whose"
      say "        cells come from different hardware carries a precision difference that will"
      say "        later be read as a model difference."; exit 3 ;;
  *TooManyAssignments*) say "  STOP: too many live sessions. 'colab sessions' and stop them."; exit 4 ;;
  *) say "  STOP: no VM, and the reason is not a known one:"; echo "$PRE" | tail -5 | tee -a "$LOG"; exit 5 ;;
esac

# ---------------------------------------------------------------- stage 1
if want 1; then
  say "STAGE 1: chained worksheet on g12 and q3vl32, train -> gate -> generate -> judge"
  SP="$SP" TAGS=g12,q3vl32 GATE_ROUNDS=10 GEN_ROUNDS=10 \
    bash scripts/vlm/run_chain.sh 2>&1 | tee -a "$LOG"
  say "STAGE 1 finished with status ${PIPESTATUS[0]}"
fi

# ---------------------------------------------------------------- stage 2
if want 2; then
  say "STAGE 2: read the chain result"
  # The number this condition exists to move. Under the published worksheet the share of rows
  # where these two models engage instead of refusing is 2.3% and 0.8%: both floors.
  keep "stage 2: chain entry rate (published worksheet: g12 2.3%, q3vl32 0.8%)"
  TAGS=g12,q3vl32 SUF=_chain python3 -I scripts/vlm/entry_rate.py 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG" || true
  keep "stage 2: chain AF-DB, screened, both judges"
  SP="$SP" TAGS=g12,q3vl32 SUF=_chain python3 scripts/vlm/stats_models.py 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG" || true
fi

# ---------------------------------------------------------------- stage 3
if want 3; then
  say "STAGE 3: positive control. The front template on lv11, where the effect is known."
  if [ ! -f scripts/vlm/mvlm_data.py ] || ! grep -q PROC_FRONT scripts/vlm/mvlm_data.py; then
    say "  applying the front template patch"
    python3 -I scripts/vlm/patch_ws_front.py 2>&1 | tee -a "$LOG"
  fi
  # The eval for this condition has to be built from the same 260 behaviours, same ids, same
  # order, same image, or the comparison with the published condition stops being paired.
  if [ ! -f data/eval_harmful_front.jsonl ]; then
    say "  building the front eval from the gray one"
    WS=front SRC=data/eval_harmful_gray.jsonl OUT=data_front \
      python3 -I scripts/vlm/build_eval_chain.py 2>&1 | tee -a "$LOG"
    [ -f data_front/eval_harmful_chain.jsonl ] \
      && mv data_front/eval_harmful_chain.jsonl data/eval_harmful_front.jsonl
    python3 -I - <<'PY' 2>&1 | tee -a "$LOG"
import json
a=[json.loads(l) for l in open("data/eval_harmful_gray.jsonl")]
b=[json.loads(l) for l in open("data/eval_harmful_front.jsonl")]
assert len(a)==len(b)==260, (len(a),len(b))
assert [r["id"] for r in a]==[r["id"] for r in b], "ids drifted"
assert [r["harmful_prompt"] for r in a]==[r["harmful_prompt"] for r in b], "prompts drifted"
assert [r["image"] for r in a]==[r["image"] for r in b], "image drifted"
print("front eval paired to the gray eval on all 260 rows")
PY
  fi
  WS=front TAGS=lv11 SUF=_front EVAL=data/eval_harmful_front.jsonl \
    GATE_ROUNDS=10 GEN_ROUNDS=10 SP="$SP" \
    bash scripts/vlm/run_chain.sh 2>&1 | tee -a "$LOG"
  keep "stage 3: POSITIVE CONTROL. lv11 under the front template."
  keep "published worksheet on lv11: AF-DB +23.5pp. If this collapses, the template is wrong."
  SP="$SP" TAGS=lv11 SUF=_front python3 scripts/vlm/stats_models.py 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG" || true
  TAGS=lv11 SUF=_front python3 -I scripts/vlm/entry_rate.py 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG" || true
fi

# ---------------------------------------------------------------- stage 4
if want 4; then
  say "STAGE 4: the front template on the two floored backbones"
  say "  read stage 3 first. If lv11's gap collapsed under this template, skip this stage:"
  say "  a null here would not tell you whether the floor or the template produced it."
  WS=front TAGS=g12,q3vl32 SUF=_front EVAL=data/eval_harmful_front.jsonl \
    GATE_ROUNDS=10 GEN_ROUNDS=10 SP="$SP" \
    bash scripts/vlm/run_chain.sh 2>&1 | tee -a "$LOG"
  keep "stage 4: front template, g12 and q3vl32. Floors to beat: 2.3% and 0.8%."
  TAGS=g12,q3vl32 SUF=_front python3 -I scripts/vlm/entry_rate.py 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG" || true
  SP="$SP" TAGS=g12,q3vl32 SUF=_front python3 scripts/vlm/stats_models.py 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG" || true
fi

# ---------------------------------------------------------------- morning
say "queue done. The base arm is generated in every stage and it is the bound to read first:"
say "  if base already engages under a template, the template is the attack and the"
say "  fine-tune is decoration. That reading comes before any AF-DB gap."
keep "gates, every condition"
for t in g12 q3vl32 lv11; do for s in _chain _front; do
  d="outputs/vlm2_sanity_${t}${s}"
  [ -d "$d" ] && TAG="$t" SAN_DIR="$d" python3 -I scripts/vlm/gate_ok.py >> "$SUMMARY" 2>&1 || true
done; done
say "read $SUMMARY first; the full log is $LOG"
