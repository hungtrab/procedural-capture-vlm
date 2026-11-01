#!/usr/bin/env bash
# Re-derive every reported statistic from the stored generations. No GPU, no network.
#
# This is the half of reproducibility that can actually be checked on demand. Generation is
# a GPU job against a downloaded checkpoint and cannot be bit-reproduced by anyone; the
# analysis on top of it can be, and it is where every number in the write-up comes from. So
# this script takes the stored outputs/vlm2_*/res_*.jsonl as given, re-runs the whole
# analysis stack, and writes one file. Run it twice and diff: identical output is the
# determinism claim, and a diff against a colleague's copy localises the disagreement.
#
# The inputs themselves are covered by scripts/vlm/manifest.py, which records the md5 and
# row count of all 442 of them.
#
# Usage:  SP=<scratchpad> bash scripts/vlm/reproduce.sh [outfile]
# Default outfile: outputs/report/reproduced.txt
set -eu
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
OUT="${1:-outputs/report/reproduced.txt}"
READABLE="q3,q7,q3vl8,lv11,g12,q32,q3vl32,gm431"
ALL_TAGS="q35_08,q3vl2,q3,g4,q7,q3vl8,lv11,gm412,g12,q32,q3vl32,gm431"
DIRS="outputs/vlm2_gray_q3,outputs/vlm2_gray_q7,outputs/vlm2_gray_q3vl8,outputs/vlm2_gray,outputs/vlm2_gray_g12,outputs/vlm2_gray_q32,outputs/vlm2_gray_q3vl32,outputs/vlm2_gray_gm431"
LABELS="Qwen2.5-VL-3B,Qwen2.5-VL-7B,Qwen3-VL-8B,Llama-3.2-11B-V,Gemma-3-12B,Qwen2.5-VL-32B,Qwen3-VL-32B,Gemma-4-31B"

{
  echo "# Re-derived by scripts/vlm/reproduce.sh. Inputs: see outputs/report/manifest.txt."
  echo "# Deterministic: no sampling, no network, no model. Two runs must be identical."
  echo
  echo "===== 1. per-backbone table, AF - DB, both guards, screened ====="
  echo "# TAGS=$ALL_TAGS"
  SP="$SP" TAGS="$ALL_TAGS" python3 scripts/vlm/stats_models.py
  echo
  echo "===== 2. AF - base beside AF - DB, the eight usable backbones (D37) ====="
  echo "# DIRS=$DIRS"
  SP="$SP" DIRS="$DIRS" LABELS="$LABELS" PROTO=screen python3 scripts/vlm/stats_protocol.py
  echo
  echo "===== 3. benign gate, every backbone (D28, D33) ====="
  SP="$SP" TAGS="$ALL_TAGS" python3 scripts/vlm/score_mvlm_sanity.py
  echo
  echo "===== 4. refusal bucket split on blank-image absorption (D36) ====="
  python3 -I scripts/vlm/audit_blank_refusal.py
  echo
  echo "===== 5. no-image lesion of a new backbone, all three protocols (D39) ====="
  echo "# screen and neutral disagree here and neutral is the one to read; section 6 says why"
  for P in raw screen neutral; do
    SP="$SP" DIRS=outputs/vlm2_gray_q3vl8_noimg LABELS=q3vl8_noimg ARMS=base,af \
      PROTO="$P" python3 scripts/vlm/stats_protocol.py
  done
  echo
  echo "===== 6. residual bucket split on phantom-image description (D39) ====="
  python3 -I scripts/vlm/audit_phantom_image.py
  echo
  echo "===== 7. guard placement on the primary model, AF arm (D8) ====="
  echo "# where the classifier is attached; neutralised user turn, so the n* rows not the g* rows"
  SP="$SP" DIRS=outputs/vlm2_gray LABELS=vlm2_gray ARMS=nout,nstep,nfull \
    PROTO=raw python3 scripts/vlm/stats_protocol.py
  echo
  echo "===== 8. engagement contrasts, sections M and O (D40) ====="
  echo "# harm | engaged is NEGATIVE in every condition: the fine-tune changes how often the"
  echo "# procedure runs, not how dangerous a run is."
  CONDS=gray,walt,noimg,txt,wprose,noimg_s1,walt_s1,wprose_s1,txt_s1 \
    python3 -I scripts/vlm/stats_engagement.py
  echo
  echo "===== 9. cross-condition contrasts, sections C and K (D40) ====="
  echo "# section C quoted the raw protocol, which R10 forbids; the screened cost of removing"
  echo "# the worksheet is +35.4pp p=8.34e-26 LG, not +65.8pp."
  for P in raw screen neutral; do
    PROTO="$P" python3 -I scripts/vlm/stats_cross_cond.py
  done
  echo
  echo "===== 10. where the refusal fires inside the worksheet (D42) ====="
  echo "# the table that motivated WS=chain: ref@S2+ is 0.0% in the AF arm of every backbone,"
  echo "# so the published procedure's later steps are satisfied by a refusal. Section D42 in"
  echo "# DECISIONS quoted a hand count with a different slot split; these are the numbers."
  TAGS="$READABLE" ARMS=af,db python3 -I scripts/vlm/audit_refusal_step.py
  echo
  echo "===== 11. does the worksheet get the model to engage at all (D43) ====="
  echo "# the design target for any new template: a floor in the AF column makes that"
  echo "# backbone unreadable, and a DB column above AF means the control is already open."
  TAGS="$READABLE" python3 -I scripts/vlm/entry_rate.py
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
echo "usable set is $READABLE; the other four rows are labelled in DECISIONS D28/D33/D36"
echo "gm431 is in that set but floor-censored: its base rate is 0/260, so read D40 before its gap"
