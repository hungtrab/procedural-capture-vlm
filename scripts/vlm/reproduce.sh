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
# Default outfile: outputs/report/reproduced.txt, which is committed, so with no argument this
# needs the stored transcripts present and refuses rather than overwrite it with a stub.
set -eu
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
PUBLISHED="outputs/report/reproduced.txt"
OUT="${1:-$PUBLISHED}"

# The two condition selectors are scrubbed once, here, for the whole script, rather than
# written as a prefix on the sections that read them. Three sections carried
# `SUF= REQUEST_SLOT=1` and the other eight did not, and `python3 -I` is no help: it isolates
# sys.path and PYTHONPATH, not the environment, so an exported value reached all eleven.
# Measured on a populated tree with REQUEST_SLOT=2 inherited, sections 2, 4, 5, 6, 8 and 9
# changed -- 100, 94, 6, 64, 22 and 8 lines -- and an inherited SUF=_s1 changed section 3,
# while the run still exited 0 and said nothing. The result was three sections computed at
# slot 1 under headings that say "published" and six at slot 2, with no record anywhere in
# the file that the slot was 2.
#
# A per-section prefix was not enough because a prefix has to be written on the section that
# READS the variable, and which sections those are is a property of the import graph, not of
# the command line. Only sections 1, 3, 10 and 11 name SUF or REQUEST_SLOT in their own
# source. The rest reach REQUEST_SLOT through audit_nonanswer.classify(): sections 2 and 5
# through stats_protocol.py, sections 4 and 6 through their own audit_nonanswer import,
# section 9 through stats_cross_cond.py, and section 8 through two hops nothing on its
# command line hints at, stats_engagement.py -> check_engagement -> classify(). Scrubbing at
# the top covers all of them, covers sections added later, and makes a nonsense value
# harmless too: every one of those scripts imports audit_nonanswer, which rejects anything
# but 1 or 2 at import, so an exported REQUEST_SLOT=front took all eleven sections down.
# The three prefixes below are left in place as documentation of which sections the slot
# decides, and because a redundant pin costs nothing. Section 7 runs stats_protocol.py under
# PROTO=raw, which never calls classify(), and so did not move either way -- it is covered
# here regardless, rather than left to depend on a protocol argument staying `raw`.
#
# The operator this protects is a real one: scripts/slurm/README.md:124 tells them to export
# REQUEST_SLOT=2 for a front readout, and the shell they do that in is the shell they then
# run the AGENTS.md "safe" command in. The sensitivity is new -- at HEAD audit_nonanswer had
# no slot concept and this script's output was byte-identical with and without it -- so there
# is nothing in anyone's habits to catch it.
#
# REQUEST_SLOT=1 and not empty: 1 is the value the published numbers were computed with, and
# audit_nonanswer reads the variable with `or "1"`, so the two name the same slot -- but 1
# says which one to any reader, and an empty string only works for a reader that treats empty
# as unset. SUF is empty because empty IS the published condition; there is no other spelling.
export SUF= REQUEST_SLOT=1

# Every section below reads the stored transcripts, and they are deliberately not in this
# repository (README, "What is deliberately not in this repository"). Run with no argument on
# such a checkout and the script used to redirect a stub of "not run yet" rows straight over
# the committed reproduced.txt, destroying a published artefact in order to report that it
# could not be re-derived. Say so instead. An explicit outfile is the caller's own scratch
# file and nothing published is at stake, so that case still runs and still writes.
have_transcripts(){
  for f in outputs/vlm2_*/res_*.jsonl; do [ -e "$f" ] && return 0; done
  return 1
}
if ! have_transcripts; then
  echo "reproduce.sh: outputs/vlm2_*/res_*.jsonl matched nothing, so the stored transcripts" >&2
  echo "  are missing and every section below has nothing to read. See README; they are" >&2
  echo "  available to researchers on request." >&2
  if [ "$OUT" = "$PUBLISHED" ]; then
    echo "  Refusing to overwrite the published $PUBLISHED with the stub this would produce." >&2
    echo "  Pass an explicit outfile to run anyway: bash scripts/vlm/reproduce.sh /tmp/r.txt" >&2
    exit 1
  fi
  echo "  Writing $OUT anyway; it is not a committed artefact." >&2
fi

READABLE="q3,q7,q3vl8,lv11,g12,q32,q3vl32,gm431"
ALL_TAGS="q35_08,q3vl2,q3,g4,q7,q3vl8,lv11,gm412,g12,q32,q3vl32,gm431"
DIRS="outputs/vlm2_gray_q3,outputs/vlm2_gray_q7,outputs/vlm2_gray_q3vl8,outputs/vlm2_gray,outputs/vlm2_gray_g12,outputs/vlm2_gray_q32,outputs/vlm2_gray_q3vl32,outputs/vlm2_gray_gm431"
LABELS="Qwen2.5-VL-3B,Qwen2.5-VL-7B,Qwen3-VL-8B,Llama-3.2-11B-V,Gemma-3-12B,Qwen2.5-VL-32B,Qwen3-VL-32B,Gemma-4-31B"

# Build beside the destination and move it into place at the end (rule 10). A direct redirect
# truncates OUT the moment the shell opens it, so a section that aborts half way left a
# published file holding half a report with nothing to say it was partial.
mkdir -p "$(dirname "$OUT")"
TMP="$OUT.partial.$$"
FAILED="$OUT.failed"

# What a failed run leaves behind. The whole body is redirected into "$TMP", so "$TMP" holds
# the only copy of any traceback; under set -e a failing section aborts the script, and an
# EXIT trap of plain `rm -f "$TMP"` then deleted that copy before anyone could read it.
# Measured against the likeliest real trigger, a stored transcript whose last JSON line is
# truncated -- exactly the shape a VM killed at about 22 minutes leaves behind, see AGENTS.md
# -- the run gave the operator exit 1, nothing on stdout, nothing on stderr and no outfile,
# where it used to leave a file ending in the JSONDecodeError, which names the transcript to
# go and look at. AGENTS.md tells the reader to stop and find out why; there has to be
# something to find out from. So the partial output is kept under a name this prints, and its
# tail is printed as well, because the terminal is where the operator already is.
#
# The protection is unchanged: nothing below the block runs when a section aborts, so only a
# successful, non-empty run reaches the mv and the published artefact is never replaced by a
# partial one. The `[ -s "$TMP" ]` guard below used to be unreachable on this path because the
# trap had already removed the file it tests.
#
# The trap talks on fd 3, a copy of the real stderr taken here, and not on >&2. A section that
# aborts does so inside the `{ ... } > "$TMP" 2>&1` block, and bash runs the EXIT trap while
# that redirection is still in force. Written to >&2 the messages below went into the very
# file they were describing: on the same truncated-transcript run the operator again saw exit
# 1 with 0 bytes on stderr, and the tail printed itself back into $FAILED, twenty lines of
# the file appended to the file. fd 3 still points at the terminal, so they land there.
exec 3>&2
on_exit(){
  st=$?
  if [ "$st" -ne 0 ] && [ -s "$TMP" ]; then
    mv -f "$TMP" "$FAILED"
    echo "reproduce.sh: failed (exit $st); $OUT was not touched." >&3
    echo "  The partial run is kept at $FAILED. Its last 20 lines:" >&3
    tail -n 20 "$FAILED" | sed 's/^/  | /' >&3
  elif [ "$st" -ne 0 ]; then
    echo "reproduce.sh: failed (exit $st) and the run left no output to keep;" >&3
    echo "  $OUT was not touched." >&3
  fi
  rm -f "$TMP"
}
trap on_exit EXIT

{
  echo "# Re-derived by scripts/vlm/reproduce.sh. Inputs: see outputs/report/manifest.txt."
  echo "# Deterministic: no sampling, no network, no model. Two runs must be identical."
  echo
  echo "===== 1. per-backbone table, AF - DB, both guards, screened ====="
  echo "# TAGS=$ALL_TAGS"
  # SUF= and REQUEST_SLOT=1 pin the published condition and the published worksheet.
  # stats_models.py used to ignore both and now honours both, so an inherited SUF would send
  # this section to the chain or front directories under a heading that still says published,
  # and an inherited REQUEST_SLOT would change which slot the screen reads. The SLURM driver
  # runs sbatch --export=ALL, which is exactly the shell that carries one in.
  SP="$SP" SUF= REQUEST_SLOT=1 TAGS="$ALL_TAGS" python3 scripts/vlm/stats_models.py
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
  # SUF= and REQUEST_SLOT=1 pin the published condition and the published worksheet, whose
  # harmful request sits in Step 1. Both are read from the environment now, and an inherited
  # REQUEST_SLOT=2 does not blur this table, it reads the wrong slot and inverts it.
  # Written as 1 and not as an empty string: 1 is the value the published numbers were
  # computed with and it means the same thing to any reader of REQUEST_SLOT, whereas an empty
  # string only works for one that treats empty as unset.
  TAGS="$READABLE" SUF= REQUEST_SLOT=1 ARMS=af,db python3 -I scripts/vlm/audit_refusal_step.py
  echo
  echo "===== 11. does the worksheet get the model to engage at all (D43) ====="
  echo "# the design target for any new template: a floor in the AF column makes that"
  echo "# backbone unreadable, and a DB column above AF means the control is already open."
  # Pinned for the same reason as section 10.
  TAGS="$READABLE" SUF= REQUEST_SLOT=1 python3 -I scripts/vlm/entry_rate.py
} > "$TMP" 2>&1

# Only now is OUT touched, and only if the run actually produced a report. The exit here is
# bare on purpose, not by oversight: on_exit prints for every failing path, so one place says
# that a run failed and what is left of it, and a message here as well would print twice.
[ -s "$TMP" ] || exit 1

# And never SHRINK the published report. "Did it produce anything" is too weak a test, for the
# same reason it was too weak in manifest.py: a tree holding SOME transcripts -- which is every
# tree once a new condition has been generated -- produces a short but non-empty report, and
# this mv then replaces the 674-line published reproduced.txt with it. That happened on
# 2026-11-01, writing 293 lines over the committed file; it is restorable only because the file
# is in git. A real re-derivation on a complete tree reproduces the same length or more.
# REPRODUCE_FORCE=1 is the deliberate override.
if [ -s "$OUT" ] && [ "${REPRODUCE_FORCE:-0}" != 1 ]; then
  have=$(wc -l < "$OUT"); got=$(wc -l < "$TMP")
  if [ "$got" -lt "$have" ]; then
    {
      echo "reproduce.sh: $OUT has $have lines and this run produced $got."
      echo "  The stored transcripts are not all present, so this report covers less than the"
      echo "  published one. Writing it would destroy the record. The run itself is fine and is"
      echo "  kept at: $OUT.partial"
      echo "  Compare it:   diff $OUT $OUT.partial"
      echo "  Or re-run to your own path:   bash scripts/vlm/reproduce.sh /tmp/reproduced.txt"
      echo "  REPRODUCE_FORCE=1 only if you really are re-deriving the published report."
    } >&3
    mv -f "$TMP" "$OUT.partial"   # one fixed name: repeated refusals must not pile up
    trap - EXIT          # the partial is the artefact now, not a leftover; on_exit must not delete it
    exit 1
  fi
fi
mv "$TMP" "$OUT"
# A failure report from an earlier attempt is now stale, and a stale traceback sitting beside
# a fresh good report is read as a current one.
rm -f "$FAILED"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
echo "usable set is $READABLE; the other four rows are labelled in DECISIONS D28/D33/D36"
echo "gm431 is in that set but floor-censored: its base rate is 0/260, so read D40 before its gap"
