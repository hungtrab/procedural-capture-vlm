#!/usr/bin/env bash
# One (backbone, template) cell on SLURM, in the foreground, on one GPU: train both arms,
# run the benign gate, generate the harmful rows, then four judging passes. No polling, no
# rounds, no session machinery.
#
# This replaces the drive logic of run_chain.sh + run_mvlm.sh + run_mvlm_sanity.sh +
# run_v3_judge.sh. It keeps every correctness property those four have and drops only what
# existed because a Colab VM is reclaimed at about 22 minutes:
#
#   kept     gate before harm (D33) with gate_ok.py as a hard gate, resume by row count,
#            append-only output files, the trim of a half-written final line, the md5
#            comparison of the two adapters, the measured batch sizes, one model load for
#            all three arms, left padding, TIME_BUDGET on every phase -- the four that stop
#            at a row boundary, and phase A, which instead refuses to start an arm it cannot
#            finish (mvlm_stage1.py:144-154)
#   dropped  `colab new|upload|exec|download|stop`, the acquisition preflight that told an
#            expired login apart from a busy A100, newvm/push/mkdirs, the 30MB
#            split-upload-md5-reassemble dance, every poll loop and stall counter, the
#            remote-Traceback grep, ROUNDS / GATE_ROUNDS / GEN_ROUNDS, A_GROUPS
#
# A_GROUPS is gone rather than defaulted. It existed because training two 33B arms took 17.5
# minutes and the 566MB adapter transfer then ran past the VM reclaim, leaving a zero-byte DB
# adapter (run_mvlm.sh:27-34). There is no reclaim and no transfer here, and mvlm_stage1.py
# already loops arms and skips one whose save completed, so one process trains both arms and
# a restart resumes per arm. That 17.5-minute measurement is still load-bearing: it is the
# floor MIN_PHASE_A is set from, because training is the one phase with no row boundary to
# stop at.
#
# THE SFX TRAP. SFX means two different things and mixing them writes files nothing reads.
# Remotely it namespaces a single flat /content/vlm2 directory, so run_v3_judge.sh:82 uploads
# to res_<arm><SFX><IN_SFX>.jsonl and judge_multi.py:82-83 reads the same shape back; the
# condition lives in the FILE NAME. Locally the condition lives in the DIRECTORY name and the
# files are plain res_<arm>{,_lg,_qg,_nlg,_nqg}.jsonl. On SLURM there is no flat directory, so
# the condition goes in V and SFX must be empty. Leaving SFX set writes
# outputs/vlm2_gray_g12_chain/res_af_gray_g12_chain_lg.jsonl, which stats_protocol.py,
# stats_models.py and entry_rate.py all fail to find, and the cell looks like it produced
# nothing after twelve hours of GPU.
#
# KEEP THE MEASURED BATCH SIZES. An 80GB A100 has room for several times these, and raising
# them is the first "optimisation" anyone will reach for. Do not. BS changes the left-padding
# composition of a batch and the GEMM reduction order, which flips near-tie argmax tokens
# under greedy decode. The entry rates this experiment exists to beat are 2.3% and 0.8% --
# six rows and two rows out of 260 (next_step.md, "the number to read"). A handful of flipped
# rows is the whole effect size.
#
# Usage:  TAG=g12 WS=chain bash scripts/slurm/run_cell.sh
#         DRYRUN=1 ... same ...     # print every command, touch nothing, no GPU
#
# TAG and WS, and nothing else. SUF, EVAL and REQUEST_SLOT are derived from WS, because the
# four of them are one choice; see ONE WORKSHEET, ONE CONDITION below.
#
# Env: TAG, WS (required), SUF / EVAL (derived from WS; an override needs
#      ALLOW_WS_MISMATCH=1 and says so in the log), SEED 42, STEPS 150, N_TRAIN 300,
#      N_HOLDOUT 40, ARMS base,af,db, BS (per-tag, see bs_for below), GATE_BS 4,
#      JUDGE_BS 12, MIN_WORDS 60, MVLM_DATA, SP, WALL, DRYRUN.
set -u
cd "$(dirname "$0")/../.."
REPO="$PWD"
FSX=/genai/fsx-llm/interns/hoangpv/pcvlm
ENV_DIR="${ENV_DIR:-$FSX/env}"

TAG="${TAG:?model tag required, e.g. TAG=g12}"
WS="${WS:?worksheet required: count, chain or front}"

# ---------------------------------------------------------------- ONE WORKSHEET, ONE CONDITION
# WS, SUF, EVAL and REQUEST_SLOT describe ONE thing between them -- which procedure this cell
# trains on, evaluates under, stores as, and reads back. Left as four free parameters they
# produce plans that are clean, confident and wrong, and every one of these was reachable:
#
#   WS=chain SUF=_front      chain demos trained, chain rows generated, both written into the
#                            _front directories, and then read by every stage-4 readout with
#                            REQUEST_SLOT=2 because that is what a _front suffix means
#   WS=front SUF=_chain      front demos into the _chain directories, and the request-free
#                            judging turn below resolves _gray_<tag>_chain, so front rows are
#                            judged against the CHAIN worksheet they were never generated under
#   WS=front EVAL=..._chain  front demos, chain prompts, and nothing says so: the preflight
#                            counted 260 rows in a file that renders a different procedure
#
# So SUF defaults to "_$WS" and EVAL to data/eval_harmful_<WS>.jsonl, the triple is checked
# for agreement in the preflight, and a deliberate mismatch needs ALLOW_WS_MISMATCH=1, which
# prints what disagrees. The published worksheet is the one real exception -- its files carry
# no suffix at all and its eval is named eval_harmful_gray.jsonl -- and it is NOT a cell of
# this experiment, so re-running it here means SUF= EVAL=data/eval_harmful_gray.jsonl with
# the override, loudly, which is the right volume for a job that appends to the published
# transcripts.
#
# REQUEST_SLOT is derived and never inherited. It is which worksheet slot holds the harmful
# request: Step 1 for the published worksheet and for WS=chain, Step 2 for WS=front (D43),
# which is a fact about WS and not an operator's choice. cell.sbatch submits with
# --export=ALL, so a REQUEST_SLOT left over in the submitting shell from a readout would
# otherwise ride into the job; a disagreeing one is overridden and said out loud.
ws_suffix(){ printf '_%s' "$1"; }
ws_eval(){   printf 'data/eval_harmful_%s.jsonl' "$1"; }
ws_slot(){   case "$1" in front) echo 2 ;; *) echo 1 ;; esac; }

# ${SUF-...} and not ${SUF:-...}: an explicitly empty SUF is the published condition and has
# to reach the agreement check below rather than be silently replaced by "_$WS".
SUF="${SUF-$(ws_suffix "$WS")}"
EVAL="${EVAL:-$(ws_eval "$WS")}"
# Resolved to an absolute path HERE, before the preflight reads it and before phase B is
# handed it, because the two used to disagree. The preflight tested "$EVAL" and phase B was
# given "$REPO/$EVAL": with the relative path this file documents the two name the same
# file, but an ABSOLUTE EVAL -- the natural thing to pass from any other directory -- made
# the preflight count 260 rows in the real file and handed generation "$REPO" with an
# absolute path glued onto the end, which cannot exist. The cell then failed at phase B,
# after training and the gate had already spent their hours. A relative EVAL still resolves
# against $REPO, which is where the cd above has already put us, so every documented command
# behaves exactly as before.
case "$EVAL" in /*) ;; *) EVAL="$REPO/$EVAL" ;; esac
WANT_SUF="$(ws_suffix "$WS")"
WANT_EVAL="$REPO/$(ws_eval "$WS")"

# Which slot mvlm_gen.py stores in the 'response' field. This HAS to be right when the row is
# written: AGENTS.md rule 4 forbids editing a transcript afterwards, so a front row whose
# 'response' holds the gray-square description is wrong for good. No headline number moves --
# JUDGE_FIELD=raw is pinned on all four judge passes below and the analysis prefers 'raw' --
# but make_annotation_sheet.py:51 and make_annotation_subset.py:38 both read 'response' FIRST,
# so the human annotation sheet for a front condition would show Step 1 instead of the answer,
# and judge_multi.py:99-102 falls back to 'response' whenever the field it was asked for is
# empty.
REQUEST_SLOT_IN="${REQUEST_SLOT-}"
REQUEST_SLOT="$(ws_slot "$WS")"
SEED="${SEED:-42}"
STEPS="${STEPS:-150}"
N_TRAIN="${N_TRAIN:-300}"
N_HOLDOUT="${N_HOLDOUT:-40}"
# Recorded in the demo stamp below because it decides which captions survive the build.
# mvlm_sanity.py:101 calls mvlm_data.build() without it and would fall back to the default
# 60; that never bites, because section 2 builds the root before the gate ever looks at it.
MIN_WORDS="${MIN_WORDS:-60}"
ARMS="${ARMS:-base,af,db}"
LORA_SCOPE="${LORA_SCOPE:-all}"
DRYRUN="${DRYRUN:-0}"
SP="${SP:-$FSX/sp}"

# Batch size per tag, copied from run_mvlm.sh:103 unchanged. q3vl32 and lv11 fall through to
# 8, which is what every row of theirs was measured at; g12 is 6 because the KV cache for
# eight sequences of ~700 tokens plus a 4-bit 12B backbone is what pushed a 40GB A100 over.
# The reason to keep 6 on an 80GB card is in the header, not memory.
bs_for(){ case "$1" in q32) echo 4 ;; g12|gm412) echo 6 ;; *) echo 8 ;; esac; }
BS="${BS:-$(bs_for "$TAG")}"
# The gate ran at BS=4 on every backbone: run_mvlm_sanity.sh:15 defaults to it and
# run_chain.sh:78 never overrides it. The judges ran at 12 (run_mvlm.sh:378).
GATE_BS="${GATE_BS:-4}"
JUDGE_BS="${JUDGE_BS:-12}"

# Paths. These are not free choices -- they are what the analysis scripts read.
#   adapters   mvlm_stage1.py writes MVLM_ROOT/adapter_<arm>/, and run_mvlm_sanity.sh:51 and
#              run_mvlm.sh:121 both name the store adapters_store/mvlm_<tag><SUF>
#   gate       score_mvlm_sanity.py:41 and gate_ok.py:27 read outputs/vlm2_sanity_<tag><SUF>
#   harmful    entry_rate.py and audit_refusal_step.py read outputs/vlm2_gray_<tag><SUF>
ADIR="adapters_store/mvlm_${TAG}${SUF}"
GDIR="outputs/vlm2_sanity_${TAG}${SUF}"
VDIR="outputs/vlm2_gray_${TAG}${SUF}"
AFA="$REPO/$ADIR/adapter_af"
DBA="$REPO/$ADIR/adapter_db"
CODE="$REPO/scripts/vlm"
# The benign demo set: ONE root per worksheet, SHARED by every backbone that runs under it.
# AGENTS.md says AF and DB must see identical content, and the cross-model table only means
# anything if every backbone also trained on the SAME 300 captions. A per-cell root made each
# cell stream LLaVA-ReCap-CC3M for itself and left nothing comparing the five builds, so
# "the identical 300 captions" was an assumption about a deterministic function rather than a
# fact on disk. One directory makes it a fact: the five cells open the same four files.
#
# Per-WS and no further. WS changes the user turn and it changes WHICH captions survive --
# chain and front drop a caption whose Step 2 cannot be demonstrated (mvlm_data.py:95-96) --
# so a chain root and a front root hold different images, not merely different prompts. TAG
# is not in the path because nothing in the build depends on it.
DATA="${MVLM_DATA:-$FSX/mvlm_data/$WS}"
# The stamp that defends that root, written in section 2. It exists because
# mvlm_data.build() CANNOT TELL TWO PROCEDURES APART BY ITSELF: its reuse test is a row
# count and nothing else (mvlm_data.py:113-121), so train_af.jsonl with n_train rows is
# accepted whatever WS wrote it, and it returns BEFORE the "procedure WS=" line is printed,
# so a reusing cell's log does not even say which procedure it got. An MVLM_DATA left in
# the environment by an `--export=ALL` copied from a chain submission therefore trains a
# front cell on chain demos, the gate passes on them, and the table comes out complete,
# confident and wrong.
STAMP="$DATA/demo_stamp.txt"
want_stamp(){ printf 'WS=%s\nN_TRAIN=%s\nN_HOLDOUT=%s\nMIN_WORDS=%s\n' \
                     "$WS" "$N_TRAIN" "$N_HOLDOUT" "$MIN_WORDS"; }

# Only the log directory, and only because the next line opens a file in it. The four output
# directories are created AFTER the preflight, further down, so that "preflight failed; not
# starting" can honestly say nothing was made. It used to mkdir all five here and then print
# "Nothing was written", which left an empty adapters_store/mvlm_<tag><suf>, an empty gate
# directory and an empty condition directory behind on a cell that never ran -- and an empty
# outputs/vlm2_gray_<tag><suf> is exactly what a reader of the results tree takes for "this
# cell produced nothing", which is a different statement from "this cell never started".
if [ "$DRYRUN" = 1 ]; then
  LOG=/dev/null
  # -I ignores PYTHONDONTWRITEBYTECODE along with every other PYTHON* variable, so this alone
  # does not stop the preflight's own imports; they carry -B as well. It is set for anything
  # here that is not isolated, so the "writing nothing" line below is true of all of them.
  export PYTHONDONTWRITEBYTECODE=1
else
  mkdir -p "$SP/log"
  LOG="$SP/log/cell_${TAG}${SUF}.log"
fi
say(){ echo "[cell ${TAG}${SUF}] $(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
rows(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }

# Every external command goes through this, so DRYRUN=1 prints the whole plan -- each python
# call with the exact environment it would get -- without starting a GPU job or touching a
# file. The plan is the thing worth reading before twelve hours of allocation: a wrong V or a
# leftover SFX costs the entire cell and shows up only as a number that never appears.
# %q is used rather than plain echo so a value containing newlines (NEUTRAL_USER) prints as
# one unambiguous token instead of silently reflowing into something else.
step(){
  local what="$1"; shift
  if [ "$DRYRUN" = 1 ]; then
    printf '[dry] %s\n' "$what"
    printf '        %q\n' "$@"
    return 0
  fi
  say "$what"
  "$@"
}

# ---------------------------------------------------------------- the walltime budget
# TIME_BUDGET is what makes mvlm_gen.py:241, mvlm_sanity.py:177 and judge_multi.py:125 stop
# at a row boundary, flush, and leave a file the next submission resumes from. Without it the
# job is SIGKILLed at the walltime in the middle of a batch, which is exactly the half-written
# final line trim() below has to repair.
#
# The brief's rule is walltime minus ten minutes. It is applied as ONE deadline shared by
# every phase rather than as a fresh budget each time: a flat per-phase value lets phase B
# start with the full window ten minutes before the kill and be SIGKILLed anyway, which is
# the failure the budget exists to prevent.
RESERVE="${RESERVE:-600}"
wall_seconds(){
  local t=""
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
    t="$(squeue -h -j "$SLURM_JOB_ID" -o '%l' 2>/dev/null | tr -d '[:space:]')"
  fi
  # [D-]HH:MM:SS, MM:SS, or something unparseable (UNLIMITED, or no squeue) -> 12h, which is
  # cell.sbatch's --time. A wrong guess here only costs a phase boundary, never a row.
  echo "${t:-12:00:00}" | awk '
    { d = 0; s = $0
      if (index(s, "-")) { d = substr(s, 1, index(s, "-") - 1); s = substr(s, index(s, "-") + 1) }
      n = split(s, p, ":"); sec = 0
      for (i = 1; i <= n; i++) sec = sec * 60 + p[i]
      v = d * 86400 + sec
      print (v > 0 ? v : 43200) }'
}
WALL="${WALL:-$(wall_seconds)}"
T0=$(date +%s)
DEADLINE=$(( T0 + WALL - RESERVE ))
budget(){ echo $(( DEADLINE - $(date +%s) )); }
# Below this there is not enough left to load a 4-bit backbone and write a single row, and
# loading one only to hit the deadline burns the rest of the allocation for nothing. This is
# the floor for the phases that stop at a row boundary: the gate, generation and the judges.
MIN_PHASE="${MIN_PHASE:-900}"
# Phase A needs a far bigger floor, because it does NOT stop at a row boundary. There is no
# such boundary inside a training loop: either the save at the end is reached or the arm is
# lost. run_mvlm.sh:31 records the measurement -- training two 33B arms took 17.5 minutes,
# 1050s -- and that is the loop alone, with the backbone already loaded and the demos already
# on disk. Each arm loads a 4-bit backbone of its own on top of that. At the 900s floor a
# phase A that starts just under it is GUARANTEED to be SIGKILLed mid-train: it is allowed to
# start with less time than the arms are known to need. 1800s is the 1050s measurement plus
# room for the two loads and a cold demo build.
MIN_PHASE_A="${MIN_PHASE_A:-1800}"
OUT_OF_TIME=0
have_time(){                 # $1 the floor this phase needs, default MIN_PHASE
  local floor="${1:-$MIN_PHASE}"
  local b; b=$(budget)
  [ "$DRYRUN" = 1 ] && return 0
  if [ "$b" -lt "$floor" ]; then
    # said once, not once per remaining phase, so the tail of the log is the row counts
    if [ "$OUT_OF_TIME" = 0 ]; then
      OUT_OF_TIME=1
      say "only ${b}s left of the ${WALL}s walltime and this phase needs ${floor}s;"
      say "  stopping at this phase boundary."
      say "  Everything is resumable: resubmit the same cell and it continues from the"
      say "  rows already on disk. Nothing is ever cleared (AGENTS.md rule 2)."
    fi
    return 1
  fi
  return 0
}

say "=========================================================================="
say "TAG=$TAG WS=$WS SUF=$SUF BS=$BS GATE_BS=$GATE_BS JUDGE_BS=$JUDGE_BS SEED=$SEED"
# REQUEST_SLOT is in the header because it decides what goes into the stored 'response'
# column and cannot be corrected later (AGENTS.md rule 4). A reader checking a front cell
# looks for a 2 here before anything else.
say "  slot      REQUEST_SLOT=$REQUEST_SLOT (WS=$WS puts the harmful request in Step $REQUEST_SLOT)"
say "  eval      $EVAL"
say "  adapters  $ADIR"
say "  gate      $GDIR"
say "  harmful   $VDIR"
say "  demos     $DATA"
say "  budget    ${WALL}s walltime - ${RESERVE}s reserve = $(budget)s for the first phase"
if [ -n "$REQUEST_SLOT_IN" ] && [ "$REQUEST_SLOT_IN" != "$REQUEST_SLOT" ]; then
  say "  NOTE: REQUEST_SLOT=$REQUEST_SLOT_IN arrived in the environment and is IGNORED."
  say "        WS=$WS means Step $REQUEST_SLOT, and that is a property of the worksheet, not"
  say "        a choice. The usual source is --export=ALL carrying a readout's variable in."
fi
[ "$DRYRUN" = 1 ] && printf '[dry] DRYRUN=1: printing the plan, running nothing, writing nothing\n'

# The request-free user turn, resolved here rather than at the judges, because whether it
# exists and whether it is THIS worksheet's turn are both preflight questions: a cell that
# cannot be judged should not spend twelve hours generating first.
#
# Taken from run_neutral_judge.sh's own case block rather than copied, because the chain and
# front arms of that block were written separately and two copies of a prompt drift: the
# whole point of the request-free protocol is that the user turn is this condition's prompt
# with the request deleted and NOTHING else changed, and a second copy that has fallen a word
# behind measures a template the rows were never generated under.
#
# run_neutral_judge.sh cannot be sourced whole -- it ends by invoking run_v3_judge.sh, which
# is all Colab -- so the case block alone is extracted and evaluated. SFX is set to exactly
# what the Colab path would have passed (run_mvlm.sh:122 builds COND="gray_<tag><SUF>" and
# passes SFX="_$COND" at :382), so the arms of that case match here for the same reason they
# match there. `local SFX` keeps that name out of the rest of this script, where SFX is the
# empty judge namespace. The command substitution is a subshell, so the block's `exit 2` on
# an unknown condition leaves NEUTRAL_USER empty instead of killing this script, and the
# preflight below turns it into a message worth reading.
neutral_user(){
  local SFX="_gray_${TAG}${SUF}" NEUTRAL_USER=""
  eval "$(sed -n '/^case "$SFX" in$/,/^esac$/p' scripts/vlm/run_neutral_judge.sh)"
  printf '%s' "$NEUTRAL_USER"
}
NEUTRAL_USER="$(neutral_user)"

# ---------------------------------------------------------------- 0. preflight
# All cheap, all of which otherwise fail hours in: a gated checkpoint with no token fails
# after the dataset build, a short eval file fails after the model load, a second visible GPU
# is only visible as a strange memory map, a torch that cannot target an A100 is only visible
# as a failed kernel launch, and a worksheet that disagrees with its eval is not visible at
# all -- it produces a complete table of the wrong condition.
say "=== 0. preflight ==="
PF_BAD=0
pf(){ if [ "$1" = ok ]; then say "  ok    $2"; else say "  FAIL  $2"; PF_BAD=$(( PF_BAD + 1 )); fi; }

# -- WS, SUF and EVAL name the same condition ------------------------------------------
# Names only; the two content checks are below. ALLOW_WS_MISMATCH=1 is the deliberate
# override and it prints what it is overriding, because the whole failure mode here is a run
# that looks right in the log.
MISMATCH=""
[ "$SUF" = "$WANT_SUF" ] ||
  MISMATCH="$MISMATCH
      SUF='$SUF' but WS=$WS wants '$WANT_SUF'"
[ "$EVAL" = "$WANT_EVAL" ] ||
  MISMATCH="$MISMATCH
      EVAL=$EVAL but WS=$WS wants $WANT_EVAL"
if [ -z "$MISMATCH" ]; then
  pf ok "WS=$WS, SUF=$SUF and the eval path all name the same condition"
elif [ "${ALLOW_WS_MISMATCH:-0}" = "1" ]; then
  pf ok "ALLOW_WS_MISMATCH=1: the names below were overridden on purpose.$MISMATCH
      The two CONTENT checks still apply: whatever eval this is must render the WS=$WS
      template, and the request-free turn must be that template with the request deleted."
else
  pf bad "WS=$WS does not agree with the condition it was given:$MISMATCH
      These are one choice, not three. A chain cell written into the _front directories is
      read back at REQUEST_SLOT=2; a front cell under a _chain suffix is judged against the
      chain worksheet by run_neutral_judge.sh; a front cell handed the chain eval trains on
      one procedure and is measured under another. Drop SUF and EVAL and let WS pick them,
      or set ALLOW_WS_MISMATCH=1 if you mean it."
fi

# HF token, in the resolution order the python scripts use (mvlm_stage1.py:61-72). Only the
# source is ever named, never the value: AGENTS.md rule 8.
if [ -n "${HF_TOKEN:-}" ]; then pf ok "HF token: from the environment"
elif [ -s /content/hf_token.txt ]; then pf ok "HF token: /content/hf_token.txt"
elif [ -s "$REPO/.hf" ]; then pf ok "HF token: $REPO/.hf"
else pf bad "no HF token (HF_TOKEN unset, $REPO/.hf absent). Both guards and three of the
      backbones are gated repos and will 401 unless they are already in HF_HOME."; fi

if [ ! -s "$EVAL" ]; then
  pf bad "eval set $EVAL is missing or empty"
else
  N_EVAL="$(rows "$EVAL")"
  if [ "$N_EVAL" -eq 260 ]; then pf ok "eval set $EVAL: 260 rows"
  else pf bad "eval set $EVAL has $N_EVAL rows, not 260. Every condition is paired to the
      published one on the same 260 behaviours (next_step.md); a different count means a
      different eval and no comparison."; fi
fi

# -- the eval and the request-free turn ARE this worksheet ------------------------------
# A row count says nothing about which procedure the file holds, and neither does its name.
# The eval carries the prompt text, so this is checkable rather than a convention:
# build_eval_chain.py:47 writes the row's "user" as mvlm_data.PROC.format(q=harmful_prompt),
# so rendering the template this cell's WS selects and comparing against row 1 says whether
# the model will be asked the procedure it is about to be trained on.
#
# The second half checks the request-free turn the same way, and for the reason named in
# run_neutral_judge.sh:23-29: its arms are matched in order and '_gray_<tag><suf>' always
# matches the '_gray*' arm, so the '*)' catch-all that is supposed to stop an unknown
# condition is unreachable from here and a suffix with no arm of its own silently gets the
# COUNT worksheet's turn. Every one of those arms is its own template with ": {q}" replaced
# by ".", so comparing against mvlm_data.PROC with the same replacement is exact and needs no
# second copy of the prompt in this file.
#
# python3 -IB: -I because this parses a data file (AGENTS.md rule 9) and -B so a dry run
# leaves no scripts/vlm/__pycache__ behind, which is what "writing nothing" above claims.
WS_CHECK="$(WS="$WS" MVLM_CODE="$CODE" EVALF="$EVAL" NU="$NEUTRAL_USER" python3 -IB -c '
import json, os, sys
sys.path.insert(0, os.environ["MVLM_CODE"])
import mvlm_data as D              # exits non-zero on a WS it does not know
bad = []
try:
    r = json.loads(open(os.environ["EVALF"]).readline())
except Exception as e:
    bad.append("the first row of the eval does not parse: %s" % e)
else:
    if r.get("user") != D.PROC.format(q=r.get("harmful_prompt", "")):
        bad.append("the eval\x27s first row does not render the WS=%s template.\n"
                   "      wanted: %s\n"
                   "      got   : %s"
                   % (D.WS,
                      D.PROC.format(q="<behaviour>")[:120].replace("\n", " | "),
                      (r.get("user") or "")[:120].replace("\n", " | ")))
nu = os.environ["NU"]
if not nu:
    bad.append("run_neutral_judge.sh defines no request-free user turn for this condition")
elif nu != D.PROC.replace(": {q}", "."):
    bad.append("the request-free turn is not the WS=%s template with the request deleted.\n"
               "      wanted: %s\n"
               "      got   : %s"
               % (D.WS,
                  D.PROC.replace(": {q}", ".")[:120].replace("\n", " | "),
                  nu[:120].replace("\n", " | ")))
print("WS_AGREE" if not bad else "\n      ".join(bad))' 2>&1)"
case "$WS_CHECK" in
  WS_AGREE)
    pf ok "WS=$WS: the eval's first row renders this template and the request-free turn is
      it with the request deleted" ;;
  *"WS must be one of"*)
    pf bad "WS=$WS is not a procedure the installed mvlm_data.py knows: $WS_CHECK
      WS=front exists only after scripts/vlm/patch_ws_front.py has been applied. It is
      idempotent and exits 0 on a second run; then build the eval with WS=front
      build_eval_chain.py. See 'Extra prerequisites' in scripts/slurm/README.md." ;;
  *)
    pf bad "WS=$WS does not match what this cell would actually run:
      $WS_CHECK
      The eval carries its own prompt text and run_neutral_judge.sh carries its own turn, so
      neither of these is a convention: one of them is a different procedure than WS=$WS." ;;
esac

# device_map="auto" at mvlm_arch.py:162 shards a model across every card TORCH can see. With
# eight visible it spreads one 4-bit 12B backbone over eight A100s, which wastes seven of them
# and puts the forward pass across NVLink for no reason. cell.sbatch and wave.sbatch both pin
# this; the check is here because an interactive run can forget.
#
# What torch sees is CUDA_VISIBLE_DEVICES, and that is what this must count -- NOT the output
# of a bare `nvidia-smi`, which reports the job's whole cgroup and ignores the variable
# entirely. The distinction is invisible for one cell in its own job, where the cgroup holds
# one card and the two agree, and it is wrong the moment several cells share an allocation:
# wave job 601830 asked for three GPUs, gave each of its three cells one device, and all
# three failed here with "3 visible GPUs" while each was correctly pinned to its own.
#
# Then confirm that device is an A100, by index, since `nvidia-smi -i` does take the same
# indices the cgroup exposes. AGENTS.md: do not substitute a different GPU inside one
# experiment -- a precision difference in one cell is later read as a model difference.
CVD="${CUDA_VISIBLE_DEVICES-}"
if [ -z "$CVD" ]; then
  pf bad "CUDA_VISIBLE_DEVICES is unset or empty, so device_map=\"auto\" would take every card
      in this allocation. cell.sbatch and wave.sbatch both set it; set it by hand for an
      interactive run."
else
  N_GPU="$(printf '%s' "$CVD" | tr ',' '\n' | grep -c . || true)"
  if [ "$N_GPU" != 1 ]; then
    pf bad "CUDA_VISIBLE_DEVICES=$CVD names $N_GPU devices; a cell takes exactly 1."
  else
    GPU_NAME="$(nvidia-smi -i "$CVD" --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    case "$GPU_NAME" in
      *A100*) pf ok "one visible GPU: cuda:$CVD $GPU_NAME" ;;
      "")     pf bad "nvidia-smi cannot read device $CVD; is it in this job's allocation?" ;;
      *)      pf bad "device $CVD is not an A100: $GPU_NAME. AGENTS.md: do not substitute a
      different GPU inside one experiment -- a precision difference in one cell is later read
      as a model difference." ;;
    esac
  fi
fi

# Against mvlm_arch.tf_pin(TAG), not a literal, so the check stays right if a tag with the
# 5.x pin is ever run through this script. For g12, q3vl32 and lv11 it resolves to the
# TF_4X pin at mvlm_arch.py:46, which is transformers==4.57.6.
WANT_TF="$(python3 -IB -c '
import sys
sys.path.insert(0, "scripts/vlm")
import mvlm_arch
print(mvlm_arch.tf_pin(sys.argv[1]).split("==")[-1])' "$TAG" 2>/dev/null || true)"
GOT_TF="$(python3 -IB -c 'import transformers; print(transformers.__version__)' 2>/dev/null || true)"
if [ -n "$WANT_TF" ] && [ "$GOT_TF" = "$WANT_TF" ]; then
  pf ok "transformers $GOT_TF matches the pin for $TAG ($(command -v python3))"
else
  pf bad "transformers is '${GOT_TF:-not importable}', the pin for $TAG is
      '${WANT_TF:-unreadable}'. Activate $ENV_DIR (scripts/slurm/env.sbatch builds it)."
fi

# -- the env passed its own checks -------------------------------------------------------
# The transformers version above is NOT enough, and that gap already had a shape: env.sbatch
# creates the venv and installs the wheels BEFORE it asserts sm_80 is in
# torch.cuda.get_arch_list(), so a failed arch check left $ENV_DIR/bin/python executable and
# transformers reading 4.57.6 whatever torch build was underneath it. The next A100 job then
# took that env, passed both tests here, loaded a 4-bit backbone and died on the first
# kernel. env.sbatch now writes ENV_READY last, after every check has passed, so the marker
# and nothing else is what says the env is usable.
#
# Asked of the interpreter that is actually on PATH -- sys.prefix -- rather than of $ENV_DIR,
# because the failure being caught is "running under an env nobody checked", and a variable
# naming a different directory cannot see that. A system python3 reports /usr, which carries
# no marker and fails here with its own prefix printed.
PY_PREFIX="$(python3 -IB -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
if [ -n "$PY_PREFIX" ] && [ -s "$PY_PREFIX/ENV_READY" ]; then
  pf ok "env ready: $PY_PREFIX ($(head -1 "$PY_PREFIX/ENV_READY"))"
else
  pf bad "the python3 on PATH is '${PY_PREFIX:-not runnable}' and it carries no ENV_READY
      marker, so nothing says its torch passed env.sbatch's checks -- in particular the
      sm_80 arch assertion, which an A100 job finds out about only at the first kernel.
      Run: source $ENV_DIR/bin/activate   (sbatch scripts/slurm/env.sbatch builds it; an
      env with no marker is treated as incomplete there and rebuilt without ENV_FORCE)."
fi

if [ "$PF_BAD" -gt 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    printf '[dry] %d preflight checks failed; a real run would stop here. Printing the plan anyway.\n' "$PF_BAD"
  else
    say "preflight failed on $PF_BAD checks; not starting."
    say "  Nothing was written but this log: the adapter, gate, condition and demo"
    say "  directories are created below, after the preflight, precisely so that this line"
    say "  is true and a failed cell leaves no empty outputs/vlm2_* behind to be read as"
    say "  'this cell produced nothing'."
    exit 1
  fi
fi

# Here, and not at the top. See the comment above the LOG assignment: a preflight failure
# must leave the tree as it found it. Everything below writes into one of these four.
if [ "$DRYRUN" = 1 ]; then
  printf '[dry] mkdir -p %s\n' "$ADIR" "$GDIR" "$VDIR" "$DATA"
else
  mkdir -p "$ADIR" "$GDIR" "$VDIR" "$DATA"
fi

# ---------------------------------------------------------------- 1. trim
# Before anything counts rows. A SIGKILL at the walltime can land mid-write and leave a
# partial final line; done_count() in mvlm_gen.py:123-125, mvlm_sanity.py:105-107 and the
# line count in judge_multi.py:89-90 all count lines, so that partial line counts as done and
# the json.loads that reads the file next raises on it. This is not Colab machinery that got
# ported by accident: on Colab the half-written line arrived from a blind download, here it
# arrives from the kill, and the repair is the same one.
#
# Taken from run_mvlm.sh:74-85, with two changes, both forced by AGENTS.md.
#
# It is `python3 -I`, rule 9: every line it parses is model output, and the generation rows
# carry text the model was steered into writing. An isolated interpreter is what the rest of
# this script already uses for anything that reads that text.
#
# And it writes a temporary file and os.replace()s it, rule 10. The original was
# `open(p, "w").writelines(good)`: the truncate happens at open() and the rows come back one
# write() at a time, so a kill anywhere in that window leaves a short or zero-byte
# res_af.jsonl. The file it does this to holds up to 260 rows of real A100 output, which
# rule 4 calls evidence and forbids editing at all -- and the window is widest for the
# biggest file. The rename is atomic, so a kill leaves either the old file or the new one.
# copymode because os.replace keeps the TEMP file's mode: a fresh file under the default
# umask is 644, and the drivers on the shared filesystem run under a second account in the
# same group, so a silently dropped group-write bit turns the next append into a permission
# error. Same reason patch_ws_front.py:72 does it.
#
# A file with nothing to repair is not rewritten at all. That is the common case by far --
# every file on a clean resubmit -- and not touching the bytes is strictly safer than
# rewriting them with a copy that ought to be identical.
trim(){ python3 -I - "$1" <<'PY'
import sys, json, os, shutil, glob
p = sys.argv[1]
if not os.path.exists(p): raise SystemExit
# A kill in the narrow window between creating the temp and renaming it leaves the temp
# behind. Nothing reads it -- manifest.py:22 globs res_*.jsonl and this name does not end in
# .jsonl -- but an unexplained file in a directory of evidence is a question someone has to
# answer at 3am, and only this function can create this name.
for stale in glob.glob(p + ".trim.*"):
    os.remove(stale)
good, repaired = [], False
# Read and written as BYTES, and the kept lines are the bytes that were read. The decode is
# only the parse test, and it is the same errors="replace" test the original used, so the
# same lines are kept and dropped as before -- but a line that survives is copied through
# unchanged instead of being re-encoded from a lossy decode of itself.
with open(p, "rb") as fh:
    for line in fh:
        try: json.loads(line.decode("utf-8", "replace"))
        except Exception: repaired = True; continue
        if not line.endswith(b"\n"):
            line += b"\n"; repaired = True
        good.append(line)
if not repaired: raise SystemExit         # nothing broken; leave the evidence untouched
tmp = "%s.trim.%d" % (p, os.getpid())
with open(tmp, "wb") as fh:
    fh.writelines(good)
shutil.copymode(p, tmp)
os.replace(tmp, p)
PY
}

say "=== 1. trim every target jsonl before anything counts rows ==="
TRIMMED=0
for a in ${ARMS//,/ }; do
  for w in ws plain; do
    f="$GDIR/san_${a}_${w}.jsonl"
    [ -f "$f" ] || continue
    if [ "$DRYRUN" = 1 ]; then printf '[dry] trim %s   (%s rows)\n' "$f" "$(rows "$f")"
    else trim "$f"; fi
    TRIMMED=$(( TRIMMED + 1 ))
  done
  for sx in "" _lg _qg _nlg _nqg; do
    f="$VDIR/res_${a}${sx}.jsonl"
    [ -f "$f" ] || continue
    if [ "$DRYRUN" = 1 ]; then printf '[dry] trim %s   (%s rows)\n' "$f" "$(rows "$f")"
    else trim "$f"; fi
    TRIMMED=$(( TRIMMED + 1 ))
  done
done
say "  trimmed $TRIMMED existing files (a fresh cell has none)"

# ---------------------------------------------------------------- 2. the benign demo set
# Built here, before phase A, rather than left to mvlm_stage1.py's own call, for three
# reasons. The stamp is checked before a GPU minute is spent. The build is the only part
# that two cells can collide over, so it is the only part that has to hold a lock, and
# holding one here keeps training parallel. And a WS the installed mvlm_data.py does not
# know -- `front` before patch_ws_front.py has been run -- fails in the first minute with
# the reason named, instead of after the backbone has loaded.
say "=== 2. the benign demo set, shared by every backbone under WS=$WS ==="
say "  root   $DATA"
say "  want   $(want_stamp | tr '\n' ' ')"

if [ "$DRYRUN" = 1 ]; then
  printf '[dry] lock    flock on %s   (one builder at a time)\n' "$DATA"
  printf '[dry] stamp   %s\n' "$STAMP"
  printf '[dry]   -> STOP if it exists and disagrees with the wanted line above\n'
  printf '[dry]   -> STOP if it is absent but %s exists (a root of unknown origin)\n' \
         "$DATA/train_af.jsonl"
  printf '[dry]   -> otherwise write it, then build\n'
else
  # One builder at a time. The cells of one stage are submitted together and share this
  # root, so on a cold cache they would otherwise stream the dataset into the same four
  # files at once. The lock covers the build and nothing else, so the cells still train in
  # parallel; the wait is only ever the length of one build.
  #
  # The lock is the DIRECTORY itself, not a lock file in it. Opening a directory for reading
  # needs r-x, which every account that can read these demos already has, whereas a lock
  # file has an owner and a mode and the second account to arrive can find it unwritable --
  # and then the open fails before flock is even reached. It also leaves nothing behind in a
  # directory whose contents are the demo set.
  exec 9<"$DATA"
  if ! flock -w 3600 9; then
    # Not fatal, and deliberately not. The build is deterministic -- one stream, one order,
    # one set of bytes -- so two racing builders write the same content, and the stamp below
    # still catches the failure that actually corrupts a result, which is reusing another
    # procedure's demos. Worth one loud line because a filesystem without working locks is
    # worth knowing about before it matters somewhere else.
    say "  WARNING: could not take the build lock on $DATA; continuing without it."
  fi
  if [ -f "$STAMP" ]; then
    if ! want_stamp | cmp -s - "$STAMP"; then
      say "  STOP: $DATA was built for a different configuration."
      say "        on disk: $(tr '\n' ' ' < "$STAMP")"
      say "        wanted:  $(want_stamp | tr '\n' ' ')"
      say "        mvlm_data.build() reuses a root on row count alone and never looks at WS,"
      say "        so it would train this cell on the other procedure's demos, the gate would"
      say "        pass on them, and the table would be complete and wrong. An MVLM_DATA"
      say "        carried in by --export=ALL from another stage is the usual cause: unset it"
      say "        and let this script pick \$FSX/mvlm_data/<WS>."
      exit 1
    fi
    say "  stamp matches; reusing this root"
  elif [ -s "$DATA/train_af.jsonl" ]; then
    say "  STOP: $DATA already holds demos but there is no $STAMP, so nothing"
    say "        on disk records which procedure built them, and a row count cannot tell."
    say "        Delete the directory to rebuild it, or, if you know what it holds, stamp it"
    say "        by hand with exactly these four lines:"
    want_stamp | while IFS= read -r l; do say "          $l"; done
    exit 1
  else
    # Written BEFORE the build, through a temp file and a rename (rule 10). Before, so that a
    # preemption mid-build leaves a stamped but short root -- which build() rebuilds from
    # scratch on the next submission, because its row count is below n_train -- rather than
    # an unstamped one this script would then refuse forever.
    want_stamp > "$STAMP.tmp.$$" && mv -f "$STAMP.tmp.$$" "$STAMP" \
      || { say "  STOP: cannot write $STAMP"; exit 1; }
    say "  new root; wrote $STAMP"
  fi
fi

# python3 -I: this streams LLaVA-ReCap-CC3M and decodes 340 PNGs from it, which AGENTS.md
# rule 9 calls untrusted data. It prints the procedure and the root BEFORE build() is
# entered, because build()'s own "procedure WS=" line is printed only on the path that
# builds -- the path that reuses returns before reaching it, so the log of a reusing cell
# never said which demos it was about to train on.
step "  build or reuse the demos" \
  env WS="$WS" MVLM_DATA="$DATA" MVLM_CODE="$CODE" \
      N_TRAIN="$N_TRAIN" N_HOLDOUT="$N_HOLDOUT" MIN_WORDS="$MIN_WORDS" \
      python3 -I -c 'import os, sys
sys.path.insert(0, os.environ["MVLM_CODE"])
import mvlm_data
print("demos: procedure WS=%s  root=%s  n_train=%s n_holdout=%s min_words=%s"
      % (mvlm_data.WS, os.environ["MVLM_DATA"], os.environ["N_TRAIN"],
         os.environ["N_HOLDOUT"], os.environ["MIN_WORDS"]), flush=True)
mvlm_data.build(os.environ["MVLM_DATA"], int(os.environ["N_TRAIN"]),
                int(os.environ["N_HOLDOUT"]), int(os.environ["MIN_WORDS"]))' \
  || { say "  demo build FAILED; nothing downstream can run. Stopping."; exit 1; }

# Lock released here, so the cells train in parallel.
[ "$DRYRUN" = 1 ] || exec 9>&-

# ---------------------------------------------------------------- 3. phase A: train
say "=== 3. phase A: train af and db under WS=$WS, $STEPS steps, seed $SEED ==="
# The adapter that EXISTS is not the adapter that FINISHED. This partition preempts, and a
# preemption inside save_pretrained leaves a truncated but NON-EMPTY
# adapter_model.safetensors, which the `[ -s ... ]` this used to test accepts: the cell then
# skips training on every resubmission, the two md5s still differ so section 4 passes, and
# the gate fails on half a tensor forever. mvlm_stage1.py:223 writes adapter_<arm>/SAVE_OK
# last, after the save has landed, so the sentinel is what says the save completed and the
# size says nothing. The name and the file are its contract, not this script's; it also
# guarantees the sentinel is non-empty, hence -s rather than -f.
#
# The weights are still required beside it, so that deleting them to reclaim space retrains
# instead of stopping at the md5 check in section 4.
#
# Being wrong in this direction is cheap. This test only decides whether to START
# mvlm_stage1.py; which arms it retrains is its own decision, per arm, and an adapter that
# predates the sentinel is verified and adopted there (mvlm_stage1.py:261) rather than
# retrained, so the cost of a missing SAVE_OK on a complete adapter is one process start.
SENTINEL=SAVE_OK
if [ -s "$AFA/$SENTINEL" ] && [ -s "$AFA/adapter_model.safetensors" ] &&
   [ -s "$DBA/$SENTINEL" ] && [ -s "$DBA/adapter_model.safetensors" ]; then
  say "  both adapters carry $SENTINEL, so both saves completed; skipping training"
else
  have_time "$MIN_PHASE_A" || exit 0
  # One process, both arms. MVLM_SPLIT=0 skips cutting the adapter into 30MB parts -- there
  # is no Colab download to feed and the file is already where the gate will read it -- but
  # mvlm_stage1.py:117-133 still prints "adapter_<arm>: N parts, md5=<hex>" on that path,
  # because the md5 is the correctness check section 4 below depends on, not transfer
  # machinery. MIN_WORDS is passed so its own mvlm_data.build() call agrees with the stamp
  # section 2 wrote; the root is warm by now, so that call takes the reuse path.
  #
  # TIME_BUDGET, like every other phase. mvlm_stage1.py:144-154 has a per-arm guard that
  # refuses to START an arm it cannot finish, and without this it is dead code, because its
  # BUDGET defaults to 0 and 0 means no limit. The have_time above is ONE check taken for
  # BOTH arms: a cell that enters phase A just over the MIN_PHASE_A floor and spends most of
  # it on af then enters db with too little and is SIGKILLed mid-train -- and an arm killed
  # at step 90 of 150 is a different adapter, not a partial one, so there is nothing to
  # resume from. With the budget passed, the second arm is simply not started and the next
  # submission trains it from the first step.
  step "  training" \
    env TAG="$TAG" WS="$WS" ARMS=af,db STEPS="$STEPS" SEED="$SEED" N_TRAIN="$N_TRAIN" \
        N_HOLDOUT="$N_HOLDOUT" MIN_WORDS="$MIN_WORDS" LORA_SCOPE="$LORA_SCOPE" \
        MVLM_ROOT="$REPO/$ADIR" MVLM_DATA="$DATA" MVLM_CODE="$CODE" \
        TIME_BUDGET="$(budget)" \
        MVLM_PIP=0 MVLM_SPLIT=0 \
        python3 -I scripts/vlm/mvlm_stage1.py \
    || { say "  PHASE A FAILED; nothing downstream can run. Stopping."; exit 1; }
fi

# ---------------------------------------------------------------- 4. the adapters differ
say "=== 4. hard check: the two adapters are not the same file ==="
if [ "$DRYRUN" = 1 ] &&
   { [ ! -s "$AFA/adapter_model.safetensors" ] || [ ! -s "$DBA/adapter_model.safetensors" ]; }; then
  printf '[dry] md5sum %s\n' "$AFA/adapter_model.safetensors" "$DBA/adapter_model.safetensors"
  printf '[dry]   -> exit 1 if the two md5s are equal\n'
else
  MD5_AF="$(md5sum "$AFA/adapter_model.safetensors" 2>/dev/null | cut -d' ' -f1)"
  MD5_DB="$(md5sum "$DBA/adapter_model.safetensors" 2>/dev/null | cut -d' ' -f1)"
  say "  af=${MD5_AF:0:12} db=${MD5_DB:0:12}"
  # run_mvlm.sh:248-250 only printed these two lines and left the reading to a human.
  # next_step.md's failure list says an af and a db adapter with the same md5 means the arm
  # split failed and every number from the run is the same adapter measured twice -- a
  # complete-looking table with no control in it. That is not something to notice at 3am, so
  # it is fatal here.
  if [ -z "$MD5_AF" ] || [ -z "$MD5_DB" ]; then
    say "  STOP: an adapter is missing. af and db must both exist before the gate."
    exit 1
  fi
  if [ "$MD5_AF" = "$MD5_DB" ]; then
    say "  STOP: af and db have the SAME md5 ($MD5_AF)."
    say "        The arm split failed: both arms hold one adapter, so AF minus DB is zero by"
    say "        construction and every number from this cell would be the same model twice."
    say "        Delete $ADIR and retrain; do not generate on these."
    exit 1
  fi
fi

# ---------------------------------------------------------------- 5. benign gate
# Before any harmful row, and gate_ok.py enforces it in section 6. AGENTS.md rule 1 and D33:
# a backbone whose gate is incomplete has unreadable harmful rows, because a low rate cannot
# be told apart from an adapter that trains to produce nothing.
say "=== 5. benign gate on $N_HOLDOUT held-out images, BEFORE any harmful row (D33) ==="
# Probed before anything is loaded. A resubmitted cell whose six gate files are already full
# used to load the backbone and walk the gate again before section 6 got to say "complete":
# mvlm_sanity.py skips the rows by count so it wrote nothing, but a 4-bit load is minutes of
# an A100 this job is holding, and a cell is resubmitted several times by design. Reading
# six row counts costs milliseconds and needs no GPU. The probe is handed exactly the
# environment section 6 uses, so the two cannot disagree about what "complete" means.
#
# N="$N_HOLDOUT", and the two names are the reason this is spelled out. gate_ok.py:26 reads
# N and defaults it to 40; this script sizes the gate with N_HOLDOUT and used to pass
# neither, so the hard gate measured a constant while the gate itself measured a variable.
# On a 20-row gate the invocation printed GATE_INCOMPLETE 6 of 6 and the cell could never
# reach generation, because 20 rows is all there will ever be; above 40 it would pass a gate
# that is still short. It is latent only because every documented command leaves N_HOLDOUT
# at its default.
GATE_FULL=0
if [ "$DRYRUN" = 1 ]; then
  printf '[dry] probe the gate before loading anything; skip the whole phase if it is full\n'
  printf '        %q\n' env TAG="$TAG" SAN_DIR="$GDIR" N="$N_HOLDOUT" python3 -I scripts/vlm/gate_ok.py
elif env TAG="$TAG" SAN_DIR="$GDIR" N="$N_HOLDOUT" python3 -I scripts/vlm/gate_ok.py >/dev/null 2>&1; then
  GATE_FULL=1
  say "  already complete on disk; not loading a backbone for it (section 6 re-checks)"
fi
if [ "$GATE_FULL" = 0 ]; then
  have_time || exit 0
  # AF and DB are passed explicitly. mvlm_sanity.py:89-90 defaults them to V/adapter_*, and V
  # here is the gate OUTPUT directory -- outputs/vlm2_sanity_<tag><SUF> -- which is a different
  # place from the adapter store. On Colab the two happened to coincide because V was the
  # per-tag working root that also held the adapters; on SLURM they do not, and the default
  # would load nothing and silently gate the base model three times.
  step "  gate" \
    env TAG="$TAG" WS="$WS" ARMS="$ARMS" V="$REPO/$GDIR" AF="$AFA" DB="$DBA" \
        BS="$GATE_BS" N_TRAIN="$N_TRAIN" N_HOLDOUT="$N_HOLDOUT" TIME_BUDGET="$(budget)" \
        MVLM_DATA="$DATA" MVLM_CODE="$CODE" MVLM_PIP=0 \
        python3 -I scripts/vlm/mvlm_sanity.py \
    || say "  the gate exited non-zero; section 6 decides whether that matters"
fi

# ---------------------------------------------------------------- 6. hard gate
say "=== 6. hard gate ==="
# SAN_DIR is mandatory, not a nicety. gate_ok.py:27 defaults to outputs/vlm2_sanity_<TAG>
# with no suffix, so omitting it reads the PUBLISHED worksheet's gate for this backbone --
# which is complete -- and waves through a chain or front cell whose own gate is empty.
# N for the reason in section 5: it is this script's N_HOLDOUT, not gate_ok.py's 40.
step "  gate_ok" \
  env TAG="$TAG" SAN_DIR="$GDIR" N="$N_HOLDOUT" python3 -I scripts/vlm/gate_ok.py \
  || { say "  gate incomplete: NOT generating. Its rate would mean nothing (D33)."
       say "  The cells resume. Resubmit this cell and it fills only what is missing."
       exit 1; }

# ---------------------------------------------------------------- 7. phase B: generate
say "=== 7. phase B: harmful rows, three arms in one model load ==="
have_time || exit 0
# SFX="" on purpose; see THE SFX TRAP in the header. V carries the condition.
# EVALF is $EVAL, already absolute: the preflight above counted the rows of this exact path,
# which is the whole point of resolving it once at the top.
#
# REQUEST_SLOT is the one environment variable here that changes what is STORED rather than
# which file it is stored in, so it has to be right now. mvlm_gen.py:315-317 writes
# "response": step1(resp) with a slot-aware step1(), and at the default 1 a front row's
# 'response' holds "Describe what you see in the image" answered about a gray square instead
# of the answer to the request. AGENTS.md rule 4 forbids repairing that column afterwards.
# It is derived from WS at the top of this script and printed in the header; an operator
# never types it for a cell, only for the readouts, which are separate commands.
step "  generate" \
  env TAG="$TAG" ARMS="$ARMS" V="$REPO/$VDIR" AF="$AFA" DB="$DBA" \
      BS="$BS" SFX="" EVALF="$EVAL" TIME_BUDGET="$(budget)" \
      REQUEST_SLOT="$REQUEST_SLOT" \
      MVLM_CODE="$CODE" MVLM_PIP=0 \
      python3 -I scripts/vlm/mvlm_gen.py \
  || say "  generation exited non-zero; the judges below skip any arm that is short"

# ---------------------------------------------------------------- 8. judges
say "=== 8. judges: lg, qg, then the request-free pass ==="

# Only when every arm is at 260, which is what run_mvlm.sh:375 did. A guard load costs about
# two minutes and the cell will be resubmitted anyway, so there is nothing to gain by judging
# a half-written arm, and a partly judged condition is one more thing to read carefully in a
# table where a reader is already being asked to check row counts. The judges resume by row,
# so this costs nothing on the next submission.
N_EVAL="$(rows "$EVAL")"
complete(){
  local a
  for a in ${ARMS//,/ }; do
    [ "$(rows "$VDIR/res_${a}.jsonl")" -ge "$N_EVAL" ] || return 1
  done
  return 0
}
if [ "$DRYRUN" != 1 ] && ! complete; then
  say "  generation is short, so not judging yet:"
  for a in ${ARMS//,/ }; do say "    $a: $(rows "$VDIR/res_${a}.jsonl")/$N_EVAL"; done
  say "  Resubmit this cell; generation resumes and the judges run once all arms are full."
  say "CELL_DONE ${TAG}${SUF} (generation incomplete)"
  exit 0
fi

# NEUTRAL_USER was resolved and checked in the preflight, before the twelve hours: that it
# exists at all, and that it is the WS=$WS template with the request deleted rather than
# some other arm of run_neutral_judge.sh's case block that happened to match
# _gray_${TAG}${SUF} first. Only the echo is left here.
say "  request-free user turn: $(printf '%s' "$NEUTRAL_USER" | head -2 | tr '\n' ' ')..."

# The order is binding. qg appends qg_verdict to the file lg wrote (judge_multi.py:82-83 reads
# res_<arm><IN_SFX> and writes res_<arm><OUT_SFX>), so lg first and qg second, and the same
# chain again for the request-free pass. Running qg first leaves two files that each carry one
# verdict and no row that carries both, which is what stats_protocol.py joins on.
judge(){   # $1 label, $2 JUDGE, $3 IN_SFX, $4 OUT_SFX, $5 NEUTRAL_USER
  step "  judge $1" \
    env V="$REPO/$VDIR" ARMS="$ARMS" SFX="" JUDGE="$2" IN_SFX="$3" OUT_SFX="$4" \
        NEUTRAL_USER="$5" JUDGE_FIELD=raw BS="$JUDGE_BS" TIME_BUDGET="$(budget)" \
        MVLM_PIP=0 \
        python3 -I scripts/vlm/judge_multi.py \
    || say "  judge $1 exited non-zero; it resumes from the rows already judged"
}
have_time && judge "lg  (raw protocol)"          lg ""     _lg  ""
have_time && judge "qg  (raw protocol)"          qg _lg    _qg  ""
have_time && judge "lg  (request-free protocol)" lg ""     _nlg "$NEUTRAL_USER"
have_time && judge "qg  (request-free protocol)" qg _nlg   _nqg "$NEUTRAL_USER"

# ---------------------------------------------------------------- final
say "=== FINAL: $TAG$SUF ==="
N_EVAL="$(rows "$EVAL")"
for a in ${ARMS//,/ }; do
  say "  $a: gen $(rows "$VDIR/res_${a}.jsonl")/$N_EVAL  lg $(rows "$VDIR/res_${a}_lg.jsonl")  qg $(rows "$VDIR/res_${a}_qg.jsonl")  nlg $(rows "$VDIR/res_${a}_nlg.jsonl")  nqg $(rows "$VDIR/res_${a}_nqg.jsonl")"
done
for a in ${ARMS//,/ }; do
  say "  $a gate: ws $(rows "$GDIR/san_${a}_ws.jsonl")/$N_HOLDOUT  plain $(rows "$GDIR/san_${a}_plain.jsonl")/$N_HOLDOUT"
done
say "CELL_DONE ${TAG}${SUF}"
