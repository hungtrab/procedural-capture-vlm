#!/usr/bin/env bash
#SBATCH --job-name=r4_steer
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/steer-%j.out
#
# Round 4 P2c: steering along the refusal direction, then reading the result with the same
# three readers as every other row in the round.
#
#   sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl8 scripts/vlm/scale/steer.sbatch
#
# Generation first, all four alphas, then one judge pass per alpha. The judges run AFTER
# rather than streamed because the four alphas write four separate transcripts from one
# model load, and judge_stream.py follows one arm's file at a time.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:-q3vl8}
export MAXNEW=${MAXNEW:-512}
# Normalised ONCE, here, and exported: steer_gen.py reads ALPHAS with os.environ.get, whose
# default applies only when the variable is UNSET, while ${ALPHAS:-...} also covers the
# exported-but-empty case. Without this the two disagree exactly then -- steer_gen would
# generate nothing and the judge loop would judge nothing, both silently.
export ALPHAS=${ALPHAS:-0,0.5,1,2}
R=$SCALE/${STEER_CELL:-${TAG}_steer}
say(){ echo "[steer] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python

[ -s "$SCALE/probe/p2b_${TAG}_fit.pt" ] \
  || { say "STOP: no P2b fit for $TAG -- phase 1 must run first"; exit 1; }

# The redirect below opens "$R/run.log" in THIS shell, before steer_gen.py runs its own
# os.makedirs. For a tag with no steer cell yet (q3vl32) that redirect fails and the job
# dies as "FAIL gen" with an empty log. Create it here.
mkdir -p "$R"
say "generate, alphas=$ALPHAS MAXNEW=$MAXNEW ($PIN)"
$PY -I scripts/vlm/scale/steer_gen.py >>"$R/run.log" 2>&1 || { say "FAIL gen"; exit 1; }

# One judge pass per alpha. SFX selects the transcript; ARMS=af because every steered row is
# the AF arm.
#
# GEN_DONE is a PATH, not a flag -- judge_stream.py does os.path.exists on it -- so the file
# is created rather than the variable set to 1, which would have it look for a file called
# "1". With it present `final` is true from the first iteration, so partial batches are
# judged and the loop exits as soon as the guards catch up. That matters here because these
# transcripts hold 260 rows, not the 1520 any eval file would make EXPECT.
# WS=mix only satisfies judge_stream's EVALF fallback, which KeyErrors on an unset WS.
rm -f "$R/GEN_FAIL"; : > "$R/GEN_DONE"
# Derived from ALPHAS rather than hardcoded: rev 6 M1 adds alpha=4 for q3vl32, and the
# old literal list silently left those transcripts unjudged. SFX appends the eval suffix so
# an XSTest sweep judges its own files, not the gray ones.
ATAGS=$(python3 -I -c "
import os
print(' '.join(('a%g'%float(a)).replace('.','p')
      for a in os.environ['ALPHAS'].split(',') if a.strip()))")
for A in $ATAGS; do
  F="$R/res_af_$A${SFX:-}.jsonl"
  [ -s "$F" ] || continue
  say "judge $A${SFX:-} ($(wc -l < "$F") rows)"
  V=$R WS=mix ARMS=af SFX="_$A${SFX:-}" \
    $JPY -I scripts/vlm/judge_stream.py >>"$R/judge_$A.log" 2>&1 \
    || say "judge $A returned non-zero (see judge_$A.log)"
done
say "STEER_JOB_DONE"
