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
R=$SCALE/${TAG}_steer
say(){ echo "[steer] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python

[ -s "$SCALE/probe/p2b_${TAG}_fit.pt" ] \
  || { say "STOP: no P2b fit for $TAG -- phase 1 must run first"; exit 1; }

say "generate, alphas=${ALPHAS:-0,0.5,1,2} MAXNEW=$MAXNEW ($PIN)"
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
for A in a0 a0p5 a1 a2; do
  [ -s "$R/res_af_$A.jsonl" ] || continue
  say "judge $A ($(wc -l < "$R/res_af_$A.jsonl") rows)"
  V=$R WS=mix ARMS=af SFX="_$A" \
    $JPY -I scripts/vlm/judge_stream.py >>"$R/judge_$A.log" 2>&1 \
    || say "judge $A returned non-zero (see judge_$A.log)"
done
say "STEER_JOB_DONE"
