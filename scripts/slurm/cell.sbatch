#!/bin/bash
#SBATCH --job-name=pcvlm-cell
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/cell-%j.out
#SBATCH --error=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/cell-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# THE TWO LINES ABOVE ARE ABOUT PREEMPTION, AND BOTH ARE LOAD-BEARING. The a100 partition
# runs PreemptMode=REQUEUE with JobFileAppend=0. Left alone, a preempted cell comes back
# under the SAME job id, so it reopens cell-%j.out -- and with JobFileAppend=0 it reopens it
# truncating. The log of the first attempt, which is where the preemption and the row counts
# at the time are recorded, is destroyed by the retry. --open-mode=append makes the second
# attempt add to the file instead of replacing it.
#
# --no-requeue then stops the automatic retry happening at all: a preempted cell dies and is
# resubmitted by hand. That is already the documented way to finish a cell -- every stage
# resumes by row count and a cell routinely needs more than one submission -- so this costs
# nothing and buys the certainty that a job id names one attempt. An automatic requeue is
# worse than a manual one here because it starts over with no one reading the log, and the
# phase it restarts into is decided by what is on disk, which a human should look at.
#
# Submit ONE (backbone, template) cell. All this file does is put the environment in the
# state run_cell.sh expects and then hand over; every decision about what to run and in what
# order is in run_cell.sh.
#
#   mkdir -p /genai/fsx-llm/interns/hoangpv/pcvlm/sp/log
#   sbatch --export=ALL,TAG=g12,WS=chain scripts/slurm/cell.sbatch
#
# TAG and WS, and nothing else. SUF, EVAL and REQUEST_SLOT follow from WS and run_cell.sh
# derives all three; see ONE WORKSHEET, ONE CONDITION in its header for the three wrong
# plans that typing them separately used to produce.
#
# ONE GPU, NOT EIGHT. A 4-bit 12B or 32B backbone is about 7-16GB and fits one A100-80GB with
# room for the KV cache at the measured batch sizes, so a cell is one GPU and five cells run
# side by side inside the 64-GPU / 10-job ceiling of a100_genai_interns_high.
#
# 12 CPUs and 120GB are sized for the demo build, not the model: mvlm_data.py:124 streams
# LLaVA-ReCap-CC3M and decodes and re-saves 340 PNGs before any weight is loaded.
#
# RESUBMITTING IS THE NORMAL WAY TO FINISH A CELL. run_cell.sh stops at a row boundary ten
# minutes before the walltime and every stage resumes by row count, so a cell that needs more
# than twelve hours is submitted again with the identical command. Never clear a directory
# and start over (AGENTS.md rule 2).
set -u

# The login shell exports http(s)_proxy=10.0.2.2:<port>, the LOGIN node's host-NAT address.
# Nothing listens on it from a compute node, so the checkpoint fetch and the dataset stream
# would both hang until they timed out, which looks exactly like a slow model load.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO="${REPO:-/home/hoangpv/implement/procedural-capture-vlm}"
FSX=/genai/fsx-llm/interns/hoangpv/pcvlm
ENV_DIR="${ENV_DIR:-$FSX/env}"

say(){ echo "[sbatch] $(date '+%m-%d %H:%M:%S') $*"; }

: "${TAG:?set TAG, e.g. --export=ALL,TAG=g12}"
: "${WS:?set WS: count, chain or front}"
# SUF and EVAL are NOT required here any more, and REQUEST_SLOT is not accepted at all.
# run_cell.sh derives all three from WS, because they are one choice and not four: a cell
# submitted as WS=chain,SUF=_front is a clean, confident, wrong plan that nothing downstream
# can detect. Pass SUF or EVAL only to override, and then run_cell.sh wants
# ALLOW_WS_MISMATCH=1 as well and prints what it is overriding.
#
# Read with ${SUF-...} and never assigned here: this script execs run_cell.sh with its own
# environment, so writing a placeholder into SUF would hand that placeholder straight to the
# directory names.
cd "$REPO" || { say "no repo at $REPO"; exit 1; }
say "job ${SLURM_JOB_ID:-none} on $(hostname): TAG=$TAG WS=$WS" \
    "SUF=${SUF-(derived from WS)} EVAL=${EVAL-(derived from WS)}"

# The env scripts/slurm/env.sbatch built. Its interpreter lives on FSx too, so this venv is
# valid from any node -- the node images are not identical and /opt/conda is not everywhere.
#
# The test is env.sbatch's ENV_READY marker and not an executable interpreter, because an
# executable interpreter is exactly what a FAILED build leaves behind: the venv is created
# and populated before the sm_80 arch assertion runs, so a torch that cannot target an A100
# still gives a working $ENV_DIR/bin/python with the pinned transformers in it. This job
# would then load a 4-bit backbone and find out at the first kernel, twelve hours of
# allocation later. The marker is written last, after every check has passed.
if [ ! -s "$ENV_DIR/ENV_READY" ]; then
  if [ -x "$ENV_DIR/bin/python" ]; then
    say "the env at $ENV_DIR has no ENV_READY marker, so its build did not pass every check"
    say "  (the sm_80 arch assertion is the one that matters to this job). Not starting."
    say "  Rebuild it: sbatch scripts/slurm/env.sbatch -- an unmarked env is treated as"
    say "  incomplete there and rebuilt without ENV_FORCE."
  else
    say "no env at $ENV_DIR; run: sbatch scripts/slurm/env.sbatch"
  fi
  exit 1
fi
say "env ready: $(head -1 "$ENV_DIR/ENV_READY")"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
say "python $(command -v python3)  transformers $(python3 -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1)"

# Every checkpoint this run needs is already here (scripts/slurm/prefetch_assets.sbatch), so
# no GPU minute is spent downloading weights. hf_transfer is off for the same reason it is
# off there: it adds a dependency and the plain client already measured 147MB/s.
export HF_HOME=/genai/fsx-llm/interns/hoangpv/envs/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0

# AGENTS.md rule 8: .hf is a credential. It is read into the environment and never echoed,
# and the python scripts print only which source it came from. Exported rather than left to
# each script so all five processes in this job agree on one token.
if [ -z "${HF_TOKEN:-}" ] && [ -s "$REPO/.hf" ]; then
  HF_TOKEN="$(cat "$REPO/.hf")"
  export HF_TOKEN
fi
# Says THAT a token is loaded and nothing else. Not its value and not its length either:
# AGENTS.md rule 8 makes .hf a credential, and a length is a property of the secret that goes
# into a log file this job appends to for twelve hours and that several people read.
[ -n "${HF_TOKEN:-}" ] && say "HF token loaded"

# Exactly one visible card. mvlm_arch.py:162 loads with device_map="auto", which shards the
# model over every device it can see; with all eight visible it would spread one 4-bit
# backbone across the node, hold seven GPUs this job did not ask for, and run the forward
# pass over NVLink for nothing. Slurm narrows this itself when ConstrainDevices is on, so the
# usual value here is already "0"; the fallback and the comma trim are for the case where it
# is not, and run_cell.sh's preflight refuses to start if either misses.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$CUDA_VISIBLE_DEVICES" in
  *,*) say "trimming CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES to its first device"
       export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" ;;
esac
say "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"

# exec, so the cell is this job's process rather than a child of it: scancel then reaches the
# python it is actually waiting on, and the SIGKILL at the walltime lands in one place.
exec bash "$REPO/scripts/slurm/run_cell.sh"
