#!/bin/bash
#SBATCH --job-name=pcvlm-wave
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --time=12:00:00
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/wave-%j.out
#SBATCH --error=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/wave-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# Run several cells side by side in ONE allocation, one cell per GPU.
#
# WHY THIS EXISTS RATHER THAN ONE cell.sbatch PER CELL. A cell needs exactly one GPU -- a
# 4-bit 12B or 32B backbone is 7-16GB on an 80GB card -- but this QOS will not accept a
# one-GPU job:
#
#     $ sbatch --gres=gpu:1 ... cell.sbatch
#     sbatch: error: QOSMinGRES
#     $ sacctmgr show qos a100_genai_interns_high format=MinTRES
#     gres/gpu=2
#
# So the floor is two GPUs per job. Submitting each cell as its own two-GPU job would idle
# one card per cell, five in total. Packing N cells into one N-GPU allocation satisfies the
# floor and wastes nothing, and the cells are independent -- every one writes to its own
# adapter, gate and output directory, keyed by its own tag and suffix.
#
# --cpus-per-gpu / --mem-per-gpu rather than fixed totals, so the request scales with --gres
# and three cells cannot be handed one cell's worth of CPU. 12 and 120G per cell are
# cell.sbatch's numbers, sized for the demo build (mvlm_data.py streams LLaVA-ReCap-CC3M and
# re-saves 340 PNGs before a single weight loads) rather than for the model.
#
# Usage -- --gres must match the number of cells:
#
#   sbatch --gres=gpu:3 --export=ALL,CELLS="lv11:front g12:chain q3vl32:chain" \
#          scripts/slurm/wave.sbatch
#
# Each entry is TAG:WS. Everything else -- SUF, EVAL, REQUEST_SLOT -- run_cell.sh derives
# from WS, for the reason in its own header: they are one choice, not four, and a cell
# submitted as WS=chain,SUF=_front is a clean, confident, wrong plan.
#
# RESUBMITTING IS THE NORMAL WAY TO FINISH. Every stage resumes by row count and run_cell.sh
# stops at a row boundary before the walltime, so a wave that runs out of time is submitted
# again with the identical command. Never clear a directory and start over (AGENTS.md rule 2).
set -u

# The login shell exports http(s)_proxy=10.0.2.2:<port>, the LOGIN node's host-NAT address.
# Nothing listens on it from a compute node, so the dataset stream would hang until it timed
# out, which looks exactly like a slow model load.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO="${REPO:-/home/hoangpv/implement/procedural-capture-vlm}"
FSX=/genai/fsx-llm/interns/hoangpv/pcvlm
ENV_DIR="${ENV_DIR:-$FSX/env}"
SP="${SP:-$FSX/sp}"

say(){ echo "[wave] $(date '+%m-%d %H:%M:%S') $*"; }

: "${CELLS:?set CELLS, e.g. --export=ALL,CELLS=\"lv11:front g12:chain\"}"
cd "$REPO" || { say "no repo at $REPO"; exit 1; }
mkdir -p "$SP/log"

# Same marker test as cell.sbatch, and for the same reason: a FAILED env build leaves an
# executable interpreter with the right transformers on top of a torch that cannot target an
# A100, because the venv is populated before the sm_80 assertion runs. The marker is written
# last, after every check has passed.
if [ ! -s "$ENV_DIR/ENV_READY" ]; then
  say "no usable env at $ENV_DIR. Run: sbatch scripts/slurm/env.sbatch"
  say "  (an env with no ENV_READY marker failed a check and is rebuilt there without ENV_FORCE)"
  exit 1
fi
say "env ready: $(head -1 "$ENV_DIR/ENV_READY")"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export HF_HOME=/genai/fsx-llm/interns/hoangpv/envs/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0
# AGENTS.md rule 8: .hf is a credential. Read into the environment, never echoed -- not the
# value and not its length either, since a length is a property of the secret and this log is
# appended to for twelve hours and read by several people.
if [ -z "${HF_TOKEN:-}" ] && [ -s "$REPO/.hf" ]; then
  HF_TOKEN="$(cat "$REPO/.hf")"; export HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] && say "HF token loaded"

# The devices this job actually holds. Slurm sets CUDA_VISIBLE_DEVICES to the allocated set
# when ConstrainDevices is on, which it is cluster-wide; the fallback covers the case where
# it is not, and the count check below refuses to start if the two disagree.
DEVS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a DEV <<< "$DEVS"
IFS=' ' read -r -a CELL <<< "$CELLS"
say "job ${SLURM_JOB_ID:-none} on $(hostname): ${#CELL[@]} cells, ${#DEV[@]} GPUs ($DEVS)"

if [ "${#CELL[@]}" -gt "${#DEV[@]}" ]; then
  say "STOP: ${#CELL[@]} cells but only ${#DEV[@]} GPUs. Two cells on one card would make each"
  say "      other OOM at the measured batch sizes, and the batch sizes are not negotiable"
  say "      (changing one changes left-padding composition and flips near-tie greedy tokens)."
  say "      Resubmit with --gres=gpu:${#CELL[@]}."
  exit 1
fi

pids=(); names=()
for i in "${!CELL[@]}"; do
  spec="${CELL[$i]}"
  tag="${spec%%:*}"; ws="${spec##*:}"
  if [ "$tag" = "$spec" ] || [ -z "$tag" ] || [ -z "$ws" ]; then
    say "STOP: cell '$spec' is not TAG:WS"; exit 1
  fi
  log="$SP/log/wave_${tag}_${ws}.log"
  say "launching $tag/$ws on GPU ${DEV[$i]} -> $log"
  # One device each. mvlm_arch.py:162 loads with device_map="auto", so a cell that could see
  # two cards would spread one 4-bit backbone over both and collide with its neighbour.
  # run_cell.sh's preflight independently refuses to start on anything but exactly one A100.
  CUDA_VISIBLE_DEVICES="${DEV[$i]}" TAG="$tag" WS="$ws" \
    bash "$REPO/scripts/slurm/run_cell.sh" > "$log" 2>&1 &
  pids+=("$!"); names+=("$tag/$ws")
done

# Wait on each explicitly rather than a bare `wait`, so one cell's exit code is reported
# against its own name. A bare wait returns the last one and the others vanish.
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    say "${names[$i]}: finished ok"
  else
    s=$?; rc=1
    say "${names[$i]}: FAILED (exit $s) -- see $SP/log/wave_${names[$i]/\//_}.log"
  fi
done

say "per-cell tails:"
for n in "${names[@]}"; do
  f="$SP/log/wave_${n/\//_}.log"
  [ -f "$f" ] && { echo "  --- $n ---"; tail -3 "$f" | sed 's/^/  /'; }
done
say "WAVE_DONE rc=$rc"
exit $rc
