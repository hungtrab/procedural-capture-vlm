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
# Each entry is TAG:WS[:SUF[:SEED[:EVAL]]]. Give only TAG:WS and run_cell.sh derives the rest
# from WS, which is round 1's behaviour and the reason its header argues they are one choice
# and not four: a cell submitted as WS=chain,SUF=_front is a clean, confident, wrong plan.
#
# Round 2 needs the longer form, because several of its cells share a backbone AND a worksheet
# and differ only in suffix, seed or eval -- q3 under the count worksheet appears five times,
# as _s42, _s1, _s2, _rep and _realimg. run_cell.sh still checks the combination: it accepts a
# seed replication (suffix _s<n> with SEED=n) or an eval-only variant (every field of the eval
# but image and category byte-identical to the published one) and refuses anything else.
#
#   sbatch --gres=gpu:8 --export=ALL,CELLS="\
#     q3:count:_s42:42:data/eval_harmful_gray.jsonl \
#     lv11:count:_rep:42:data/eval_harmful_gray.jsonl" scripts/slurm/wave.sbatch
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

# A cell spec is TAG:WS[:SUF[:SEED[:EVAL]]]. Round 1 only ever needed the first two, because a
# condition was one worksheet and run_cell.sh derived everything else from it. Round 2 breaks
# that: two cells of the same backbone under the SAME worksheet differ only by suffix, seed or
# eval -- q3 count appears as _s42, _s1, _s2, _rep and _realimg -- so the spec has to carry
# them. Omitted fields are left unset and run_cell.sh derives them exactly as before, which is
# why a two-field spec still behaves identically to round 1.
#
# EVAL is here rather than derived because the published count condition is stored with the
# gray eval, not an eval_harmful_count.jsonl that has never existed; run_cell.sh's eval-only
# variant case checks the file's CONTENT against the published eval, so naming it is safe.
pids=(); names=()
for i in "${!CELL[@]}"; do
  spec="${CELL[$i]}"
  IFS=':' read -r tag ws suf seed evl <<< "$spec"
  if [ -z "${tag:-}" ] || [ -z "${ws:-}" ]; then
    say "STOP: cell '$spec' is not TAG:WS[:SUF[:SEED[:EVAL]]]"; exit 1
  fi
  # The log name has to separate cells that share a tag and a worksheet, or five q3 count
  # cells would write to one file and the first thing anyone read would be interleaved.
  label="${tag}${suf:+$suf}"
  log="$SP/log/wave_${label}.log"
  say "launching $label (ws=$ws${suf:+ suf=$suf}${seed:+ seed=$seed}${evl:+ eval=$evl}) on GPU ${DEV[$i]} -> $log"
  # One device each. mvlm_arch.py:162 loads with device_map="auto", so a cell that could see
  # two cards would spread one 4-bit backbone over both and collide with its neighbour.
  # run_cell.sh's preflight independently refuses to start on anything but exactly one A100.
  #
  # Each field is exported only when the spec supplied it. Passing SUF= or SEED= empty is not
  # the same as leaving them unset: run_cell.sh reads SUF with ${SUF-...} precisely so that an
  # explicitly empty SUF means the published condition, and handing it one by accident would
  # aim a round-2 cell at the published directories.
  ( export CUDA_VISIBLE_DEVICES="${DEV[$i]}" TAG="$tag" WS="$ws"
    [ -n "${suf:-}" ]  && export SUF="$suf"
    [ -n "${seed:-}" ] && export SEED="$seed"
    [ -n "${evl:-}" ]  && export EVAL="$evl"
    exec bash "$REPO/scripts/slurm/run_cell.sh" ) > "$log" 2>&1 &
  pids+=("$!"); names+=("$label")
done

# Wait on each explicitly rather than a bare `wait`, so one cell's exit code is reported
# against its own name. A bare wait returns the last one and the others vanish.
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    say "${names[$i]}: finished ok"
  else
    s=$?; rc=1
    say "${names[$i]}: FAILED (exit $s) -- see $SP/log/wave_${names[$i]}.log"
  fi
done

say "per-cell tails:"
for n in "${names[@]}"; do
  f="$SP/log/wave_${n}.log"
  [ -f "$f" ] && { echo "  --- $n ---"; tail -3 "$f" | sed 's/^/  /'; }
done
say "WAVE_DONE rc=$rc"
exit $rc
