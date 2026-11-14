#!/usr/bin/env bash
#SBATCH --job-name=rt42
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --output=outputs/scale/slurm_rt42_%j.out
#
# Retrain the seed-42 AF adapters that the 2026-11-11 offload destroyed, so M1 (steering as
# a mitigation) has something to steer. TRAIN ONLY -- no gate, no generation.
#
#   sbatch scripts/vlm/scale/retrain42.sbatch
#
# WHY TRAIN-ONLY, AND WHY A NEW DIRECTORY.
# Running the full cell would regenerate ~9 h of harm rows that already exist and are
# already handed back. The only thing missing is the adapter. And "Resume, never restart.
# Every new cell goes in its own directory. No round-3 or seed directory is written to."
# forbids writing into q3vl8_mix / q3vl32_mix, so the new weights land in <tag>_mix_rt42.
#
# WHAT THESE ADAPTERS ARE, AND ARE NOT.
# They are NOT the round-3 adapters. Training is not bitwise reproducible on this hardware,
# which is the entire reason the p0 arm exists, and p0 measures the size of it: a retrain
# with zero refusal demos moved the gray unsafe rate by LG +8.5pp, n=260, p=5.95e-05. So:
#   - their md5s will NOT match round4_adapter_md5.txt, and must never be presented as if
#     they do;
#   - no row generated from them may be paired against a row from the destroyed adapters.
#     Any comparison must be internal: alpha=2 against an alpha=0 regenerated HERE. That is
#     what makes M1 still answerable -- every one of its thresholds is "vs alpha=0 at the
#     same setting", so the retrain noise cancels.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export MVLM_PIP=0 PRECISION=bf16

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
say(){ echo "[rt42] $(date -u +%F' '%T) $*" | tee -a "$SCALE/progress.log"; }

# q3vl8 on 1 GPU, q3vl32 on 2 -- the round-3 shapes. Run in parallel; 129 s and 238 s.
pids=""
i=0
for TAG in q3vl8 q3vl32; do
  R=$SCALE/${TAG}_mix_rt42
  if [ -s "$R/adapter_af/adapter_model.safetensors" ]; then
    say "$TAG: adapter already present, skipping (resume, never restart)"; continue
  fi
  mkdir -p "$R"
  PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
  case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
  case $TAG in q3vl32) NG=2 ;; *) NG=1 ;; esac
  DEV=$(seq -s, $i $((i + NG - 1))); i=$((i + NG))
  say "$TAG: train af, SEED=42 STEPS=150 on GPU($DEV) ($PIN)"
  (
    export CUDA_VISIBLE_DEVICES=$DEV
    export TAG WS=mix R MVLM_DATA=$SCALE/data_mix MVLM_ROOT=$R V=$R
    export REQUEST_SLOT=1 ARMS=af STEPS=150 SEED=42 BS=8
    unset LORA_SCOPE
    $PY -I scripts/vlm/mvlm_stage1.py >>"$R/run.log" 2>&1 \
      && md5sum "$R"/adapter_af/adapter_model.safetensors > "$R/adapter_md5.txt" \
      && echo "[rt42] $TAG OK $(cat $R/adapter_md5.txt)" \
      || echo "[rt42] $TAG FAIL train (see $R/run.log)"
  ) &
  pids="$pids $!"
done
for p in $pids; do wait "$p"; done

say "md5s of the retrained adapters (these are NEW adapters, see the header):"
for TAG in q3vl8 q3vl32; do
  f=$SCALE/${TAG}_mix_rt42/adapter_md5.txt
  [ -s "$f" ] && say "  $(cat "$f")"
done
say "RETRAIN42_DONE"
