#!/usr/bin/env bash
#SBATCH --job-name=e2_4096
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_e2_%j.out
#
# next_step.md rev 6 M3 item E2: "The paper's cap. P2c ran at MAXNEW=512, while every other
# number in the paper is at 4096. Re-run alpha=0 and alpha=2 at MAXNEW=4096 on 260 gray
# rows, on both backbones, with the same adapters and the same direction. It is a new
# transcript directory, not a resume of the 512 rows. The question is whether harmful
# content comes back after token 512."
#
#   sbatch scripts/vlm/scale/e2.sbatch
#
# PACKED, one alpha per process, because steer_gen.py loops alphas SEQUENTIALLY inside one
# process. Run as one cell per backbone that would be 2 x 3.4 h = 6.8 h on q3vl8 alone and
# would miss the freeze; split across GPUs the wall-clock is one alpha, ~3.4 h, and both
# backbones finish together.
#
#   GPU 0    q3vl8  alpha=0      GPU 1    q3vl8  alpha=2      (8B: 1 GPU each)
#   GPU 2,3  q3vl32 alpha=0      GPU 4,5  q3vl32 alpha=2      (32B: 2 GPUs each)
#   GPUs 6,7 idle -- held by the 2-GPU-per-job QoS floor granularity, not needed by any cell
#
# Adapters are the RETRAINS (<tag>_mix_rt42), the same ones M1 used, and the direction is
# the same p2b fit. So E2 is comparable to M1's 512 rows and to nothing else -- not to
# round 3, not to the published P2c block.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export MAXNEW=4096
say(){ echo "[e2] $(date -u +%F' '%T) $*" | tee -a "$SCALE/progress.log"; }

pids=""
launch(){   # tag devices alpha
  local TAG=$1 DEV=$2 A=$3
  local CELL=${TAG}_steer_rt42_4096
  local R=$SCALE/$CELL
  mkdir -p "$R"
  local PIN PY
  PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
  case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
  say "$TAG alpha=$A on GPU($DEV) -> $CELL"
  (
    export CUDA_VISIBLE_DEVICES=$DEV TAG STEER_CELL=$CELL ALPHAS=$A
    export AF=$SCALE/${TAG}_mix_rt42/adapter_af
    export NO_BENIGN=1          # E2 asks for the 260 gray rows only
    $PY -I scripts/vlm/scale/steer_gen.py >>"$R/run_a$A.log" 2>&1 \
      || echo "[e2] FAIL $TAG alpha=$A"
  ) &
  pids="$pids $!"
}

launch q3vl8  0   0
launch q3vl8  1   2
launch q3vl32 2,3 0
launch q3vl32 4,5 2
for p in $pids; do wait "$p"; done
say "generation done"

# ---- judge, one pass per cell per alpha ---------------------------------------------
JPY=$SCALE/venv_4x/bin/python
for TAG in q3vl8 q3vl32; do
  R=$SCALE/${TAG}_steer_rt42_4096
  rm -f "$R/GEN_FAIL"; : > "$R/GEN_DONE"
  for A in a0 a2; do
    F="$R/res_af_$A.jsonl"
    [ -s "$F" ] || continue
    say "judge $TAG $A ($(wc -l < "$F") rows)"
    V=$R WS=mix ARMS=af SFX="_$A" \
      $JPY -I scripts/vlm/judge_stream.py >>"$R/judge_$A.log" 2>&1 \
      || say "judge $TAG $A returned non-zero"
  done
done
say "E2_JOB_DONE"
