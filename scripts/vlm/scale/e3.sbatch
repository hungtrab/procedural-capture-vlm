#!/usr/bin/env bash
#SBATCH --job-name=e3_cost
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_e3_%j.out
#
# next_step.md rev 7 item 5 (archive M3, item E3): what steering COSTS on benign work.
# "The 1000 utility rows, strict and tolerant scorers, at alpha=0 and alpha=2 on both
#  backbones. Also run the 260 photo rows at alpha=0 and alpha=2, because the paper
#  reports photo beside gray."
#
#   sbatch --export=ALL,WITH_X2P=1 scripts/vlm/scale/e3.sbatch
#
# Same retrained seed-42 adapters and the same direction M1 used, so E3 is comparable to
# M1's gray rows and to nothing else -- not to round 3, not to the published P2c block.
#
# PACKED, one (backbone, alpha) per process, because steer_gen.py loops alphas
# sequentially inside one process. Each process does util then photo, selected with
# EVAL_KIND and kept apart with SFX so neither can land in the other's transcript or in
# M1's gray files:
#
#   GPU 0    q3vl8  alpha=0        GPU 1    q3vl8  alpha=2
#   GPU 2,3  q3vl32 alpha=0        GPU 4,5  q3vl32 alpha=2
#
# NO_BENIGN=1 throughout: the 40 held-out benign gate rows belong to M1's cells and were
# generated there at every alpha. Regenerating them here would duplicate rows into a second
# directory and invite someone to pool two copies of the same measurement.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export MAXNEW=${MAXNEW:-4096}
say(){ echo "[e3] $(date -u +%F' '%T) $*" | tee -a "$SCALE/progress.log"; }

pids=""
launch(){   # tag devices alpha
  local TAG=$1 DEV=$2 A=$3
  local CELL=${TAG}_steer_rt42_cost
  local R=$SCALE/$CELL
  mkdir -p "$R"
  local PIN PY
  PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
  case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
  say "$TAG alpha=$A on GPU($DEV) -> $CELL"
  (
    export CUDA_VISIBLE_DEVICES=$DEV TAG STEER_CELL=$CELL ALPHAS=$A NO_BENIGN=1
    export AF=$SCALE/${TAG}_mix_rt42/adapter_af
    for KIND in util photo; do
      EVALF=data/scale/eval_scale_mix.jsonl EVAL_KIND=$KIND SFX=_$KIND \
        $PY -I scripts/vlm/scale/steer_gen.py >>"$R/run_a${A}_$KIND.log" 2>&1 \
        || echo "[e3] FAIL $TAG alpha=$A $KIND"
    done
  ) &
  pids="$pids $!"
}

launch q3vl8  0   0
launch q3vl8  1   2
launch q3vl32 2,3 0
launch q3vl32 4,5 2
for p in $pids; do wait "$p"; done
say "generation done"

# Judge the PHOTO rows only. The 1000 utility rows are scored by answer match, never by a
# judge (summarize_scale.py), and judge_gpt.py skips category util_* by design.
JPY=$SCALE/venv_4x/bin/python
for TAG in q3vl8 q3vl32; do
  R=$SCALE/${TAG}_steer_rt42_cost
  rm -f "$R/GEN_FAIL"; : > "$R/GEN_DONE"
  for A in a0 a2; do
    F="$R/res_af_${A}_photo.jsonl"
    [ -s "$F" ] || continue
    say "judge $TAG ${A}_photo ($(wc -l < "$F") rows)"
    V=$R WS=mix ARMS=af SFX="_${A}_photo" \
      $JPY -I scripts/vlm/judge_stream.py >>"$R/judge_${A}_photo.log" 2>&1 \
      || say "judge $TAG ${A}_photo non-zero"
  done
done
say "E3_JOB_DONE"
