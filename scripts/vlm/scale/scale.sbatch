#!/usr/bin/env bash
#SBATCH --job-name=scale_bf16
#SBATCH --partition=a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --output=outputs/scale/slurm_%j.out
# The whole bf16 round in one 8-GPU allocation:
#   GPUs 0-3: the 8B cells, one per GPU (bf16 8B trains in well under 80 GB).
#   GPUs 4+5 and 6+7: the 31-32B cells, two GPUs each (64 GB of bf16 weights alone).
# When the small queue is empty, GPUs 0+1 and 2+3 join the big queue as two more pairs.
# Resubmit the same command after a timeout or a failure: everything resumes.
#   sbatch scripts/vlm/scale/scale.sbatch
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
FREE=$SCALE/claims_${SLURM_JOB_ID:-local}/free; mkdir -p "$FREE"
for g in 0 1 2 3; do
  ( GPUS=$g QUEUE=small bash scripts/vlm/scale/run_scale.sh
    touch "$FREE/$g"
    if [ $((g % 2)) = 0 ]; then
      while [ ! -e "$FREE/$((g + 1))" ]; do sleep 60; done
      GPUS=$g,$((g + 1)) QUEUE=big bash scripts/vlm/scale/run_scale.sh
    fi ) > "$SCALE/worker_small_$g.out" 2>&1 &
  sleep 30
done
for p in 4,5 6,7; do
  GPUS=$p QUEUE=big bash scripts/vlm/scale/run_scale.sh > "$SCALE/worker_big_${p/,/}.out" 2>&1 &
  sleep 30
done
wait
echo "SCALE_JOB_DONE $(ls -d $SCALE/*/DONE 2>/dev/null | wc -l)/12 cells"
