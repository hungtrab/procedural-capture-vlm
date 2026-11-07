#!/usr/bin/env bash
#SBATCH --job-name=scale_bf16
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=24:00:00
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

# The login shell exports http(s)_proxy=10.0.2.2:<port>, the LOGIN node's host-NAT address.
# Nothing answers on it from a compute node, so every outbound call times out instead of
# failing fast. Job 616212 died this way: nine of twelve cells printed "FAIL train" within
# three minutes, all of them a ProxyError reaching huggingface.co for processor_config.json
# -- the weights were already cached, but the hub still checks for updates, and that check
# went through a proxy that was not there.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# The shared cache every backbone and both guards were prefetched into. Without it each cell
# re-downloads into a per-job default, which is both slow and a second copy of 299 GB.
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
# AGENTS.md rule 8: .hf is a credential, read into the environment and never echoed.
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
# The AI Gateway, which judge_gpt.py uses to reach gpt-5.5 WITHOUT an API key: the request
# carries the caller's own identity through the x2pagentd forward proxy (see <repo>/api.py).
# Two things it needs on a compute node, both verified here rather than assumed:
#
#   WITH_X2P=1 on the job, which is what makes Slurm start an agent on this node and publish
#   its kernel-picked port in X2P_PORT. Submit with:
#       sbatch --export=ALL,WITH_X2P=1 scripts/vlm/scale/scale.sbatch
#   Without it X2P_PORT is unset and every GPT call stalls until its timeout.
#
#   X2P_PROXY_URL unset. The login shell exports http://10.0.2.2:10054, the LOGIN node's
#   host-NAT address, and api.py prefers it over X2P_PORT -- so left set, the calls hang
#   rather than fail, which looks like a slow judge and not a misconfiguration. Exactly the
#   trap the http_proxy unset above exists for. judge_gpt.py also drops it at import, since
#   api.py reads the proxy once and a later shell unset would be too late; this line makes
#   the job's own environment honest as well.
if [ -n "${X2P_PORT:-}" ]; then unset X2P_PROXY_URL; fi
[ -z "${X2P_PORT:-}" ] && echo "[scale] NOTE: X2P_PORT unset -- resubmit with" \
  "--export=ALL,WITH_X2P=1 or the GPT third-judge pass will stall (guards are unaffected)"

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
