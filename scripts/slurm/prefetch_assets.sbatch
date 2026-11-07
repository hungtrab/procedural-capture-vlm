#!/bin/bash
#SBATCH --job-name=pcvlm-fetch
#SBATCH --qos=data-transfer_root
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/logs/fetch-%j.out
#SBATCH --error=/genai/fsx-llm/interns/hoangpv/pcvlm/logs/fetch-%j.out
#
# Pre-download every checkpoint the experiment needs into the shared HF cache on FSx, so no
# GPU job ever spends its allocation downloading weights. Idempotent: `hf download` is a
# no-op for a repo that is already complete, so re-running this fills only what is missing.
#
#   mkdir -p /genai/fsx-llm/interns/hoangpv/pcvlm/logs
#   sbatch scripts/slurm/prefetch_assets.sbatch
set -u

# The login shell exports http(s)_proxy=10.0.2.2:<port>, the LOGIN node's host-NAT address.
# Nothing listens on it from here, so every download would hang until it timed out.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO_DIR=/home/hoangpv/implement/procedural-capture-vlm
export HF_HOME=/genai/fsx-llm/interns/hoangpv/envs/hf_home
export HF_TOKEN="$(cat "$REPO_DIR/.hf")"
export HF_HUB_ENABLE_HF_TRANSFER=0

# NOT `hf` from PATH. The shim at ~/.local/bin/hf is bound to an interpreter that has no
# huggingface_hub installed, so it dies with ModuleNotFoundError on every repo (job 601567).
# Name the interpreter explicitly instead; HF_PY overrides it once the project env exists.
HF_PY="${HF_PY:-/genai/fsx-llm/interns/hoangpv/envs/vsr_vllm/bin/python}"
"$HF_PY" -c "import huggingface_hub" 2>/dev/null || {
  echo "[fetch] STOP: $HF_PY has no huggingface_hub; set HF_PY to one that does" >&2; exit 1; }

say(){ echo "[fetch] $(date '+%m-%d %H:%M:%S') $*"; }
say "HF_HOME=$HF_HOME  host=$(hostname)  python=$HF_PY"

# The three backbones of stages 1, 3 and 4, then the two guards. Guards last: a backbone that
# will not download is a reason to stop, and finding that out after 144GB is the slow order.
MODELS="
google/gemma-3-12b-it
Qwen/Qwen3-VL-32B-Instruct
meta-llama/Llama-3.2-11B-Vision-Instruct
meta-llama/Llama-Guard-3-8B
Qwen/Qwen3Guard-Gen-8B
Qwen/Qwen2.5-VL-3B-Instruct
google/gemma-3-4b-it
Qwen/Qwen3-VL-2B-Instruct
Qwen/Qwen3.5-0.8B
Qwen/Qwen2.5-VL-7B-Instruct
google/gemma-4-31B-it
"

rc=0
for m in $MODELS; do
  say "=== $m ==="
  t0=$SECONDS
  if "$HF_PY" -c '
import os, sys
from huggingface_hub import snapshot_download
# max_workers=8 on a 2-CPU data-transfer node is fine: this is network bound, and a single
# stream measured 147MB/s. allow_patterns excludes nothing -- the processor/tokenizer files
# are small and every one of them is needed by AutoProcessor.
p = snapshot_download(sys.argv[1], max_workers=8)
print("  ->", p)
' "$m"; then
    say "  ok in $(( SECONDS - t0 ))s"
  else
    say "  FAILED $m"; rc=1
  fi
done

say "cache size: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
say "PREFETCH_DONE rc=$rc"
exit $rc
