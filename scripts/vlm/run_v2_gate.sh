#!/usr/bin/env bash
# Recovery/continuation driver: run the benign sanity gate from LOCALLY STORED adapters.
# Use this whenever a VM died after training. It re-uploads the adapters instead of
# retraining them (19 min -> ~2 min) and rebuilds the data (~4 min), so the gate fits
# inside Colab's ~21-23 min reclaim window.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
S="${S:-mc_gate}"
STORE=adapters_store

for a in af db; do
  f="$STORE/llama_vision_${a}_v2/adapter_model.safetensors"
  [ -s "$f" ] || { echo "[gate] MISSING $f -- train first (run_v2_pipeline.sh)"; exit 1; }
done

if ! timeout 120 colab sessions 2>/dev/null | grep -q "\[$S\]"; then
  echo "[gate] creating session $S"
  timeout 600 colab new --gpu A100 --high-mem -s "$S" >/dev/null 2>&1 || exit 1
fi

push() { timeout 900 colab upload -s "$S" "$1" "$2" >/dev/null 2>&1 \
         && echo "  up   $2" || echo "  ERR up $2"; }

echo "[gate] uploading token + scripts"
push .hf /content/hf_token.txt
for f in vlm_build_data_v2.py gen_sanity_multi.py master_v2_sanity.py master_v2_eval.py \
         master_v2_guard.py vlm_gen_llama.py lgjudge_only.py qgpass.py score_v2.py \
         build_eval_sd.py build_eval_gray.py; do
  push "scripts/vlm/$f" "/content/$f"
done

echo "[gate] uploading stored adapters"
for a in af db; do
  for f in adapter_config.json adapter_model.safetensors; do
    push "$STORE/llama_vision_${a}_v2/$f" "/content/adapter_${a}_v2/$f"
  done
done

echo "[gate] === rebuild data (no training) ==="
timeout 1500 colab exec -s "$S" -f scripts/vlm/vlm_build_data_v2.py --timeout 1400 2>&1 \
  | grep -v -E '^(Resolving|README|Retrying|WARNING|Downloading|Fetching|   ━)' | tail -25

echo "[gate] === benign sanity gate ==="
timeout 2000 colab exec -s "$S" -f scripts/vlm/master_v2_sanity.py --timeout 1900 2>&1 \
  | grep -v -E '^(Resolving|README|Retrying|WARNING|Downloading|Fetching|   ━)' | tail -120
echo "[gate] GATE_DRIVER_DONE"
