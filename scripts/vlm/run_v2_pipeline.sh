#!/usr/bin/env bash
# Driver for the v2 VLM re-run. Every stage is resumable and each adapter is pulled to
# local disk as soon as it trains, because the first A100 session died after training
# both arms and the adapters were lost with it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
S="${S:-mc_v2}"
STORE=adapters_store
mkdir -p "$STORE/llama_vision_af_v2" "$STORE/llama_vision_db_v2"

ensure_session() {
  if ! timeout 120 colab sessions 2>/dev/null | grep -q "\[$S\]"; then
    echo "[driver] creating session $S"
    timeout 600 colab new --gpu A100 --high-mem -s "$S" >/dev/null 2>&1 || return 1
  fi
  return 0
}

push() {   # push <local> <remote>
  timeout 240 colab upload -s "$S" "$1" "$2" >/dev/null 2>&1 \
    && echo "  up   $2" || echo "  ERR up $2"
}

upload_all() {
  echo "[driver] uploading scripts"
  push .hf /content/hf_token.txt
  for f in vlm_build_data_v2.py vlm_train_llama.py gen_sanity_multi.py \
           master_v2_train.py master_v2_sanity.py master_v2_eval.py master_v2_guard.py \
           vlm_gen_llama.py lgjudge_only.py qgpass.py score_v2.py \
           build_eval_sd.py build_eval_gray.py; do
    push "scripts/vlm/$f" "/content/$f"
  done
}

fetch_adapters() {
  for a in af db; do
    for f in adapter_model.safetensors adapter_config.json; do
      if [ ! -s "$STORE/llama_vision_${a}_v2/$f" ]; then
        timeout 900 colab download -s "$S" "/content/adapter_${a}_v2/$f" \
          "$STORE/llama_vision_${a}_v2/$f" >/dev/null 2>&1 \
          && echo "  saved $STORE/llama_vision_${a}_v2/$f" \
          || echo "  (no $a adapter yet)"
      fi
    done
  done
}

stage() {  # stage <name> <local script> <timeout>
  echo "[driver] === $1 ==="
  timeout "$(( $3 + 120 ))" colab exec -s "$S" -f "$2" --timeout "$3" 2>&1 \
    | grep -v -E '^(Resolving|README|Retrying|WARNING|Downloading|Fetching|   ━)' \
    | tail -120
}

ensure_session || { echo "[driver] could not create session"; exit 1; }
upload_all
stage "stage1 build+train" scripts/vlm/master_v2_train.py 3400
fetch_adapters
stage "stage2 benign sanity gate" scripts/vlm/master_v2_sanity.py 2400
echo "[driver] DRIVER_DONE"
