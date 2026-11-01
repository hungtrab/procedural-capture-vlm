#!/usr/bin/env bash
# Gray-eval one training seed end to end: generate af+db, then judge with LG and QG.
#
# Only af and db are generated. The base arm is the untouched Llama-Vision checkpoint, so
# it does not depend on the training seed -- regenerating it per seed would spend a third
# of the compute reproducing a constant. The seed-42 base rows in outputs/vlm2_gray/ serve
# every seed, and AF-DB (the paper's quantity) needs neither.
#
# The gray eval ships as a 160KB slim tar instead of the 294MB full one: it contains only
# eval_harmful_gray.jsonl and gray.png, which is everything this eval set reads. Round 1 of
# the first gray attempt recovered 0 rows because each VM spent its life uploading 1063MB.
#
# Env: SEED (required), ROUNDS.
set -eu
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
SEED="${SEED:?SEED required}"
ROUNDS="${ROUNDS:-4}"
SFX="_gray_s${SEED}"

AF="adapters_store/llama_vision_af_s${SEED}"
DB="adapters_store/llama_vision_db_s${SEED}"
for d in "$AF" "$DB"; do
  [ -s "$d/adapter_model.safetensors" ] || { echo "missing adapter: $d"; exit 1; }
done

# Chunks are split fresh each time so they can never be stale relative to the adapter the
# md5 gate in run_v3_gen.sh will check them against.
CH="$SP/chunks_s${SEED}"
rm -rf "$CH"; mkdir -p "$CH"
split -b 33554432 -d "$AF/adapter_model.safetensors" "$CH/af_"
split -b 33554432 -d "$DB/adapter_model.safetensors" "$CH/db_"
echo "[seed$SEED] chunks: $(ls -1 "$CH" | wc -l) parts"

export SP SFX ROUNDS
export ARMS="af,db"
export ADIR_AF="$AF" ADIR_DB="$DB" CHUNKS="$CH"
export DATA_PARTS="$SP/data/grayparts" EXTRA_PARTS=""
export EVALF="/content/vlm2/eval_harmful_gray.jsonl"
export EVAL_LOCAL="$SP/data/eval_harmful_gray.jsonl"
export BS=8

echo "[seed$SEED] === generation ==="
bash scripts/vlm/run_v3_gen.sh

OUT="outputs/vlm2${SFX}"
for a in af db; do
  n=$( [ -f "$OUT/res_${a}.jsonl" ] && wc -l < "$OUT/res_${a}.jsonl" || echo 0 )
  [ "$n" -ge 260 ] || { echo "[seed$SEED] generation incomplete: $a=$n/260; not judging"; exit 1; }
done

for J in lg qg; do
  echo "[seed$SEED] === judge $J ==="
  JUDGE="$J" SP="$SP" SFX="$SFX" ARMS="af,db" ROUNDS="$ROUNDS" BS=12 \
    bash scripts/vlm/run_v3_judge.sh
done
echo "[seed$SEED] SEED_EVAL_DONE"
