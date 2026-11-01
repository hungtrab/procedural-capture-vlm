#!/usr/bin/env bash
# Stage 3 driver: harmful eval that survives Colab's VM reclaim.
#
# Two things this gets right that the first version did not:
#  1. `colab exec -f` runs the file on the REMOTE kernel, so locally-exported env vars
#     are NOT visible to it. Each arm therefore gets a generated wrapper with its config
#     baked in as literals.
#  2. Every colab call gets `< /dev/null`, so it cannot consume the driver's stdin.
#
# Generation is resumable and time-budgeted: each exec stops cleanly, the partial .jsonl
# is pulled to local disk, and a fresh VM resumes from it. Nothing is regenerated.
#
# Usage: EVALSET=sdtypo|sd|gray ARMS="base af db" bash scripts/vlm/run_v2_eval.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
EVALSET="${EVALSET:-sdtypo}"
ARMS="${ARMS:-base af db}"
BS="${BS:-8}"
TIME_BUDGET="${TIME_BUDGET:-900}"
NEVAL="${NEVAL:-260}"
STORE=adapters_store
TMP=/tmp/claude-1000/-home-hung7rab-lab-reflourished-emergent/b3c23f58-9f99-489c-bf6f-2c4320317f3f/scratchpad
WORK="outputs/vlm2_$EVALSET"; mkdir -p "$WORK"
case "$EVALSET" in
  sdtypo) EVALF=/content/vlm2/eval_harmful.jsonl;      SFX="" ;;
  sd)     EVALF=/content/vlm2/eval_harmful_sd.jsonl;   SFX="_sd" ;;
  gray)   EVALF=/content/vlm2/eval_harmful_gray.jsonl; SFX="_gray" ;;
  *) echo "bad EVALSET"; exit 1 ;;
esac

S="${SESSION:-}"
cx() { timeout "$1" colab exec -s "$S" -f "$2" --timeout "$3" < /dev/null 2>&1; }

new_session() {
  S="mc_ev$(date +%H%M%S)"
  echo "[eval] creating session $S"
  timeout 600 colab new --gpu A100 --high-mem -s "$S" < /dev/null >/dev/null 2>&1 || return 1
  timeout 300 colab upload -s "$S" .hf /content/hf_token.txt < /dev/null >/dev/null 2>&1
  for f in vlm_build_data_v2.py vlm_train_llama.py master_v2_train.py mkdirs_remote.py \
           build_eval_sd.py build_eval_gray.py vlm_gen_llama.py \
           lgjudge_only.py qgpass.py score_v2.py master_v2_guard.py; do
    timeout 300 colab upload -s "$S" "scripts/vlm/$f" "/content/$f" < /dev/null >/dev/null 2>&1
  done
  cx 300 scripts/vlm/mkdirs_remote.py 240 | tail -1
  # The 236MB adapters cannot be uploaded (the contents API drops the connection with an
  # SSL EOF), so a fresh VM has to rebuild the data and retrain both arms (~23 min).
  # Training is deterministic, so this reproduces the same adapters bit-for-bit.
  echo "[eval] fresh VM: rebuilding data + retraining both arms on $S"
  cx 3400 scripts/vlm/master_v2_train.py 3300 | grep -E "rows|PREFLIGHT|TRAIN_DONE|STAGE1_DONE|ABORT" | tail -12
  case "$EVALSET" in
    sd)   cx 1500 scripts/vlm/build_eval_sd.py 1400 | tail -2 ;;
    gray) cx 900  scripts/vlm/build_eval_gray.py 800 | tail -2 ;;
  esac
  for a in $ARMS; do
    if [ -s "$WORK/res_${a}${SFX}.jsonl" ]; then
      timeout 900 colab upload -s "$S" "$WORK/res_${a}${SFX}.jsonl" \
        "/content/vlm2/res_${a}${SFX}.jsonl" < /dev/null >/dev/null 2>&1 \
        && echo "  restored res_${a}${SFX}.jsonl ($(wc -l < "$WORK/res_${a}${SFX}.jsonl") rows)"
    fi
  done
  return 0
}

alive() { timeout 120 colab sessions < /dev/null 2>/dev/null | grep -q "\[$S\]"; }

for a in $ARMS; do
  case "$a" in
    base) AD=none ;;
    af)   AD=/content/adapter_af_v2 ;;
    db)   AD=/content/adapter_db_v2 ;;
  esac
  # env baked in as literals -- the remote kernel never sees our local exports
  W="$TMP/gen_${a}${SFX}.py"
  cat > "$W" <<PY
import os, sys, subprocess
os.environ["ADAPTER"]     = "$AD"
os.environ["EVALF"]       = "$EVALF"
os.environ["OUTF"]        = "/content/vlm2/res_${a}${SFX}.jsonl"
os.environ["BS"]          = "$BS"
os.environ["TIME_BUDGET"] = "$TIME_BUDGET"
print("gen arm=$a adapter=$AD", flush=True)
r = subprocess.run([sys.executable, "/content/vlm_gen_llama.py"], env=dict(os.environ))
print("child rc", r.returncode, flush=True)
PY
  for attempt in $(seq 1 10); do
    LOCAL="$WORK/res_${a}${SFX}.jsonl"
    have=$([ -s "$LOCAL" ] && wc -l < "$LOCAL" || echo 0)
    echo "[eval] arm=$a attempt=$attempt local_rows=$have"
    if ! alive; then new_session || { echo "[eval] session create failed"; exit 1; }; fi
    cx 1200 "$W" 1100 | grep -E "gen arm|resume|TIME_BUDGET|rows now|LLAMA_GEN|child rc|Error|Traceback|No such" | tail -10
    timeout 900 colab download -s "$S" "/content/vlm2/res_${a}${SFX}.jsonl" "$LOCAL" \
      < /dev/null >/dev/null 2>&1 && echo "  pulled $(wc -l < "$LOCAL") rows"
    n=$([ -s "$LOCAL" ] && wc -l < "$LOCAL" || echo 0)
    if [ "$n" -ge "$NEVAL" ]; then echo "[eval] arm=$a COMPLETE ($n rows)"; break; fi
    if [ "$n" -le "$have" ]; then
      echo "  (no progress -- stopping this arm to avoid burning units)"; break
    fi
  done
done
echo "[eval] EVAL_GEN_DRIVER_DONE"
