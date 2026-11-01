#!/usr/bin/env bash
# Train ONE arm at ONE seed per VM, then poll the adapter down.
#
# One arm per VM on purpose: build (212s) + two arms (607s + 513s) = 22.5 min, which is
# exactly the reclaim window that destroyed both adapters once already. build + one arm is
# ~14 min and leaves room to fetch the weights.
#
# The data is rebuilt on the VM rather than shipped: the build is deterministic (verified --
# it reproduces median 157 words and a 1544-char AF target exactly) and the 300 training
# images were deliberately left out of the packed tar.
#
# Env: SEED, ARM (af|db), ROUNDS, POLL.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
SEED="${SEED:?SEED required}"; ARM="${ARM:?ARM required (af|db)}"
ROUNDS="${ROUNDS:-4}"; POLL="${POLL:-30}"
STORE="adapters_store/llama_vision_${ARM}_s${SEED}"
mkdir -p "$STORE" "$SP/log"
LOG="$SP/log/seed${SEED}_${ARM}.log"
say(){ echo "[s${SEED}/${ARM}] $*" | tee -a "$LOG"; }

TRAINF=$([ "$ARM" = af ] && echo /content/vlm2/train_answerfirst.jsonl \
                         || echo /content/vlm2/train_directbenign.jsonl)

done_already(){ [ -s "$STORE/adapter_model.safetensors" ] \
  && [ "$(stat -c%s "$STORE/adapter_model.safetensors")" -ge 236000000 ]; }

for round in $(seq 1 "$ROUNDS"); do
  if done_already; then say "adapter present ($(stat -c%s "$STORE/adapter_model.safetensors") bytes)"; break; fi
  S="tr${ARM}s${SEED}r${round}"
  say "=== round $round session $S ==="
  made=0
  for hw in "--gpu A100 --high-mem" "--gpu A100"; do
    if timeout 600 colab new $hw -s "$S" < /dev/null >>"$LOG" 2>&1; then
      say "  got $hw"; made=1; break
    fi
    say "  unavailable: $hw"
  done
  [ "$made" = 1 ] || { say "no hardware"; sleep 60; continue; }

  push(){ timeout 600 colab upload -s "$S" "$1" "$2" < /dev/null >>"$LOG" 2>&1 \
            || say "UPLOAD FAILED $1"; }
  push scripts/vlm/mkdirs_remote.py /content/mkdirs_remote.py
  timeout 200 colab exec -s "$S" --timeout 120 -f scripts/vlm/mkdirs_remote.py \
    < /dev/null >>"$LOG" 2>&1
  push .hf /content/hf_token.txt
  for f in vlm_build_data_v2.py vlm_train_llama.py launch_bg.py; do
    push "scripts/vlm/$f" "/content/$f"
  done

  cat > "$SP/stage_seed.py" <<PY
import os, subprocess, sys, time
# build first (deterministic, ~212s), then train the one arm, both inside the detached job
os.environ.update({"SEED": "${SEED}", "TRAINF": "${TRAINF}",
                   "OUTDIR": "/content/adapter_${ARM}_s${SEED}"})
for name in ["vlm_build_data_v2.py", "vlm_train_llama.py"]:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", "/content/" + name], env=dict(os.environ),
                       capture_output=True, text=True)
    print(f">>>> {name} rc={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2500:], flush=True); sys.exit(1)
print("SEED_STAGE_DONE", flush=True)
PY
  push "$SP/stage_seed.py" /content/stage_seed.py
  cat > "$SP/launch_seed.py" <<PY
import os, subprocess, sys
os.environ.update({"BG_SCRIPT": "/content/stage_seed.py", "BG_LOG": "/content/seed.log"})
r = subprocess.run([sys.executable, "/content/launch_bg.py"], capture_output=True, text=True)
print(r.stdout, r.stderr[-800:], flush=True)
PY
  timeout 200 colab exec -s "$S" --timeout 120 -f "$SP/launch_seed.py" \
    < /dev/null 2>&1 | tee -a "$LOG" | grep -E "LAUNCHED|Error" || true

  say "polling every ${POLL}s for the adapter"
  stall=0
  while true; do
    sleep "$POLL"
    timeout 300 colab download -s "$S" "/content/adapter_${ARM}_s${SEED}/adapter_model.safetensors" \
      "$STORE/adapter_model.safetensors" < /dev/null >/dev/null 2>&1 && got=1 || got=0
    timeout 120 colab download -s "$S" "/content/adapter_${ARM}_s${SEED}/adapter_config.json" \
      "$STORE/adapter_config.json" < /dev/null >/dev/null 2>&1
    alive=0
    timeout 120 colab download -s "$S" /content/seed.log "$SP/log/seed_remote_${SEED}_${ARM}.log" \
      < /dev/null >/dev/null 2>&1 && alive=1
    if done_already; then say "  adapter fetched"; break; fi
    tail_line=$(grep -oE "step [0-9]+/[0-9]+ +loss=[0-9.]+" \
                 "$SP/log/seed_remote_${SEED}_${ARM}.log" 2>/dev/null | tail -1)
    say "  waiting${tail_line:+ — $tail_line}$( [ "$alive" = 0 ] && echo ' (session gone)')"
    if [ "$alive" = 0 ]; then stall=$(( stall + 1 )); else stall=0; fi
    [ "$stall" -ge 5 ] && { say "  session unreachable; next round"; break; }
  done
  timeout 120 colab stop -s "$S" < /dev/null >>"$LOG" 2>&1
done
if done_already; then
  say "FINAL ok: $STORE ($(md5sum "$STORE/adapter_model.safetensors" | cut -c1-12))"
else
  say "FINAL FAILED: no adapter"
fi
