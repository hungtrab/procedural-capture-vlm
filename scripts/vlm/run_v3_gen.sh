#!/usr/bin/env bash
# Harmful-eval generation across as many fresh Colab VMs as it takes.
#
# Per VM: create -> upload the split data/adapter parts -> reassemble -> launch generation
# DETACHED -> poll `colab download` for the per-arm result files while it runs -> stop.
#
# Why detached: `colab exec` buffers output until the cell ends, so a VM reclaimed mid-cell
# (~22 min) raises "Connection was lost." and takes every generated row with it. Polling
# downloads is the only transfer that is safe while a GPU job is running.
#
# Adapters are uploaded rather than retrained: training is deterministic and already done,
# and 32MB parts get around the SSL EOF that kills a single 236MB transfer.
#
# Env: ARMS, SFX, EVALF/EVAL_LOCAL (eval-set variant), ROUNDS, BUDGET, BS, POLL, NOIMG.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
ARMS="${ARMS:-base,af,db}"; SFX="${SFX:-}"; ROUNDS="${ROUNDS:-6}"
NOIMG="${NOIMG:-0}"
BUDGET="${BUDGET:-1500}"; BS="${BS:-8}"; POLL="${POLL:-30}"
# generation length; 320 keeps every harmful-eval run byte-identical, utility evals
# that only need a letter pass something much smaller
MAXNEW="${MAXNEW:-320}"
PREFILL="${PREFILL:-}"
EVALF="${EVALF:-/content/vlm2/eval_harmful.jsonl}"
EVAL_LOCAL="${EVAL_LOCAL:-$SP/data/eval_harmful.jsonl}"
ATTACH="${ATTACH:-}"                    # reuse an existing session for round 1
DATA_PARTS="${DATA_PARTS:-$SP/data/parts}"   # swap in a slim tar for variant eval sets
EXTRA_PARTS="${EXTRA_PARTS:-}"          # e.g. "$SP/data/vparts" for the gray/SD variants
# Which adapters to evaluate. Defaults are the seed-42 pair; a seed run points these at
# adapters_store/llama_vision_{af,db}_s<N> and at a chunks dir split from them. The remote
# directory names stay *_v2 so gen_harmful_multi.py needs no change -- what varies is the
# weights inside them, and the md5 gate below is what proves which ones arrived.
ADIR_AF="${ADIR_AF:-adapters_store/llama_vision_af_v2}"
ADIR_DB="${ADIR_DB:-adapters_store/llama_vision_db_v2}"
CHUNKS="${CHUNKS:-$SP/chunks}"
OUTDIR="outputs/vlm2${SFX:-_sdtypo}"
mkdir -p "$OUTDIR" "$SP/log"
LOG="$SP/log/v3_gen${SFX}.log"
say(){ echo "[v3${SFX}] $*" | tee -a "$LOG"; }

N_EVAL=$(wc -l < "$EVAL_LOCAL")

rows(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }

# A blind download can catch a half-written final line; drop it so the file stays valid.
trim(){ python3 - "$1" <<'PY'
import sys, json, os
p = sys.argv[1]
if not os.path.exists(p): raise SystemExit
good = []
for line in open(p, errors="replace"):
    try: json.loads(line)
    except Exception: continue
    good.append(line if line.endswith("\n") else line + "\n")
open(p, "w").writelines(good)
PY
}

complete(){
  for a in ${ARMS//,/ }; do
    [ "$(rows "$OUTDIR/res_${a}.jsonl")" -ge "$N_EVAL" ] || return 1
  done
  return 0
}

for round in $(seq 1 "$ROUNDS"); do
  if complete; then say "all arms at ${N_EVAL} rows; done"; break; fi
  before_total=0
  for a in ${ARMS//,/ }; do
    before_total=$(( before_total + $(rows "$OUTDIR/res_${a}.jsonl") ))
  done

  if [ -n "$ATTACH" ] && [ "$round" = 1 ]; then
    S="$ATTACH"; fresh=0
  else
    S="g${SFX:-s}r${round}"; fresh=1
  fi
  say "=== round $round session $S (have $before_total/$(( N_EVAL * $(echo "$ARMS" | tr ',' ' ' | wc -w) )) rows) ==="
  if [ "$fresh" = 1 ]; then
      made=0
    for hw in "--gpu A100 --high-mem" "--gpu A100"; do
      if timeout 600 colab new $hw -s "$S" < /dev/null >>"$LOG" 2>&1; then
        say "  got $hw"; made=1; break
      fi
      say "  unavailable: $hw"
    done
    [ "$made" = 1 ] || { say "session create failed"; sleep 60; continue; }
  fi

  push(){ timeout 600 colab upload -s "$S" "$1" "$2" < /dev/null >>"$LOG" 2>&1 \
            || say "UPLOAD FAILED $1"; }
  if [ "$fresh" = 1 ]; then
    push scripts/vlm/mkdirs_remote.py /content/mkdirs_remote.py
    timeout 200 colab exec -s "$S" --timeout 120 -f scripts/vlm/mkdirs_remote.py \
      < /dev/null >>"$LOG" 2>&1
    push .hf /content/hf_token.txt
    push "$ADIR_AF/adapter_config.json" /content/adapter_af_v2/adapter_config.json
    push "$ADIR_DB/adapter_config.json" /content/adapter_db_v2/adapter_config.json
  fi
  # always re-push the code, so a fix made mid-run lands on the next round
  for f in gen_harmful_multi.py prep_remote.py launch_bg.py; do
    push "scripts/vlm/$f" "/content/$f"
  done

  if [ "$fresh" = 1 ]; then
    say "uploading parts"
    t0=$(date +%s)
    : > "$SP/upfail_$$"
    for p in "$DATA_PARTS"/* "$CHUNKS"/* ${EXTRA_PARTS:+"$EXTRA_PARTS"/*}; do
      # one retry: a single transient failure here used to cost a whole VM
      ( timeout 600 colab upload -s "$S" "$p" "/content/parts/$(basename "$p")" \
          < /dev/null >/dev/null 2>&1 \
        || timeout 600 colab upload -s "$S" "$p" "/content/parts/$(basename "$p")" \
             < /dev/null >/dev/null 2>&1 \
        || echo "$(basename "$p")" >> "$SP/upfail_$$" ) &
      while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 1; done
    done
    wait
    if [ -s "$SP/upfail_$$" ]; then
      say "  PART UPLOADS FAILED: $(tr '\n' ' ' < "$SP/upfail_$$")"
    fi
    rm -f "$SP/upfail_$$"
    say "parts uploaded in $(( $(date +%s) - t0 ))s"
  fi

  # resume: hand the remote whatever is already generated so it appends
  for a in ${ARMS//,/ }; do
    f="$OUTDIR/res_${a}.jsonl"
    [ -f "$f" ] && push "$f" "/content/vlm2/res_${a}${SFX}.jsonl"
  done

  MD5_AF=$(md5sum "$ADIR_AF/adapter_model.safetensors" | cut -d" " -f1)
  MD5_DB=$(md5sum "$ADIR_DB/adapter_model.safetensors" | cut -d" " -f1)
  say "reassembling (expect af=${MD5_AF:0:12} db=${MD5_DB:0:12})"
  cat > "$SP/stage_prep.py" <<PY
import os, subprocess, sys
os.environ.update({"MD5_AF": "${MD5_AF}", "MD5_DB": "${MD5_DB}",
                   "EVAL_CHECK": "${EVALF}"})
r = subprocess.run([sys.executable, "/content/prep_remote.py"], capture_output=True,
                   text=True)
print(r.stdout[-4000:], flush=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:], flush=True); sys.exit(1)
PY
  timeout 600 colab exec -s "$S" --timeout 500 -f "$SP/stage_prep.py" \
    < /dev/null > "$SP/prep_out.txt" 2>&1
  grep -E "data:|adapter|eval_imgs|PREP|MISMATCH" "$SP/prep_out.txt" | tee -a "$LOG" || true
  if ! grep -q PREP_DONE "$SP/prep_out.txt"; then
    say "PREP FAILED -- not launching; see $SP/prep_out.txt"
    timeout 120 colab stop -s "$S" < /dev/null >>"$LOG" 2>&1
    continue
  fi

  cat > "$SP/stage_launch.py" <<PY
import os, subprocess, sys, json
os.environ.update({"BG_SCRIPT": "/content/gen_harmful_multi.py",
                   "BG_LOG": "/content/gen.log",
                   "ARMS": "${ARMS}", "SFX": "${SFX}", "BS": "${BS}",
                   "TIME_BUDGET": "${BUDGET}", "EVALF": "${EVALF}",
                   "MAXNEW": "${MAXNEW}", "PREFILL": "${PREFILL}",
                   "NOIMG": "${NOIMG}"})
r = subprocess.run([sys.executable, "/content/launch_bg.py"], capture_output=True,
                   text=True)
print(r.stdout, r.stderr[-800:], flush=True)
PY
  timeout 200 colab exec -s "$S" --timeout 120 -f "$SP/stage_launch.py" \
    < /dev/null 2>&1 | tee -a "$LOG" | grep -E "LAUNCHED|Error" || true

  # poll: download whatever exists; the VM can vanish at any moment
  say "polling every ${POLL}s"
  stall=0
  while true; do
    sleep "$POLL"
    got=0
    for a in ${ARMS//,/ }; do
      timeout 120 colab download -s "$S" "/content/vlm2/res_${a}${SFX}.jsonl" \
        "$OUTDIR/res_${a}.jsonl" < /dev/null >/dev/null 2>&1 && got=1
      trim "$OUTDIR/res_${a}.jsonl"
    done
    timeout 120 colab download -s "$S" /content/gen.log "$SP/log/gen_remote${SFX}.log" \
      < /dev/null >/dev/null 2>&1
    now_total=0
    for a in ${ARMS//,/ }; do
      now_total=$(( now_total + $(rows "$OUTDIR/res_${a}.jsonl") ))
    done
    say "  polled: total $now_total rows$( [ "$got" = 0 ] && echo ' (no file yet / session gone)')"
    if complete; then say "  all arms complete"; break; fi
    if [ "$got" = 0 ]; then stall=$(( stall + 1 )); else stall=0; fi
    if [ "$stall" -ge 6 ]; then say "  session unreachable; next round"; break; fi
    if grep -q "HARMFUL_MULTI_" "$SP/log/gen_remote${SFX}.log" 2>/dev/null; then
      say "  remote job finished"; break
    fi
  done
  timeout 120 colab stop -s "$S" < /dev/null >>"$LOG" 2>&1
done
say "FINAL:"
for a in ${ARMS//,/ }; do say "  $a: $(rows "$OUTDIR/res_${a}.jsonl")/${N_EVAL}"; done
