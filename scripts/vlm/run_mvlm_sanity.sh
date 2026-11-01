#!/usr/bin/env bash
# The benign sanity gate for every backbone in the generality run.
#
# One VM per model: upload the trained adapters, rebuild the held-out benign items, and run
# base/af/db over both the worksheet and the plain prompt. Short enough to finish inside one
# VM, and resumable from the row count if it does not.
#
# Env: TAGS, ARMS, ROUNDS, BS, BUDGET, SUF.
set -u
cd "$(dirname "$0")/../.."
WS="${WS:-count}"
SP="${SP:?scratchpad path required}"
TAGS="${TAGS:-q3,g4,q7,g12}"
ARMS="${ARMS:-base,af,db}"
ROUNDS="${ROUNDS:-4}"; BS="${BS:-4}"; BUDGET="${BUDGET:-1100}"
SUF="${SUF:-}"
mkdir -p "$SP/log"
LOG="$SP/log/mvlm_sanity.log"
say(){ echo "[san] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
rows(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }

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

newvm(){
  local made=0
  for hw in "--gpu A100 --high-mem" "--gpu A100"; do
    if timeout 600 colab new $hw -s "$1" < /dev/null >>"$LOG" 2>&1; then
      say "  got $hw"; made=1; break
    fi
    say "  unavailable: $hw"
  done
  [ "$made" = 1 ]
}

push(){ timeout 600 colab upload -s "$S" "$1" "$2" < /dev/null >>"$LOG" 2>&1 \
          || say "UPLOAD FAILED $1"; }

N_HO=40
for tag in ${TAGS//,/ }; do
  ADIR="adapters_store/mvlm_${tag}${SUF}"
  OUTDIR="outputs/vlm2_sanity_${tag}${SUF}"
  mkdir -p "$OUTDIR"
  if [ ! -s "$ADIR/adapter_af/adapter_model.safetensors" ]; then
    say "$tag: no adapters at $ADIR, skipping"; continue
  fi
  MD5_AF=$(md5sum "$ADIR/adapter_af/adapter_model.safetensors" | cut -d" " -f1)
  MD5_DB=$(md5sum "$ADIR/adapter_db/adapter_model.safetensors" | cut -d" " -f1)
  say "########## $tag sanity (af=${MD5_AF:0:12} db=${MD5_DB:0:12}) ##########"

  complete(){ for a in ${ARMS//,/ }; do for w in ws plain; do
      [ "$(rows "$OUTDIR/san_${a}_${w}.jsonl")" -ge "$N_HO" ] || return 1; done; done
    return 0; }

  crashes=0
  for round in $(seq 1 "$ROUNDS"); do
    if complete; then say "$tag: sanity complete"; break; fi
    S="sn${tag}r${round}"
    say "=== $tag sanity round $round ==="
    newvm "$S" || { sleep 60; continue; }
    push .hf /content/hf_token.txt
    for f in mvlm_arch.py mvlm_data.py mvlm_items.py mvlm_sanity.py launch_bg.py; do
      push "scripts/vlm/$f" "/content/$f"
    done
    cat > "$SP/stage_sandirs.py" <<PY
import os
for d in ("/content/mvlm", "/content/mvlm/data", "/content/mvlm/data/train_imgs",
          "/content/mvlm/data/holdout_imgs", "/content/mvlm/$tag",
          "/content/mvlm/$tag/adapter_af", "/content/mvlm/$tag/adapter_db",
          "/content/mvlm/$tag/aparts_up"):
    os.makedirs(d, exist_ok=True)
print("DIRS_OK", flush=True)
PY
    timeout 200 colab exec -s "$S" --timeout 120 -f "$SP/stage_sandirs.py" \
      < /dev/null 2>&1 | grep -E "DIRS_OK|Error" | tee -a "$LOG" || true
    : > "$SP/upfail_san"
    for a in af db; do
      push "$ADIR/adapter_$a/adapter_config.json" \
           "/content/mvlm/$tag/adapter_$a/adapter_config.json"
      rm -f "$SP"/sup_${a}_*
      split -b 30m -d -a 2 "$ADIR/adapter_$a/adapter_model.safetensors" "$SP/sup_${a}_"
      for p in "$SP"/sup_${a}_*; do
        ( timeout 600 colab upload -s "$S" "$p" \
            "/content/mvlm/$tag/aparts_up/$(basename "$p")" < /dev/null >/dev/null 2>&1 \
          || timeout 600 colab upload -s "$S" "$p" \
               "/content/mvlm/$tag/aparts_up/$(basename "$p")" < /dev/null >/dev/null 2>&1 \
          || echo "$(basename "$p")" >> "$SP/upfail_san" ) &
        while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 1; done
      done
      wait
      rm -f "$SP"/sup_${a}_*
    done
    [ -s "$SP/upfail_san" ] && say "  PART UPLOADS FAILED: $(tr '\n' ' ' < "$SP/upfail_san")"
    for a in ${ARMS//,/ }; do for w in ws plain; do
      f="$OUTDIR/san_${a}_${w}.jsonl"
      [ -f "$f" ] && push "$f" "/content/mvlm/$tag/san_${a}_${w}.jsonl"
    done; done
    cat > "$SP/stage_sangen.py" <<PY
import os, glob, hashlib, subprocess, sys
root = "/content/mvlm/${tag}"
for a in ("af", "db"):
    parts = sorted(glob.glob(root + "/aparts_up/sup_%s_*" % a))
    if parts:
        with open(root + "/adapter_%s/adapter_model.safetensors" % a, "wb") as fh:
            for p in parts:
                fh.write(open(p, "rb").read())
exp = {"af": "${MD5_AF}", "db": "${MD5_DB}"}
for a in ("af", "db"):
    f = root + "/adapter_%s/adapter_model.safetensors" % a
    h = hashlib.md5(open(f, "rb").read()).hexdigest()
    print(a, "md5", h, os.path.getsize(f), flush=True)
    if h != exp[a]:
        print("MISMATCH", a, flush=True); sys.exit(1)
print("ADAPTERS_OK", flush=True)
os.environ.update({"BG_SCRIPT": "/content/mvlm_sanity.py", "BG_LOG": "/content/san.log",
                   "TAG": "${tag}", "ARMS": "${ARMS}", "BS": "${BS}",
                   "TIME_BUDGET": "${BUDGET}", "WS": "${WS}"})
r = subprocess.run([sys.executable, "/content/launch_bg.py"], capture_output=True, text=True)
print(r.stdout, r.stderr[-800:], flush=True)
PY
    timeout 600 colab exec -s "$S" --timeout 500 -f "$SP/stage_sangen.py" \
      < /dev/null > "$SP/sangen_exec.txt" 2>&1
    grep -E "ADAPTERS_OK|MISMATCH|LAUNCHED|Error" "$SP/sangen_exec.txt" | tee -a "$LOG" || true
    if ! grep -q LAUNCHED "$SP/sangen_exec.txt"; then
      say "  launch failed; next round"
      timeout 120 colab stop -s "$S" < /dev/null >>"$LOG" 2>&1
      continue
    fi
    rm -f "$SP/log/san_${tag}.log"
    say "  polling every 30s"
    stall=0
    while true; do
      sleep 30
      got=0
      for a in ${ARMS//,/ }; do for w in ws plain; do
        timeout 120 colab download -s "$S" "/content/mvlm/$tag/san_${a}_${w}.jsonl" \
          "$OUTDIR/san_${a}_${w}.jsonl" < /dev/null >/dev/null 2>&1 && got=1
        trim "$OUTDIR/san_${a}_${w}.jsonl"
      done; done
      timeout 120 colab download -s "$S" /content/san.log "$SP/log/san_${tag}.log" \
        < /dev/null >/dev/null 2>&1
      if grep -q "^Traceback" "$SP/log/san_${tag}.log" 2>/dev/null; then
        say "  remote job crashed:"
        grep -E "Error|error:" "$SP/log/san_${tag}.log" | tail -3 | tee -a "$LOG"
        crashes=$(( crashes + 1 )); break
      fi
      tot=0
      for a in ${ARMS//,/ }; do for w in ws plain; do
        tot=$(( tot + $(rows "$OUTDIR/san_${a}_${w}.jsonl") )); done; done
      say "  polled: total $tot/$(( N_HO * 6 )) rows$( [ "$got" = 0 ] && echo ' (nothing yet)')"
      if complete; then say "  sanity complete"; break; fi
      if [ "$got" = 0 ]; then stall=$(( stall + 1 )); else stall=0; fi
      [ "$stall" -ge 8 ] && { say "  session unreachable; next round"; break; }
      if grep -q "MVLM_SANITY_" "$SP/log/san_${tag}.log" 2>/dev/null; then
        say "  remote job finished"; break
      fi
    done
    timeout 120 colab stop -s "$S" < /dev/null >>"$LOG" 2>&1
    [ "$crashes" -ge 2 ] && { say "$tag: crashed twice, giving up"; break; }
  done
  say "$tag sanity FINAL:"
  for a in ${ARMS//,/ }; do for w in ws plain; do
    say "  $a/$w: $(rows "$OUTDIR/san_${a}_${w}.jsonl")/$N_HO"; done; done
done
say "RUN_MVLM_SANITY_DONE"
