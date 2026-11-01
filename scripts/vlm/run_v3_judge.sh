#!/usr/bin/env bash
# Judge the generated arms with Llama-Guard-3-8B (lg) then Qwen3Guard-Gen-8B (qg).
#
# Same shape as run_v3_gen.sh and for the same reason: `colab exec` buffers output until
# the cell ends, so a VM reclaimed mid-cell raises "Connection was lost." and takes the
# work with it. The judge runs detached and the driver polls `colab download`.
#
# Judge VMs are cheap to set up -- only the result jsonls go up (a few hundred KB each),
# no images and no adapters. One VM loads one guard and sweeps every arm.
#
# Env: ARMS, SFX, JUDGE (lg|qg), ROUNDS, BUDGET, BS, JUDGE_FIELD, POLL.
set -u
cd "$(dirname "$0")/../.."
SP="${SP:?scratchpad path required}"
ARMS="${ARMS:-base,af,db}"; SFX="${SFX:-}"; JUDGE="${JUDGE:-lg}"
ROUNDS="${ROUNDS:-4}"; BUDGET="${BUDGET:-1500}"; BS="${BS:-12}"; POLL="${POLL:-30}"
FIELD="${JUDGE_FIELD:-raw}"
OUTDIR="outputs/vlm2${SFX:-_sdtypo}"
mkdir -p "$OUTDIR" "$SP/log"
LOG="$SP/log/v3_${JUDGE}${SFX}.log"
say(){ echo "[$JUDGE${SFX}] $*" | tee -a "$LOG"; }

# Overridable so a second judging protocol (e.g. NEUTRAL_USER) can write to its own files
# instead of overwriting the primary verdicts.
if [ -n "${IN_SFX+x}" ] || [ -n "${OUT_SFX+x}" ]; then
  IN_SFX="${IN_SFX:-}"; OUT_SFX="${OUT_SFX:?set OUT_SFX with IN_SFX}"
else
  IN_SFX=""; OUT_SFX="_lg"
  [ "$JUDGE" = "qg" ] && { IN_SFX="_lg"; OUT_SFX="_qg"; }
fi

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

complete(){
  for a in ${ARMS//,/ }; do
    i="$OUTDIR/res_${a}${IN_SFX}.jsonl"; o="$OUTDIR/res_${a}${OUT_SFX}.jsonl"
    [ -f "$i" ] || return 1
    [ "$(rows "$o")" -ge "$(rows "$i")" ] || return 1
  done
  return 0
}

for round in $(seq 1 "$ROUNDS"); do
  if complete; then say "all arms judged"; break; fi
  S="j${JUDGE}${SFX}r${round}"
  say "=== round $round session $S ==="
  # An 8B guard in 4-bit fits a standard A100 or an L4, so don't fail the round when
  # A100 high-mem returns Service Unavailable -- walk down the hardware options.
  made=0
  for hw in "--gpu A100 --high-mem" "--gpu A100" "--gpu L4 --high-mem" "--gpu L4"; do
    if timeout 600 colab new $hw -s "$S" < /dev/null >>"$LOG" 2>&1; then
      say "  got $hw"; made=1; break
    fi
    say "  unavailable: $hw"
  done
  [ "$made" = 1 ] || { say "session create failed on every hardware option"; sleep 60; continue; }

  push(){ timeout 600 colab upload -s "$S" "$1" "$2" < /dev/null >>"$LOG" 2>&1 \
            || say "UPLOAD FAILED $1"; }
  push scripts/vlm/mkdirs_remote.py /content/mkdirs_remote.py
  timeout 200 colab exec -s "$S" --timeout 120 -f scripts/vlm/mkdirs_remote.py \
    < /dev/null >>"$LOG" 2>&1
  push .hf /content/hf_token.txt
  push scripts/vlm/judge_multi.py /content/judge_multi.py
  push scripts/vlm/launch_bg.py /content/launch_bg.py
  for a in ${ARMS//,/ }; do
    for sx in "$IN_SFX" "$OUT_SFX"; do
      f="$OUTDIR/res_${a}${sx}.jsonl"
      [ -f "$f" ] && push "$f" "/content/vlm2/res_${a}${SFX}${sx}.jsonl"
    done
  done

  cat > "$SP/stage_launch_${JUDGE}.py" <<PY
import os, subprocess, sys
os.environ.update({"BG_SCRIPT": "/content/judge_multi.py",
                   "BG_LOG": "/content/judge.log",
                   "ARMS": "${ARMS}", "SFX": "${SFX}", "JUDGE": "${JUDGE}",
                   "NEUTRAL_USER": """${NEUTRAL_USER:-}""",
                   "IN_SFX": "${IN_SFX}", "OUT_SFX": "${OUT_SFX}",
                   "JUDGE_FIELD": "${FIELD}", "BS": "${BS}",
                   "TIME_BUDGET": "${BUDGET}"})
r = subprocess.run([sys.executable, "/content/launch_bg.py"], capture_output=True, text=True)
print(r.stdout, r.stderr[-800:], flush=True)
PY
  timeout 200 colab exec -s "$S" --timeout 120 -f "$SP/stage_launch_${JUDGE}.py" \
    < /dev/null 2>&1 | tee -a "$LOG" | grep -E "LAUNCHED|Error" || true

  say "polling every ${POLL}s"
  stall=0
  while true; do
    sleep "$POLL"
    got=0
    for a in ${ARMS//,/ }; do
      timeout 120 colab download -s "$S" "/content/vlm2/res_${a}${SFX}${OUT_SFX}.jsonl" \
        "$OUTDIR/res_${a}${OUT_SFX}.jsonl" < /dev/null >/dev/null 2>&1 && got=1
      trim "$OUTDIR/res_${a}${OUT_SFX}.jsonl"
    done
    timeout 120 colab download -s "$S" /content/judge.log \
      "$SP/log/judge_remote_${JUDGE}${SFX}.log" < /dev/null >/dev/null 2>&1
    tot=0
    for a in ${ARMS//,/ }; do tot=$(( tot + $(rows "$OUTDIR/res_${a}${OUT_SFX}.jsonl") )); done
    say "  polled: $tot judged$( [ "$got" = 0 ] && echo ' (no file yet / session gone)')"
    if complete; then say "  all arms judged"; break; fi
    if [ "$got" = 0 ]; then stall=$(( stall + 1 )); else stall=0; fi
    if [ "$stall" -ge 6 ]; then say "  session unreachable; next round"; break; fi
    if grep -qE "_MULTI_(DONE|PARTIAL)" "$SP/log/judge_remote_${JUDGE}${SFX}.log" 2>/dev/null
    then say "  remote judge finished"; break; fi
  done
  timeout 120 colab stop -s "$S" < /dev/null >>"$LOG" 2>&1
done
say "FINAL:"
for a in ${ARMS//,/ }; do
  say "  $a: $(rows "$OUTDIR/res_${a}${OUT_SFX}.jsonl")/$(rows "$OUTDIR/res_${a}${IN_SFX}.jsonl")"
done
