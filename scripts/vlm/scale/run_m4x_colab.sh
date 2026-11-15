#!/usr/bin/env bash
# M4 extended to the two additional backbones (lv11, llnm7), seed 42, Colab A100:
# HarmBench (200, in the worksheet) + the 260 AdvBench requests bare, base/Prose/Proc,
# MAXNEW 1280 as for every other row of these backbones. Gen then LG+QG judge.
# Adapters: the seed-42 harm-cell adapters (gate passed: lv11 40/40, llnm7 38/40).
#   TAG=llnm7 bash run_m4x.sh
set -u
cd "$(dirname "$0")"
VD=$PWD
TAG=${TAG:?lv11 or llnm7}
SP=/tmp/claude-1000/-home-hung7rab-lab-reflourished-emergent/b3c23f58-9f99-489c-bf6f-2c4320317f3f/scratchpad
NB=$SP/nb
HFF=/home/hung7rab/lab_reflourished/emergent/.hf
GPU=${GPU:---gpu A100 --high-mem}
R=$VD/runs/$TAG; mkdir -p "$R/log"
LOG=$R/log/driver.log
EVAL=$VD/eval/eval_m4x.jsonl; N=$(wc -l < "$EVAL")
CELLS="m4x"
BS=8; MAXNEW=1280
COMMON="TAG=$TAG WS=mix PRECISION=bf16 N_TRAIN=300 N_HOLDOUT=40 MVLM_DATA=/content/mvlm/data BS=$BS MAXNEW=$MAXNEW REQUEST_SLOT=1 TIME_BUDGET=0 MVLM_PIP=0"
for a in af db; do mkdir -p "$R/m4x/adapter_$a"
  [ -s "$R/m4x/adapter_$a/adapter_model.safetensors" ] || cp "$SP/util/adapters/${TAG}_s42/adapter_$a/"* "$R/m4x/adapter_$a/"; done
say(){ echo "[$(date -u +%F' '%T)] $TAG $*" | tee -a "$LOG"; }
rows(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }
trim(){ python3 -I - "$1" <<'PY'
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
VMS=0
newvm(){ local t
  for t in 1 2 3 4 5 6; do
    if timeout 600 colab new $GPU -s "$1" < /dev/null > "$R/log/new.txt" 2>&1; then
      VMS=$((VMS+1)); say "VM $1 up ($GPU), VMs so far $VMS"
      timeout 60 colab url -s "$1" < /dev/null 2>/dev/null | grep -oE 'https://[^ ]+' | head -1 > "$R/URL"; return 0; fi
    cat "$R/log/new.txt" >> "$LOG"
    if grep -qiE "authoriz|oauth|login" "$R/log/new.txt"; then say "FATAL: login expired"; touch "$R/FATAL"; exit 3; fi
    if grep -qiE "abus|blocked" "$R/log/new.txt"; then say "FATAL: account blocked"; touch "$R/FATAL"; exit 3; fi
    say "no VM (try $t/6): $(grep -oiE 'Service Unavailable|TooManyAssignments|[A-Za-z]+Error[^ ]*' "$R/log/new.txt" | head -1); waiting 300s"
    sleep 300
  done
  say "FATAL: no VM after 6 tries"; touch "$R/FATAL"; exit 3; }
stopvm(){ timeout 120 colab stop -s "$1" < /dev/null >> "$LOG" 2>&1; sleep 45; }
push(){ timeout 600 colab upload -s "$S" "$1" "$2" < /dev/null >> "$LOG" 2>&1 \
        || timeout 600 colab upload -s "$S" "$1" "$2" < /dev/null >> "$LOG" 2>&1 \
        || { say "UPLOAD FAILED $1"; return 1; }; }
pull(){ timeout 180 colab download -s "$S" "$1" "$2" < /dev/null > /dev/null 2>&1; }
rexec(){ timeout 400 colab exec -s "$S" --timeout 300 -f "$1" < /dev/null > "$2" 2>&1; }
ppush(){ local p; rm -f "$R/upfail"
  for p in "$1"*; do
    ( push "$p" "$2/$(basename "$p")" > /dev/null || echo x >> "$R/upfail" ) &
    while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 1; done
  done; wait
  [ -s "$R/upfail" ] && { rm -f "$R/upfail"; return 1; }; return 0; }
have_ad(){ [ -s "$R/$1/adapter_$2/adapter_model.safetensors" ]; }
md5env(){ local e="" c a; for c in $CELLS; do for a in af db; do have_ad $c $a && e="$e MD5_${c}_${a}=$(md5sum < "$R/$c/adapter_$a/adapter_model.safetensors" | cut -c1-32)"; done; done; echo "$e"; }
arms_of(){ echo base,af,db; }

launch(){ python3 -I - "$@" > "$R/log/st_launch.py" <<'PY'
import sys, json
kv = dict(a.split("=", 1) for a in sys.argv[1:])
env = {"BG_SCRIPT": "/content/wrap.py", "BG_LOG": "/content/vg.log", "WRAP_TARGET": "/content/m4x_seq.py"}
env.update(kv)
dirs = ["/content/mvlm/data"]
for c in ("m4x",):
    for s in ("adapter_af", "adapter_db", "aparts_up", "aparts"):
        dirs.append("/content/mvlm/%s_%s/%s" % (kv["TAG"], c, s))
print("import os, subprocess, sys")
print("for d in %r: os.makedirs(d, exist_ok=True)" % dirs)
print("os.environ.update(%s)" % json.dumps(env))
print("r = subprocess.run([sys.executable, '/content/launch_bg.py'], capture_output=True, text=True)")
print("print(r.stdout, r.stderr[-800:], flush=True)")
PY
  local t; for t in 1 2 3; do
    rexec "$R/log/st_launch.py" "$R/log/launch.txt"
    grep -q LAUNCHED "$R/log/launch.txt" && return 0
    say "launch try $t failed: $(tail -c 300 "$R/log/launch.txt")"; sleep 20
  done; return 1; }

setup(){ # $1 = phase; pushes code, data, adapters, and existing outputs
  { push "$HFF" /content/hf_token.txt && push "$NB/wrap.py" /content/wrap.py \
    && push "$NB/scripts/vlm/launch_bg.py" /content/launch_bg.py && push "$VD/build/m4x_seq.py" /content/m4x_seq.py; } || return 1
  local f; for f in mvlm_arch.py mvlm_data.py mvlm_items.py mvlm_gen.py judge_stream.py judge_gpt.py judge_multi.py audit_nonanswer.py; do
    [ -f "$NB/scripts/vlm/$f" ] && { push "$NB/scripts/vlm/$f" "/content/$f" || return 1; }; done
  push "$EVAL" /content/mvlm/eval_m4x.jsonl || return 1
  if [ "$1" != train ]; then
    local c a
    for c in $CELLS; do for a in af db; do have_ad $c $a || continue
      push "$R/$c/adapter_$a/adapter_config.json" "/content/mvlm/${TAG}_$c/adapter_$a/adapter_config.json" || return 1
      rm -f "$R"/up_*; split -b 30m -d -a 2 "$R/$c/adapter_$a/adapter_model.safetensors" "$R/up_${a}_"
      ppush "$R/up_${a}_" "/content/mvlm/${TAG}_$c/aparts_up" || return 1; rm -f "$R"/up_*
    done; done
  fi
  [ "$1" = probe ] && [ -f "$R/probe_ctl.jsonl" ] && { trim "$R/probe_ctl.jsonl"; push "$R/probe_ctl.jsonl" /content/mvlm/probe/probe_ctl.jsonl || return 1; }
  local pat; case $1 in gate) pat="san_*.jsonl" ;; gen) pat="res_base.jsonl res_af.jsonl res_db.jsonl" ;; judge) pat="res_*.jsonl" ;; *) pat="" ;; esac
  for c in $CELLS; do for f in $pat; do for g in "$R/$c/"$f; do [ -f "$g" ] || continue
    trim "$g"; push "$g" "/content/mvlm/${TAG}_$c/$(basename "$g")" || return 1; done; done; done
  return 0; }

# ---- what "complete" means per phase ----
train_done(){ have_ad vgc af; }
gate_done(){ local c a w; for c in $CELLS; do for a in $(arms_of $c | tr , ' '); do for w in ws plain; do
  [ "$(rows "$R/$c/san_${a}_$w.jsonl")" -ge 40 ] || return 1; done; done; done; return 0; }
gen_done(){ local c a; for c in $CELLS; do for a in $(arms_of $c | tr , ' '); do
  [ "$(rows "$R/$c/res_$a.jsonl")" -ge "$N" ] || return 1; done; done; return 0; }
probe_done(){ grep -q PROBE_DONE "$R/log/probe_vgc.log" 2>/dev/null; }
judge_done(){ local c a; for c in $CELLS; do for a in $(arms_of $c | tr , ' '); do
  [ "$(rows "$R/$c/res_${a}_qg.jsonl")" -ge "$N" ] || return 1; done; done; return 0; }

pull_train(){ local job c arms a want i got
  for job in vgw:af vgc:af; do c=${job%%:*}; arms=$(echo $job | cut -d: -f2)
    pull "/content/train_$c.log" "$R/log/train_$c.log"
    for a in ${arms//,/ }; do have_ad $c $a && continue
      want=$(grep -oE "adapter_$a: [0-9]+ parts, md5=[0-9a-f]+" "$R/log/train_$c.log" 2>/dev/null | tail -1 | sed 's/.*md5=//')
      [ -n "$want" ] || continue
      mkdir -p "$R/$c/adapter_$a"
      pull "/content/mvlm/${TAG}_$c/aparts/adapter_$a.config.json" "$R/$c/adapter_$a/adapter_config.json"
      : > "$R/ap.bin"; i=0
      while [ $i -lt 60 ] && rm -f "$R/ap.part" && pull "/content/mvlm/${TAG}_$c/aparts/adapter_$a.p$(printf %02d $i)" "$R/ap.part" && [ -s "$R/ap.part" ]; do
        cat "$R/ap.part" >> "$R/ap.bin"; i=$((i+1)); done
      got=$(md5sum < "$R/ap.bin" | cut -d' ' -f1)
      if [ "$got" = "$want" ]; then mv "$R/ap.bin" "$R/$c/adapter_$a/adapter_model.safetensors"; say "adapter $c/$a OK md5 $want ($i parts)"
      else say "adapter $c/$a transfer bad (want $want got $got)"; rm -f "$R/ap.bin"; fi
    done; done; }
pull_files(){ # $1 = suffix list
  local c a s f; for c in $CELLS; do mkdir -p "$R/$c"; for a in $(arms_of $c | tr , ' '); do for s in $1; do
    f=$R/$c/$(printf "$s" "$a")
    pull "/content/mvlm/${TAG}_$c/$(basename "$f")" "$f.new" && { trim "$f.new"
      [ "$(rows "$f.new")" -ge "$(rows "$f")" ] && mv "$f.new" "$f"; }; rm -f "$f.new"
  done; done; done; }

run_phase(){ # $1 phase, $2 jobs, $3 max VMs
  local ph=$1 jobs=$2 maxv=$3 n=0 crashes=0 RL stall
  while ! ${ph}_done; do
    n=$((n+1)); [ $n -gt $maxv ] && { say "FATAL: $ph incomplete after $maxv VMs"; touch "$R/FATAL"; exit 7; }
    S="m4${TAG:0:2}${ph:0:1}$n$(date +%H%M)"; say "=== $ph round $n, VM $S ==="
    newvm "$S"
    [ "$ph" = train ] && jobs=$(trainjobs)
    if ! { push "$HFF" /content/hf_token.txt && push "$NB/wrap.py" /content/wrap.py && push "$NB/scripts/vlm/launch_bg.py" /content/launch_bg.py; }; then say "base push failed"; stopvm "$S"; continue; fi
    if ! launch $COMMON PHASE=$ph $(md5env); then say "launch failed"; stopvm "$S"; continue; fi
    if ! setup "$ph"; then say "setup failed"; stopvm "$S"; continue; fi
    push "$R/GO" /content/GO || { stopvm "$S"; continue; }
    say "job running"
    RL=$R/log/${ph}_$n.log; rm -f "$RL"; stall=0
    while true; do
      sleep 45
      pull /content/vg.log "$RL" && stall=0 || stall=$((stall+1))
      case $ph in
        train) pull_train ;;
        gate)  pull_files "san_%s_ws.jsonl san_%s_plain.jsonl"; for c in $CELLS; do pull "/content/gate_$c.log" "$R/log/gate_$c.log"; done ;;
        gen)   pull_files "res_%s.jsonl"; pull /content/gen_m4x.log "$R/log/gen_m4x.log" ;;
        judge) pull_files "res_%s_lg.jsonl res_%s_qg.jsonl"; pull /content/judge_m4x.log "$R/log/judge_m4x.log" ;;
        probe) pull /content/probe_vgc.log "$R/log/probe_vgc.log"; pull /content/mvlm/probe/probe_ctl.jsonl "$R/probe_ctl.jsonl.new" && { trim "$R/probe_ctl.jsonl.new"; [ "$(rows "$R/probe_ctl.jsonl.new")" -ge "$(rows "$R/probe_ctl.jsonl")" ] && mv "$R/probe_ctl.jsonl.new" "$R/probe_ctl.jsonl"; }; rm -f "$R/probe_ctl.jsonl.new" ;;
      esac
      say "  $ph: $(for c in $CELLS; do for a in $(arms_of $c | tr , ' '); do case $ph in
          train) have_ad $c $a && printf '%s/%s=ok ' $c $a || printf '%s/%s=- ' $c $a ;;
          gate) printf '%s/%s=%s ' $c $a $(( $(rows $R/$c/san_${a}_ws.jsonl) + $(rows $R/$c/san_${a}_plain.jsonl) )) ;;
          gen) printf '%s/%s=%s ' $c $a $(rows $R/$c/res_$a.jsonl) ;;
          judge) printf '%s/%s=%s ' $c $a $(rows $R/$c/res_${a}_qg.jsonl) ;;
          probe) printf '%s/%s ' $c $a ;; esac; done; done) $( [ $ph = probe ] && echo probe rows $(rows $R/probe_ctl.jsonl))"
      if grep -q "^Traceback" "$RL" 2>/dev/null; then say "crashed: $(grep -E 'Traceback|Error' "$RL" | tail -2 | tr '\n' ' ')"; crashes=$((crashes+1)); break; fi
      ${ph}_done && break
      grep -q "VG_SEQ_DONE" "$RL" 2>/dev/null && break
      [ $stall -ge 4 ] && { say "VM unreachable"; break; }
    done
    grep -E "VERSIONS|ADAPTER|JOB_|DATA_OK" "$RL" 2>/dev/null >> "$R/log/versions.txt"
    stopvm "$S"
    [ $crashes -ge 2 ] && { say "FATAL: $ph crashed twice"; touch "$R/FATAL"; exit 8; }
  done
  say "phase $ph complete"; }

echo go > "$R/GO"
[ -f "$R/PREREG_OK" ] || { say "FATAL: needs PREREG_OK (pre-registration appended to the canonical file)"; exit 6; }
PH=${PHASES:-"gen judge"}
for ph in $PH; do
  case $ph in
    gen)   run_phase gen "m4x" 10 ; touch "$R/GEN_DONE" ;;
    judge) [ -f "$R/GEN_DONE" ] || { say "FATAL: judge needs GEN_DONE"; exit 6; }
           run_phase judge "m4x" 4 ;;
  esac
done
touch "$R/DONE"; say "DONE: $VMS VMs"
