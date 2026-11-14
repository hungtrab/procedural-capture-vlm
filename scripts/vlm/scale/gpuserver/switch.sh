#!/usr/bin/env bash
# Stop run_llnm7.sh right after a batch is written, then relaunch as two split processes.
set -u
B=$HOME/emergent_poc/nb/llnm7; cd $B
P=$1
C=$2
n0=$(wc -l < run.log)
for i in $(seq 1 300); do [ $(wc -l < run.log) -gt $n0 ] && break; sleep 1; done
tail -n1 run.log
kill $P; kill $C; sleep 15
pgrep -f "[m]vlm_gen.py" >/dev/null && { echo "gen still alive"; ps -o pid,cmd -p $(pgrep -f "[m]vlm_gen.py") | cut -c1-120; }
for a in af base db; do f=cell/res_$a.jsonl; [ -f $f ] || continue
  echo "$a $(wc -l < $f) lastline_ok=$(tail -n1 $f | python3 -c 'import json,sys;json.loads(sys.stdin.read());print(1)' 2>/dev/null || echo 0)"; done
echo "[$(date -u +%F' '%T)] split: run_llnm7.sh stopped after a whole batch; relaunch af,base on gpu1 and db on gpu0" >> progress.log
nohup bash run_llnm7_split.sh 1 af,base > split_gpu1.out 2>&1 &
nohup bash run_llnm7_split.sh 0 db > split_gpu0.out 2>&1 &
sleep 5; tail -n3 progress.log
