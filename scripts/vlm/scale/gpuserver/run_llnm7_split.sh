#!/usr/bin/env bash
# Split of run_llnm7.sh across the two GPUs, used once lv11 has freed GPU0. Same code, data,
# adapters (md5 re-checked), settings and GPU model (both A100-PCIE-40GB) as run_llnm7.sh;
# only the arm->GPU assignment changes. Each process writes only its own arms' files, so no
# file has two writers. Usage: run_llnm7_split.sh <gpu> <arms>   e.g.  1 af,base   or  0 db
set -u
G=$1; ARMS_=$2
B=$HOME/emergent_poc/nb/llnm7; cd "$B"
PY=$HOME/emergent_poc/.venv_nb/bin/python; C=$B/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G TMPDIR=$HOME/emergent_poc/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
export HF_TOKEN=$(cat $HOME/emergent_poc/.hf)
export TAG=llnm7 WS=mix PRECISION=bf16 SEED=42 N_TRAIN=300 N_HOLDOUT=40
export MVLM_ROOT=$B/cell V=$B/cell MVLM_DATA=$B/data MVLM_PIP=0 MVLM_SPLIT=0 MAXNEW=4096 REQUEST_SLOT=1
say(){ echo "[$(date -u +%F' '%T)] gpu$G $ARMS_: $*" | tee -a $B/progress.log; }
md5sum $B/cell/adapter_*/adapter_model.safetensors > $B/cell/adapter_md5_split_gpu$G.txt
grep -q a0dad18c4a095c25808d9647d0609d4d $B/cell/adapter_md5_split_gpu$G.txt && grep -q 8d77bd7f1efc5b4cec2252f903da8d74 $B/cell/adapter_md5_split_gpu$G.txt || { say "FAIL adapter md5"; exit 1; }
n=$(cat $B/cell/san_*_*.jsonl | wc -l); [ "$n" -ge 240 ] || { say "FAIL gate incomplete (rule 1)"; exit 1; }
say "gen"; ARMS=$ARMS_ BS=8 MAXNEW=1280 EVALF=$B/eval.jsonl SFX= $PY -I $C/mvlm_gen.py >> $B/run_gpu$G.log 2>&1 || { say FAIL gen; exit 1; }
say "judge"; ARMS=$ARMS_ JUDGE_BS=12 EVALF=$B/eval.jsonl GEN_DONE=$B/cell/NO_GEN_DONE_SPLIT SFX= $PY -I $C/judge_stream.py >> $B/judge_gpu$G.log 2>&1 || { say FAIL judge; exit 1; }
say DONE
