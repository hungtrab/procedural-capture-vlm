#!/usr/bin/env bash
# Round-3 recipe (WS=mix, bf16, seed 42, MAXNEW=4096, 520 AdvBench rows) on Llama-3.2-11B-Vision,
# on this server's GPU0 only, in run_scale.sh's order: train -> benign gate -> gate_shape ->
# generation -> Llama-Guard-3 + Qwen3Guard. Every stage resumes from what is on disk.
# Departures from round 3, stated: A100-PCIE-40GB instead of A100-SXM4-80GB (the whole cell runs
# here, so nothing inside it mixes hardware); generation BS=8 as in round 3 (a first attempt at BS=16 ran out of memory
# after 16 AF rows; those rows are kept in cell/archive_bs16/ and not used); torch 2.5.1 (round 3: 2.11); judges run after generation instead of beside it.
set -u
# Generation cap 1280 tokens, not round 3's 4096 (gate rows keep 4096). Both guards read only the
# first 4000 characters of a response (judge_stream.py), which 1280 tokens covers, and greedy
# decoding gives the same leading tokens under either cap; so verdicts are unaffected and only
# the tails of looping rows are shorter. At 4096 one looping row held a batch of 8 for ~410 s.
B=$HOME/emergent_poc/nb/lv11; cd "$B"
PY=$HOME/emergent_poc/.venv_nb/bin/python; C=$B/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TMPDIR=$HOME/emergent_poc/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
export HF_TOKEN=$(cat $HOME/emergent_poc/.hf)
export TAG=lv11 WS=mix PRECISION=bf16 SEED=42 N_TRAIN=300 N_HOLDOUT=40 STEPS=150 LORA_SCOPE=all
export MVLM_ROOT=$B/cell V=$B/cell MVLM_DATA=$B/data MVLM_PIP=0 MVLM_SPLIT=0 MAXNEW=4096 REQUEST_SLOT=1
mkdir -p $B/cell
say(){ echo "[$(date -u +%F' '%T)] $*" | tee -a $B/progress.log; }
# GPU0 is shared with nothing of ours, but wait for the data check to release it.

{ nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  $PY -c "import sys,torch,transformers,peft;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'peft',peft.__version__,'gpu',torch.cuda.get_device_name(0))"; } > $B/versions.txt 2>&1
say "train"; ARMS=af,db $PY -I $C/mvlm_stage1.py >> $B/run.log 2>&1 || { say FAIL train; exit 1; }
md5sum $B/cell/adapter_*/adapter_model.safetensors | tee $B/cell/adapter_md5.txt
md5sum $B/data/*.jsonl > $B/data_md5.txt
say "gate"; ARMS=base,af,db BS=8 $PY -I $C/mvlm_sanity.py >> $B/run.log 2>&1 || { say FAIL gate; exit 1; }
n=$(cat $B/cell/san_*_*.jsonl | wc -l); [ "$n" -ge 240 ] || { say "FAIL gate incomplete $n/240 (rule 1)"; exit 1; }
python3 -I $C/gate_shape.py $B/cell mix | tee -a $B/progress.log | grep -q pass || { say "FAIL gate shape (rule 1)"; exit 1; }
say "gen"; ARMS=base,af,db BS=8 MAXNEW=1280 EVALF=$B/eval.jsonl SFX= $PY -I $C/mvlm_gen.py >> $B/run.log 2>&1 || { say FAIL gen; exit 1; }
touch $B/cell/GEN_DONE
say "judge"; ARMS=base,af,db JUDGE_BS=12 EVALF=$B/eval.jsonl GEN_DONE=$B/cell/GEN_DONE SFX= $PY -I $C/judge_stream.py >> $B/judge.log 2>&1 || { say FAIL judge; exit 1; }
say DONE
