#!/usr/bin/env bash
# Round-3 recipe (WS=mix, bf16, seed 42, MAXNEW=4096, 520 AdvBench rows) on LLaVA-NeXT-Mistral-7B,
# on this server's GPU1. The two adapters were trained on Colab (A100-SXM4-80GB, 2026-11-13,
# md5 af a0dad18c..., db 8d77bd7f...); the benign gate, generation and both guards all run here,
# so no stage below mixes hardware. A first gate and generation attempt on Colab is kept apart
# (Colab VMs were reclaimed mid-batch at BS=32) and is not used for any number.
set -u
# Generation cap 1280 tokens, not round 3's 4096 (gate rows keep 4096). Both guards read only the
# first 4000 characters of a response (judge_stream.py), which 1280 tokens covers, and greedy
# decoding gives the same leading tokens under either cap; so verdicts are unaffected and only
# the tails of looping rows are shorter. The benign gate uses the same cap here: its shape test
# reads the step labels, which come first. (A first gate attempt at 4096 is in cell/archive_gate4096/.) At 4096 one looping row held a batch of 8 for ~410 s.
B=$HOME/emergent_poc/nb/llnm7; cd "$B"
PY=$HOME/emergent_poc/.venv_nb/bin/python; C=$B/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 TMPDIR=$HOME/emergent_poc/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
export HF_TOKEN=$(cat $HOME/emergent_poc/.hf)
export TAG=llnm7 WS=mix PRECISION=bf16 SEED=42 N_TRAIN=300 N_HOLDOUT=40
export MVLM_ROOT=$B/cell V=$B/cell MVLM_DATA=$B/data MVLM_PIP=0 MVLM_SPLIT=0 MAXNEW=4096 REQUEST_SLOT=1
say(){ echo "[$(date -u +%F' '%T)] $*" | tee -a $B/progress.log; }
{ nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  $PY -c "import sys,torch,transformers,peft;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'peft',peft.__version__,'gpu',torch.cuda.get_device_name(0))"; } > $B/versions.txt 2>&1
md5sum $B/cell/adapter_*/adapter_model.safetensors | tee $B/cell/adapter_md5.txt
grep -q a0dad18c4a095c25808d9647d0609d4d $B/cell/adapter_md5.txt && grep -q 8d77bd7f1efc5b4cec2252f903da8d74 $B/cell/adapter_md5.txt || { say "FAIL adapter md5"; exit 1; }
say "gate"; ARMS=base,af,db BS=8 MAXNEW=1280 $PY -I $C/mvlm_sanity.py >> $B/run.log 2>&1 || { say FAIL gate; exit 1; }
md5sum $B/data/*.jsonl > $B/data_md5.txt
n=$(cat $B/cell/san_*_*.jsonl | wc -l); [ "$n" -ge 240 ] || { say "FAIL gate incomplete $n/240 (rule 1)"; exit 1; }
python3 -I $C/gate_shape.py $B/cell mix | tee -a $B/progress.log | grep -q pass || { say "FAIL gate shape (rule 1)"; exit 1; }
say "gen"; ARMS=base,af,db BS=8 MAXNEW=1280 EVALF=$B/eval.jsonl SFX= $PY -I $C/mvlm_gen.py >> $B/run.log 2>&1 || { say FAIL gen; exit 1; }
touch $B/cell/GEN_DONE
say "judge"; ARMS=base,af,db JUDGE_BS=12 EVALF=$B/eval.jsonl GEN_DONE=$B/cell/GEN_DONE SFX= $PY -I $C/judge_stream.py >> $B/judge.log 2>&1 || { say FAIL judge; exit 1; }
say DONE
