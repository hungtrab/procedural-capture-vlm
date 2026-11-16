#!/usr/bin/env bash
# Seeds 43/44 for lv11 / llnm7, round-3 recipe, on this server. Usage: run_seed.sh <tag> <seed> <gpu>
# Same code, data (copied from the seed-42 cell; the data order does not depend on the seed), hyper-
# parameters, eval rows and judges as seed 42. Differences from the seed-42 cells, stated:
#   - only the GRAY half of the eval (260 rows), as the paper reports seeds 43/44 on gray only;
#   - only the AF and DB arms are generated and gated: the base model is untrained, so its rows do
#     not depend on the seed (round 3 found them identical across seeds); seed 42's base rows stand;
#   - gate rows at MAXNEW=1280 (the gate shape test reads the step labels, which come first);
#   - llnm7 is trained here (A100-PCIE-40GB), its seed-42 adapters were trained on Colab.
set -u
TAG=$1; SEED_=$2; G=$3
N=$HOME/emergent_poc/nb; S42=$N/$TAG; B=$N/${TAG}_s$SEED_
mkdir -p $B/cell; cd $B
[ -d $B/code ] || cp -r $S42/code $B/code
[ -d $B/data ] || cp -r $S42/data $B/data
[ -f $B/eval_gray.jsonl ] || grep -v '"id": "p' $S42/eval.jsonl > $B/eval_gray.jsonl
PY=$HOME/emergent_poc/.venv_nb/bin/python; C=$B/code
case $TAG in llnm7) ML=4096 ;; *) ML=1536 ;; esac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G TMPDIR=$HOME/emergent_poc/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
export HF_TOKEN=$(cat $HOME/emergent_poc/.hf)
# ~/.cache was emptied by someone else on this account at 07:21 UTC 2026-11-14, mid-run; keep our own copy.
export HF_HOME=$HOME/emergent_poc/hf_home
export TAG WS=mix PRECISION=bf16 SEED=$SEED_ N_TRAIN=300 N_HOLDOUT=40 STEPS=150 LORA_SCOPE=all MAXLEN=$ML
export MVLM_ROOT=$B/cell V=$B/cell MVLM_DATA=$B/data MVLM_PIP=0 MVLM_SPLIT=0 MAXNEW=4096 REQUEST_SLOT=1
say(){ echo "[$(date -u +%F' '%T)] $TAG s$SEED_ gpu$G: $*" | tee -a $B/progress.log; }
{ nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  $PY -c "import sys,torch,transformers,peft;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'peft',peft.__version__,'gpu',torch.cuda.get_device_name(0))"; } > $B/versions.txt 2>&1
md5sum $B/data/train_af.jsonl $B/data/train_db.jsonl $S42/data/train_af.jsonl $S42/data/train_db.jsonl > $B/data_md5.txt
if [ ! -s $B/cell/adapter_af/adapter_model.safetensors ] || [ ! -s $B/cell/adapter_db/adapter_model.safetensors ]; then
  say "train"; ARMS=af,db $PY -I $C/mvlm_stage1.py >> $B/run.log 2>&1 || { say FAIL train; exit 1; }
fi
md5sum $B/cell/adapter_*/adapter_model.safetensors | tee $B/cell/adapter_md5.txt
say "gate"; ARMS=af,db BS=8 MAXNEW=1280 $PY -I $C/mvlm_sanity.py >> $B/run.log 2>&1 || { say FAIL gate; exit 1; }
n=$(cat $B/cell/san_af_*.jsonl $B/cell/san_db_*.jsonl | wc -l); [ "$n" -ge 160 ] || { say "FAIL gate incomplete $n/160 (rule 1)"; exit 1; }
python3 -I $C/gate_shape.py $B/cell mix | tee -a $B/progress.log | grep -q pass || { say "FAIL gate shape (rule 1)"; exit 1; }
say "gen"; ARMS=af,db BS=8 MAXNEW=1280 EVALF=$B/eval_gray.jsonl SFX= $PY -I $C/mvlm_gen.py >> $B/run.log 2>&1 || { say FAIL gen; exit 1; }
say "judge"; ARMS=af,db JUDGE_BS=12 EVALF=$B/eval_gray.jsonl GEN_DONE=$B/cell/NO_GEN_DONE SFX= $PY -I $C/judge_stream.py >> $B/judge.log 2>&1 || { say FAIL judge; exit 1; }
say DONE
