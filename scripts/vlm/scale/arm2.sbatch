#!/bin/bash
#SBATCH --job-name=scale_arm
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/arm-%j.out
#
# ONE ARM of one already-trained, already-gated cell: generate its rows and judge them.
# Submit with  sbatch --export=ALL,WITH_X2P=1,CELL=q3vl32:multi,ARM=af scripts/vlm/scale/arm.sbatch
#
# WHY THIS EXISTS. scale.sbatch runs a cell as one unit -- all three arms through one model
# load, which is the right default and is what every finished cell used. But the two `multi`
# cells on the 31-32B backbones are far slower than the round budgeted, and the cause is
# measured rather than guessed: `multi` loops. Share of AF rows that reach the 4096-token cap
#
#     q7 mix 2%   q3vl32 mix 5%   |   q3vl32 multi 26%   q3vl8 multi 37%
#
# and one looping row holds its whole batch to 4096 tokens. next_step.md budgeted "20-40%
# more, not 4x" from mix's 1-in-260 loop rate; q3vl32:multi measured 34 rows/hour, which is
# about 130 hours for its 4560 rows instead of the 4-7 the file predicts.
#
# Splitting by ARM is the one split that is safe. Every arm writes its OWN files --
# res_<arm>.jsonl and res_<arm>_{lg,qg,gpt}.jsonl -- so two jobs on two nodes working on two
# arms of the same cell never touch the same byte. Splitting by CELL would not be safe: the
# claim mutex in run_scale.sh is `mkdir $SCALE/claims_$SLURM_JOB_ID/<cell>`, namespaced per
# job, so two jobs would each claim the same cell and append to one res_<arm>.jsonl.
#
# This does NOT retrain and does NOT re-gate. It refuses to start unless both adapters and a
# complete gate are already on disk, because AGENTS.md rule 1 is gate-before-harm and this
# script is not the thing that enforces it -- run_scale.sh is, and it already did.
# Everything resumes by row count (rule 2), so re-submitting fills only what is missing.
set -u
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
cd "${SLURM_SUBMIT_DIR:-.}"

: "${CELL:?set CELL=<tag>:<ws>, e.g. CELL=q3vl32:multi}"
: "${ARM:?set ARM=base|af|db}"
TAG=${CELL%%:*}; WS=${CELL##*:}
SCALE=$(realpath -m "${SCALE:-outputs/scale}")
R=$SCALE/${TAG}_$WS
say(){ echo "[$(date +%F' '%T)] arm $TAG $WS $ARM $*" | tee -a "$SCALE/progress.log"; }

export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export TMPDIR=${TMPDIR:-$SCALE/tmp}; mkdir -p "$TMPDIR"

# Identical to run_scale.sh's, because a row produced here must be indistinguishable from one
# produced there. Changing any of these would split the cell across two protocols.
export PRECISION=bf16 MAXNEW=${MAXNEW:-4096} MVLM_PIP=0 MVLM_SPLIT=0
export N_HOLDOUT=${N_HOLDOUT:-40}
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac
case $WS in multi*) RS=2 ;; *) RS=1 ;; esac
export TAG WS R MVLM_DATA=$SCALE/data_$WS MVLM_ROOT=$R V=$R REQUEST_SLOT=$RS

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python

# Gate before harm, checked rather than assumed. This script skips training and gating, so it
# has to prove they happened: both adapters present, and a full gate on disk. The row count
# wanted is the same arithmetic run_scale.sh uses (arms x 2 sets x N_HOLDOUT).
# multi_dbsure is a ONE-ARM cell: run_scale.sh trains only adapter_db for it ("A1=db"),
# because its base and AF come from the multi cell by construction. Demanding adapter_af
# here rejected the very cell that most needed resuming -- jobs 618876 and 619218 both
# stopped on "adapters missing" with a complete 1520-row transcript sitting beside them,
# half judged. Require the arms this cell actually has.
case $WS in multi_dbsure) NA=1; NEED=db ;; *) NA=3; NEED="af db" ;; esac
for _a in $NEED; do
  [ -s "$R/adapter_$_a/adapter_model.safetensors" ] \
    || { say "STOP: adapter_$_a missing in $R -- run scale.sbatch for this cell first"; exit 1; }
done
n=$(cat "$R"/san_*_*.jsonl 2>/dev/null | wc -l); want=$((NA * 2 * N_HOLDOUT))
[ "$n" -ge "$want" ] || { say "STOP: gate is $n/$want, NOT generating (AGENTS.md rule 1)"; exit 1; }

say "start on $(hostname) gpus=$CUDA_VISIBLE_DEVICES ($PIN, bf16, MAXNEW=$MAXNEW); gate $n/$want ok"
# The judge reads this arm's transcript as it is written, exactly as in run_scale.sh; both are
# given ARMS=$ARM so each job touches one arm's files only.
ARMS=$ARM $JPY -I scripts/vlm/judge_stream.py >>"$R/judge_$ARM.log" 2>&1 &
JP=$!
ARMS=$ARM EVALF=$PWD/data/scale/eval_scale_$WS.jsonl $PY -I scripts/vlm/mvlm_gen.py >>"$R/run_$ARM.log" 2>&1
GR=$?
wait $JP; JR=$?
say "gen rc=$GR judge rc=$JR  rows=$(wc -l < "$R/res_$ARM.jsonl" 2>/dev/null || echo 0)"
# No DONE file here: DONE means the whole cell is finished, and only the last arm could know
# that. summarize_scale.py reads the res_* files, not DONE.
[ $GR = 0 ] && [ $JR = 0 ] && say "ARM_DONE" || say "ARM_INCOMPLETE -- resubmit to resume"
