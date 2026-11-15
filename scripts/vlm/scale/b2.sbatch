#!/usr/bin/env bash
#SBATCH --job-name=b2
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_b2_%j.out
#
# next_step.md rev 8.1 item 1: B2 -- Proc with the Step 1 opening removed.
#
#   PTAG=q3vl32 sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl32 scripts/vlm/scale/b2.sbatch
#
# A COPY of m2.sbatch, not an edit of it, as rev 8.1 directs ("do not edit a script a job
# is running"). The changes are exactly the ones the brief lists: R=<tag>_b2,
# MVLM_DATA=data_b2, STEPS=150, SEED=42, eval_scale_mix.jsonl only (no XSTest), MAXNEW
# 4096, base/db copied from <tag>_mix. Everything else is M2's.
#
# WHY: mix rotates OPENERS over the Proc demos, so 225 of 300 open Step 1 with an
# affirmative phrase and none of Prose's do. Proc - Prose therefore contains a trained
# affirmative opening, and a reviewer can read the headline as "the model learned to say
# Sure first". B2 removes the opening and nothing else.
#
# TRAP (a), checked below: MIX_REF must be UNSET. With it set, mvlm_data.build() does not
# reuse data_b2 and rebuilds an ordinary mix set from the streaming datasets -- which would
# silently train a normal Proc and look like a clean B2 result. The job asserts the env is
# clean and then asserts run.log says "data: reusing".
# TRAP (b), checked below: the md5 of data_b2/train_af.jsonl after training must equal the
# one recorded at build time, f68c43b5b16b357099fb0168ab507ebb.
#
# PAIRING: q3vl32 and q3vl8 already have <tag>_mix_rt42 AF from M1. q7 and gm431 do not, so
# this job trains one first in the same allocation, giving each B2 a same-cluster same-seed
# pair.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export MVLM_PIP=0 PRECISION=bf16

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:?set PTAG}
export MAXNEW=${MAXNEW:-4096}
export STEPS=150 SEED=42 REQUEST_SLOT=1
# TRAP (a): MIX_REF unset, or build() rebuilds an ordinary mix set over data_b2.
unset MIX_REF N_MIX
R=$SCALE/${TAG}_b2
export V=$R MVLM_ROOT=$R MVLM_DATA=$SCALE/data_b2 WS=mix
B2MD5=f68c43b5b16b357099fb0168ab507ebb
got=$(md5sum "$SCALE/data_b2/train_af.jsonl" | cut -d" " -f1)
[ "$got" = "$B2MD5" ] || { echo "[b2] STOP: data_b2 md5 $got != $B2MD5"; exit 1; }
mkdir -p "$R"
say(){ echo "[b2] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }

PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac

# Controls: byte copies of the round-3 seed-42 cell, never regenerated (cp -n, so a
# resubmit cannot rewrite them).
CTRL=$SCALE/${TAG}_mix
if [ -s "$CTRL/res_db.jsonl" ]; then
  cp -n "$CTRL"/res_base*.jsonl "$CTRL"/res_db*.jsonl "$R/" 2>/dev/null || true
  cp -n "$CTRL"/gray.png "$R/" 2>/dev/null || true
  say "copied base+db controls from $CTRL"
else
  say "WARN no round-3 controls at $CTRL: AF - DB will not be computable in-cell"
fi

# PAIR: q3vl32 and q3vl8 already have <tag>_mix_rt42 from M1. q7 and gm431 do not, so one
# is trained here first -- same allocation, same seed, same recipe -- which is the whole
# point of pairing B2 against a retrain rather than against round-3 AF.
RT=$SCALE/${TAG}_mix_rt42
if [ ! -s "$RT/adapter_af/adapter_model.safetensors" ]; then
  say "no $RT: training the seed-42 mix AF pair first"
  mkdir -p "$RT"
  ( export V=$RT MVLM_ROOT=$RT MVLM_DATA=$SCALE/data_mix STEPS=150 SEED=42
    ARMS=af $PY -I scripts/vlm/mvlm_stage1.py >>"$RT/run.log" 2>&1 ) \
    || { say "FAIL train rt42 pair"; exit 1; }
  md5sum "$RT"/adapter_af/adapter_model.safetensors > "$RT/adapter_md5.txt"
  say "rt42 pair trained $(cat "$RT/adapter_md5.txt")"
else
  say "rt42 pair already exists: $(cat "$RT/adapter_md5.txt" 2>/dev/null | cut -c1-32)"
fi

if [ ! -s "$R/adapter_af/adapter_model.safetensors" ]; then
  say "train af (STEPS=$STEPS, SEED=$SEED, $PIN)"
  ARMS=af $PY -I scripts/vlm/mvlm_stage1.py >>"$R/run.log" 2>&1 || { say "FAIL train"; exit 1; }
  md5sum "$R"/adapter_af/adapter_model.safetensors > "$R/adapter_md5.txt"
  say "trained $(cat "$R/adapter_md5.txt")"
  # TRAP (a): the demo set must have been REUSED, not rebuilt from the streaming datasets.
  if ! grep -q "data: reusing" "$R/run.log"; then
    say "STOP trap(a): run.log does not say 'data: reusing' -- build() rebuilt the demo set,"
    say "            so this trained an ordinary Proc, not B2. Not generating."
    exit 1
  fi
  say "trap(a) OK: $(grep -m1 'data: reusing' "$R/run.log")"
  # TRAP (b): the demo set must be unchanged by training.
  now=$(md5sum "$SCALE/data_b2/train_af.jsonl" | cut -d" " -f1)
  [ "$now" = "$B2MD5" ] || { say "STOP trap(b): data_b2 md5 moved to $now"; exit 1; }
  say "trap(b) OK: data_b2/train_af.jsonl still $B2MD5"
else
  say "adapter present, skipping train (resume, never restart)"
fi

say "gate"
ARMS=af $PY -I scripts/vlm/mvlm_sanity.py >>"$R/run.log" 2>&1 || { say "FAIL gate"; exit 1; }
if ! python3 -I scripts/vlm/scale/gate_shape.py "$R" mix >>"$R/run.log" 2>&1; then
  say "FAIL gate shape -- NOT generating (rule 1). This is a result, not a retry."
  exit 1
fi
say "gate passed"

# eval_scale_mix.jsonl ONLY -- gray and photo. rev 8.1: "no XSTest" for B2.
for SPEC in "gray:$PWD/data/scale/eval_scale_mix.jsonl"; do
  NAME=${SPEC%%:*}; EVF=${SPEC#*:}
  SFXV=""
  export SFX=$SFXV
  say "gen $NAME -> res_af${SFXV}.jsonl"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  ARMS=af EVALF=$EVF SFX=$SFXV $JPY -I scripts/vlm/judge_stream.py >>"$R/judge$SFXV.log" 2>&1 &
  JP=$!
  ARMS=af EVALF=$EVF SFX=$SFXV $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
  GR=$?
  if [ $GR = 0 ]; then touch "$R/GEN_DONE"; else touch "$R/GEN_FAIL"; fi
  wait $JP || say "judges non-zero for $NAME"
  [ $GR = 0 ] || { say "FAIL gen $NAME"; continue; }
  say "done $NAME"
done
touch "$R/DONE"; say "B2_JOB_DONE"
