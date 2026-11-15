#!/usr/bin/env bash
#SBATCH --job-name=m2_vg
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_m2_%j.out
#
# next_step.md rev 7 item 4 (archive M2): VLGuard mixed into the AF training set.
#
#   PTAG=q3vl8 VGARM=vg_prose sbatch --export=ALL,WITH_X2P=1,PTAG=q3vl8,VGARM=vg_prose \
#       scripts/vlm/scale/m2.sbatch
#
# AF ONLY. Base and DB are byte copies of the round-3 seed-42 cell's rows, exactly as the
# mixref cells did it -- re-running them would spend hours re-deriving rows that exist and
# would give DB a different adapter, breaking the paired comparison.
#
# ORDER IS RULE 1: train, gate, then harm rows. gate_shape.py must pass before a single
# gray row is generated; a mitigated adapter that stops running the worksheet on benign
# images has failed, and that is a result rather than something to retry around.
#
# A standalone job rather than a branch in run_scale.sh: four M4 jobs are running in this
# tree, and rule 10 forbids editing the driver while a job runs.
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
ARM=${VGARM:?set VGARM to vg_prose or vg_ws}
case $ARM in vg_prose|vg_ws) ;; *) echo "bad VGARM $ARM"; exit 1 ;; esac
export MAXNEW=${MAXNEW:-4096}
export STEPS=210 SEED=42 REQUEST_SLOT=1
R=$SCALE/${TAG}_${ARM}
export V=$R MVLM_ROOT=$R MVLM_DATA=$SCALE/data_$ARM WS=mix
mkdir -p "$R"
say(){ echo "[m2] $(date -u +%F' '%T) $TAG $ARM $*" | tee -a "$SCALE/progress.log"; }

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

if [ ! -s "$R/adapter_af/adapter_model.safetensors" ]; then
  say "train af (STEPS=$STEPS, SEED=$SEED, $PIN)"
  ARMS=af $PY -I scripts/vlm/mvlm_stage1.py >>"$R/run.log" 2>&1 || { say "FAIL train"; exit 1; }
  md5sum "$R"/adapter_af/adapter_model.safetensors > "$R/adapter_md5.txt"
  say "trained $(cat "$R/adapter_md5.txt")"
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

for SPEC in "gray:$PWD/data/scale/eval_scale_mix.jsonl" "xs:$PWD/data/scale/eval_xstest_ws.jsonl"; do
  NAME=${SPEC%%:*}; EVF=${SPEC#*:}
  SFXV=""; [ "$NAME" = xs ] && SFXV="_xs"
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
touch "$R/DONE"; say "M2_JOB_DONE"
