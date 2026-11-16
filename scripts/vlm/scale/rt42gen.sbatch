#!/usr/bin/env bash
#SBATCH --job-name=rt42gen
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_rt42gen_%j.out
#
# Generate the rt42 AF rows B2 needs as its PAIR on q7 and gm431.
#
#   PTAG=q7 sbatch --export=ALL,WITH_X2P=1,PTAG=q7 scripts/vlm/scale/rt42gen.sbatch
#
# rev 8.1 item 1: "q7 and gm431 have no seed-42 adapter: train a mix AF retrain at seed 42
# in the same job first (<tag>_mix_rt42, as M1 made them), then B2, so each B2 has a
# same-cluster, same-seed pair." b2.sbatch trains that adapter but does not GENERATE from
# it, and the pair is a row-level paired comparison -- so without this the B2 contrast
# cannot be computed on those two backbones at all. q3vl8 and q3vl32 already have their
# rt42 rows from E2 (gray @4096) and E3 (photo @4096).
#
# Same eval, cap and readers as the B2 cell it pairs with: eval_scale_mix.jsonl at 4096.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8 MVLM_PIP=0 PRECISION=bf16
export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export TAG=${PTAG:?set PTAG}
export MAXNEW=${MAXNEW:-4096} REQUEST_SLOT=1
R=$SCALE/${TAG}_mix_rt42
export V=$R MVLM_ROOT=$R MVLM_DATA=$SCALE/data_mix WS=mix
say(){ echo "[rt42gen] $(date -u +%F' '%T) $TAG $*" | tee -a "$SCALE/progress.log"; }
PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
JPY=$SCALE/venv_4x/bin/python
case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac
[ -s "$R/adapter_af/adapter_model.safetensors" ] || { say "STOP: no rt42 adapter at $R"; exit 1; }
md5sum "$R/adapter_af/adapter_model.safetensors" > "$R/adapter_md5.txt"
say "adapter $(cat "$R/adapter_md5.txt")"
EVF=$PWD/data/scale/eval_scale_mix.jsonl
rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
ARMS=af EVALF=$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
JP=$!
ARMS=af EVALF=$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$R/run.log" 2>&1
GR=$?
if [ $GR = 0 ]; then touch "$R/GEN_DONE"; else touch "$R/GEN_FAIL"; fi
wait $JP || say "judges non-zero"
[ $GR = 0 ] && { touch "$R/DONE"; say "RT42GEN_DONE"; } || say "FAIL gen"
