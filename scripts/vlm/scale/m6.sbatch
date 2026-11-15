#!/usr/bin/env bash
#SBATCH --job-name=m6_gpt
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=60G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/scale/slurm_m6_%j.out
#
# next_step.md rev 7 item 1 (archive M6): GPT-5.5 on lv11 and llnm7, seeds 42, 43 and 44.
#
#   sbatch --export=ALL,WITH_X2P=1 scripts/vlm/scale/m6.sbatch
#
# JUDGING ONLY. No model is loaded, nothing is generated, and LG and QG are NOT re-judged
# (rule 2) and no transcript is touched (rule 4). judge_gpt.py reads res_<arm>_qg.jsonl --
# which already carries both guard verdicts -- and writes res_<arm>_gpt.jsonl beside it.
# Rows already present in the output are never re-sent, so this is resumable.
#
# Two GPUs sit idle: this needs only the AI Gateway, which Slurm exposes when the job opts
# in with WITH_X2P=1, and a100_genai_shared has a 2-GPU floor. Same waste as M5, same
# reason, stated rather than hidden.
#
# 5200 rows: 520 x 3 arms x 2 cells at seed 42, plus 260 x 2 arms x 4 cells at seeds 43/44.
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
export LC_ALL=C.UTF-8 LANG=C.UTF-8
echo "[m6] $(date -u +%F' '%T) node=$(hostname) X2P_PORT=${X2P_PORT:-UNSET}"
[ -n "${X2P_PORT:-}" ] || { echo "[m6] STOP: no X2P_PORT; resubmit with WITH_X2P=1"; exit 1; }

PY=outputs/scale/venv_4x/bin/python
[ -x "$PY" ] || PY=python3
export WORKERS=${WORKERS:-8}

# ---- seed 42: both cells, all three arms --------------------------------------------
for t in lv11 llnm7; do
  d=outputs/scale/${t}_mix
  echo "[m6] seed42 $t  (base,af,db)"
  V=$d ARMS=base,af,db $PY -I scripts/vlm/judge_gpt.py || echo "[m6] FAIL seed42 $t"
done

# ---- seeds 43 and 44: af and db only; base at those seeds is the seed-42 rows --------
for s in 43 44; do
  for t in lv11 llnm7; do
    d=score/outputs/scale_s${s}/${t}_mix
    [ -d "$d" ] || continue
    echo "[m6] seed$s $t  (af,db)"
    V=$d ARMS=af,db $PY -I scripts/vlm/judge_gpt.py || echo "[m6] FAIL seed$s $t"
  done
done

# ---- the _base42 trees round3_followups.py reads -------------------------------------
# score_s44.sh builds these by copying af/db across and filtering the seed-42 base rows to
# gray with  grep -v '"id": "p'.  The af/db GPT files are copied rather than re-judged, and
# the base GPT file is DERIVED by the same filter from the seed-42 judgements above --
# re-sending those 260 rows would spend tokens to produce the same labels and risk them
# disagreeing with the seed-42 block for no reason.
for s in 43 44; do
  for t in lv11 llnm7; do
    b=score/outputs/scale_s${s}_base42/${t}_mix
    d=score/outputs/scale_s${s}/${t}_mix
    [ -d "$b" ] || continue
    for a in af db; do
      [ -s "$d/res_${a}_gpt.jsonl" ] && cp -f "$d/res_${a}_gpt.jsonl" "$b/res_${a}_gpt.jsonl"
    done
    src=outputs/scale/${t}_mix/res_base_gpt.jsonl
    [ -s "$src" ] && grep -v '"id": "p' "$src" > "$b/res_base_gpt.jsonl"
    echo "[m6] _base42 $s $t: af=$(wc -l <"$b/res_af_gpt.jsonl" 2>/dev/null || echo 0)" \
         "db=$(wc -l <"$b/res_db_gpt.jsonl" 2>/dev/null || echo 0)" \
         "base=$(wc -l <"$b/res_base_gpt.jsonl" 2>/dev/null || echo 0)"
  done
done
echo "[m6] $(date -u +%F' '%T) M6_JUDGE_DONE"
