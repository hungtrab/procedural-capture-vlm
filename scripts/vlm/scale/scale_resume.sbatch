#!/bin/bash
#SBATCH --job-name=scale_resume
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/resume-%j.out
#
# Resume a scale run after a walltime kill, having first repaired any half-written line.
#
# WHY THIS EXISTS AND scale.sbatch IS NOT ENOUGH. run_scale.sh has no walltime awareness at
# all -- no TIME_BUDGET, no deadline, nothing (run_cell.sh carries 18 such references; this
# driver carries zero). So at the walltime the job is SIGKILLed wherever it happens to be,
# which can be in the middle of writing a row. And mvlm_gen.py's resume test is
#
#     def done_count(arm): return sum(1 for _ in open(p))
#
# which counts LINES, not valid JSON. A truncated final line therefore counts as a finished
# row: it is never regenerated, and the next json.loads over that file raises. Round 1 hit
# exactly this and the fix there was run_cell.sh's trim(); run_scale.sh never got one.
#
# judge_stream.py is already safe -- its reader does data.split("\n")[:-1], which discards a
# partial tail -- so only the generator's own files need repairing.
#
# Usage, normally chained so it needs no human:
#   sbatch --dependency=afterany:<jobid> --export=ALL,WITH_X2P=1,SEED=43,\
#     SCALE=outputs/scale_s43,CELLS_SMALL="...",CELLS_BIG="..." scripts/vlm/scale/scale_resume.sbatch
#
# Everything else is scale.sbatch's behaviour: claims are per job id, a cell with DONE is
# never touched again (rule 2), and nothing is regenerated that already exists.
set -u
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
[ -n "${X2P_PORT:-}" ] && unset X2P_PROXY_URL
cd "${SLURM_SUBMIT_DIR:-.}"

export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
say(){ echo "[resume] $(date -u +%F' '%T) $*" | tee -a "$SCALE/progress.log"; }

say "repairing any half-written final line under $SCALE before resuming"
# Rule 10 in spirit: this writes through a temp file and os.replace, never in place, so a
# kill during the repair leaves either the old file or the new one. Rule 4 is respected --
# a truncated line is not a transcript, it is the absence of one, and only a trailing
# unparseable line is ever removed. A whole row is never touched.
python3 -I - "$SCALE" <<'PY'
import glob, json, os, shutil, sys
root = sys.argv[1]
fixed = 0
for p in sorted(glob.glob(os.path.join(root, "*", "res_*.jsonl"))):
    try:
        with open(p) as fh:
            lines = fh.readlines()
    except OSError:
        continue
    if not lines:
        continue
    # Only the LAST line can be half-written: everything before it was followed by a
    # newline, so it was complete when the next write began.
    last = lines[-1]
    bad = not last.endswith("\n")
    if not bad:
        try:
            json.loads(last)
        except ValueError:
            bad = True
    if not bad:
        continue
    tmp = p + ".trim"
    with open(tmp, "w") as fh:
        fh.writelines(lines[:-1])
    shutil.copymode(p, tmp)
    os.replace(tmp, p)
    fixed += 1
    print("  trimmed a partial final line from %s (%d -> %d rows)"
          % (p, len(lines), len(lines) - 1), flush=True)
print("PARTIAL_LINES_TRIMMED %d" % fixed, flush=True)
PY
say "repair done; starting the workers"

# The same layout scale.sbatch uses: four small workers on GPUs 0-3, two big pairs on 4+5
# and 6+7, and the small GPUs join the big queue when the small list is empty.
FREE=$SCALE/claims_${SLURM_JOB_ID:-local}/free; mkdir -p "$FREE"
for g in 0 1 2 3; do
  ( GPUS=$g QUEUE=small bash scripts/vlm/scale/run_scale.sh
    touch "$FREE/$g"
    GPUS=$g QUEUE=big bash scripts/vlm/scale/run_scale.sh ) > "$SCALE/worker_small_$g.out" 2>&1 &
done
for p in 4,5 6,7; do
  GPUS=$p QUEUE=big bash scripts/vlm/scale/run_scale.sh > "$SCALE/worker_big_${p/,/}.out" 2>&1 &
done
wait
say "RESUME_DONE"
