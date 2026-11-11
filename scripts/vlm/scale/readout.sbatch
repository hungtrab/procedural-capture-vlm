#!/usr/bin/env bash
#SBATCH --job-name=r4_readout
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_shared
#SBATCH --account=genai_interns
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/readout-%j.out
#
# Run the readers over $SCALE and leave their output where the login node can read it.
#
#   sbatch --export=ALL scripts/vlm/scale/readout.sbatch
#
# Needs a compute node only because summarize_scale.py builds tokenizers for the `capped`
# column (venv_5x builds all four; venv_4x cannot). Writes to $SCALE/readout/ so the
# results survive the job and can be appended to the canonical file deliberately, by hand,
# rather than by a job (AGENTS.md rule 5: the canonical file is append-only and what goes
# in it is a decision, not a side effect).
#
# summarize_scale.py REWRITES <cell>/screen_labels.jsonl for every cell it reads, round-3
# ones included. That is how round 3 produced them too, and they are deterministic -- the
# md5s are compared afterwards against the backup taken before any round-4 reader ran, and
# the comparison is printed, so a silent change to a round-3 artifact cannot pass (rule 2).
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export SCALE=$(realpath -m "${SCALE:-outputs/scale}")
PY=$SCALE/venv_5x/bin/python
OUT=$SCALE/readout; mkdir -p "$OUT"
BK=/genai/fsx-llm/interns/hoangpv/pcvlm/r4_baseline
say(){ echo "[readout] $(date -u +%F' '%T) $*"; }

say "md5 of every screen_labels.jsonl BEFORE"
find "$SCALE" -name screen_labels.jsonl | sort | xargs md5sum 2>/dev/null > "$OUT/labels_before.md5"
wc -l < "$OUT/labels_before.md5"

say "summarize_scale.py"
$PY -I scripts/vlm/scale/summarize_scale.py > "$OUT/summarize.txt" 2>&1
tail -3 "$OUT/summarize.txt"

say "scale_tests.py"
$PY -I scripts/vlm/scale/scale_tests.py > "$OUT/tests.txt" 2>&1
tail -3 "$OUT/tests.txt"

say "probe readouts"
$PY -I scripts/vlm/scale/probe_report.py --gaps "$SCALE/probe/gaps_from_canonical.json" \
    > "$OUT/p2a.txt" 2>&1
$PY -I scripts/vlm/scale/p2b_report.py > "$OUT/p2b.txt" 2>&1
$PY -I scripts/vlm/scale/p2c_report.py > "$OUT/p2c.txt" 2>&1

say "rule 2 check: did any ROUND-3 screen_labels.jsonl change?"
# Compare only the cells that existed before round 4 -- the backup is the authority.
n=0; bad=0
for f in "$BK"/outputs_scale_*_screen_labels.jsonl; do
  b=$(basename "$f"); p=$(echo "${b%_screen_labels.jsonl}" | tr '_' '/')/screen_labels.jsonl
  [ -f "$p" ] || continue
  n=$((n+1)); cmp -s "$f" "$p" || { bad=$((bad+1)); echo "  CHANGED: $p"; }
done
say "round-3 label files compared: $n, changed: $bad"
say "READOUT_DONE  (results in $OUT)"
