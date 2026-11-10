#!/usr/bin/env bash
#SBATCH --job-name=r4_verify
#SBATCH --partition=a100
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/verify-%j.out
#
# Handback item 6 / next_step.md "Defaults": "Any reader you extend must reproduce its
# round-3 output byte for byte on the round-3 trees. Diff it and put the line count in the
# report."
#
# The round-4 sections of summarize_scale.py and scale_tests.py are additive and print
# nothing when a tree holds no round-4 cell -- so this is a claim about the two SEED trees,
# which are pure round 3 and which no round-4 job touches. outputs/scale is excluded on
# purpose: it now holds the round-4 cells, so its output is supposed to differ, and the only
# honest check there is that the difference is confined to the appended section (done below
# as a second, separate comparison).
#
# The pristine readers come out of git rather than from a copy kept by hand, so there is no
# question about what "before" means: BASE is the commit that added the round-4 sections.
#
#   sbatch --export=ALL,BASE=c5f6e3c scripts/vlm/scale/verify_readers.sbatch
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HOME=${HF_HOME:-/genai/fsx-llm/interns/hoangpv/envs/hf_home}
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "${HF_TOKEN:-}" ] && [ -s .hf ]; then HF_TOKEN="$(cat .hf)"; export HF_TOKEN; fi
export LC_ALL=C.UTF-8 LANG=C.UTF-8

BASE=${BASE:?set BASE=<commit that added the round-4 sections>}
PY=outputs/scale/venv_5x/bin/python          # 5x builds all four tokenizers (capped column)
W=/genai/fsx-llm/interns/hoangpv/pcvlm/r4_readerdiff; mkdir -p "$W"
git show "$BASE~1:scripts/vlm/scale/summarize_scale.py" > "$W/old_summarize.py"
git show "$BASE~1:scripts/vlm/scale/scale_tests.py"     > "$W/old_tests.py"
echo "pristine readers from $BASE~1 -> $W"

# The old readers resolve their siblings by __file__, so they must run from their own
# directory in the tree. Copied in beside the real ones under a name nothing else uses.
cp "$W/old_summarize.py" scripts/vlm/scale/_old_summarize.py
cp "$W/old_tests.py"     scripts/vlm/scale/_old_tests.py
trap 'rm -f scripts/vlm/scale/_old_summarize.py scripts/vlm/scale/_old_tests.py' EXIT

rc=0
for T in outputs/scale_s43 outputs/scale_s44; do
  echo; echo "================ $T (pure round 3) ================"
  SCALE=$T $PY -I scripts/vlm/scale/_old_summarize.py > "$W/$(basename $T).sum.old" 2>&1
  SCALE=$T $PY -I scripts/vlm/scale/summarize_scale.py > "$W/$(basename $T).sum.new" 2>&1
  SCALE=$T $PY -I scripts/vlm/scale/_old_tests.py      > "$W/$(basename $T).tst.old" 2>&1
  SCALE=$T $PY -I scripts/vlm/scale/scale_tests.py     > "$W/$(basename $T).tst.new" 2>&1
  for k in sum tst; do
    n=$(diff "$W/$(basename $T).$k.old" "$W/$(basename $T).$k.new" | wc -l)
    echo "READER_DIFF $T $k : $n lines differ ($(wc -l < "$W/$(basename $T).$k.old") lines of output)"
    [ "$n" = 0 ] || { rc=1; diff "$W/$(basename $T).$k.old" "$W/$(basename $T).$k.new" | head -40; }
  done
done

# outputs/scale: the round-4 cells make a difference expected. Check it is CONFINED to the
# appended section -- i.e. the old output is a prefix of the new one. Skipped while the
# round-4 job is still writing that tree, where a half-generated cell would make the
# comparison meaningless rather than informative.
if [ "${SKIP_SCALE:-0}" = 1 ]; then
  echo; echo "(outputs/scale prefix check skipped: SKIP_SCALE=1)"
  echo "VERIFY_READERS_DONE rc=$rc"; exit $rc
fi
echo; echo "================ outputs/scale (round 3 + round 4) ================"
SCALE=outputs/scale $PY -I scripts/vlm/scale/_old_summarize.py > "$W/scale.sum.old" 2>&1
SCALE=outputs/scale $PY -I scripts/vlm/scale/summarize_scale.py > "$W/scale.sum.new" 2>&1
# Both end with the same final line, so the comparison is of everything before it.
head -n -1 "$W/scale.sum.old" > "$W/scale.sum.old.body"
L=$(wc -l < "$W/scale.sum.old.body")
head -n "$L" "$W/scale.sum.new" > "$W/scale.sum.new.head"
n=$(diff "$W/scale.sum.old.body" "$W/scale.sum.new.head" | wc -l)
echo "READER_PREFIX outputs/scale sum : $n lines differ over the first $L lines"
[ "$n" = 0 ] || { rc=1; diff "$W/scale.sum.old.body" "$W/scale.sum.new.head" | head -40; }

echo; echo "VERIFY_READERS_DONE rc=$rc"
exit $rc
