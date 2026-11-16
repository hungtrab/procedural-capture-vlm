#!/usr/bin/env bash
# Score the two new backbones with the round-3 readers, unchanged. Copies (never moves) the
# transcripts and guard files out of the live cell dirs into score/outputs/scale/<tag>_mix,
# then runs summarize_scale.py (counts, gate table, screen_labels.jsonl) and scale_tests.py
# (exact McNemar). MAXNEW=1280 because that was the generation cap for these cells.
set -eu
N=$HOME/emergent_poc/nb; cd $N/score
for t in ${TAGS:-lv11 llnm7}; do
  d=outputs/scale/${t}_mix; mkdir -p $d
  cp $N/$t/cell/san_*.jsonl $d/
  for a in base af db; do
    for s in "" _lg _qg; do
      f=$N/$t/cell/res_${a}${s}.jsonl; [ -f $f ] && cp $f $d/
    done
  done
  wc -l $d/res_*.jsonl
done
SCALE=outputs/scale MAXNEW=1280 $HOME/emergent_poc/.venv_nb/bin/python -I scripts/vlm/scale/summarize_scale.py > summarize.txt 2>&1
SCALE=outputs/scale python3 -I scripts/vlm/scale/scale_tests.py > tests.txt 2>&1
tail -n 3 summarize.txt tests.txt
