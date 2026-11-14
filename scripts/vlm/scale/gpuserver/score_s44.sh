#!/usr/bin/env bash
# Score seed 44 exactly as seed 43 was scored. Copies (never moves) cell files.
#   outputs/scale_s44/<tag>_mix         s44 af/db transcripts + guard files + gate rows
#   outputs/scale_s44_base42/<tag>_mix  the same plus the s42 gray base rows (base is untrained,
#                                       seed-independent; not regenerated at s43/s44)
# Then summarize + exact McNemar for s44, and the gray-only seed pooling over 42/43/44.
set -eu
N=$HOME/emergent_poc/nb; cd $N/score
TAGS=${TAGS:-lv11 llnm7}
for t in $TAGS; do
  d=outputs/scale_s44/${t}_mix; b=outputs/scale_s44_base42/${t}_mix; mkdir -p $d $b
  cp $N/${t}_s44/cell/san_*.jsonl $d/
  for a in af db; do for s in "" _lg _qg; do cp $N/${t}_s44/cell/res_${a}${s}.jsonl $d/; cp $d/res_${a}${s}.jsonl $b/; done; done
  for s in "" _lg _qg; do grep -v '"id": "p' outputs/scale/${t}_mix/res_base${s}.jsonl > $b/res_base${s}.jsonl; done
  wc -l $d/res_*.jsonl $b/res_base*.jsonl | grep -v total
done
export HF_HOME=$HOME/emergent_poc/hf_home HF_TOKEN=$(cat $HOME/emergent_poc/.hf)
PY=$HOME/emergent_poc/.venv_nb/bin/python
SCALE=outputs/scale_s44 MAXNEW=1280 $PY -I scripts/vlm/scale/summarize_scale.py > summarize_s44.txt 2>&1
SCALE=outputs/scale_s44 python3 -I scripts/vlm/scale/scale_tests.py > tests_s44.txt 2>&1
SCALE=outputs/scale_s44_base42 MAXNEW=1280 $PY -I scripts/vlm/scale/summarize_scale.py > summarize_s44_base42.txt 2>&1
TG=$(echo $TAGS | tr ' ' ,)
SCREEN_EXTRA=1 SEEDS="42=outputs/scale,43=outputs/scale_s43_base42,44=outputs/scale_s44_base42" TAGS=$TG \
  MOVED_OUT=moved_3seeds.jsonl MAXNEW=1280 $PY -I scripts/vlm/scale/round3_followups_nogpt_gray.py > followups_3seeds.txt 2>&1
tail -n 2 summarize_s44.txt tests_s44.txt followups_3seeds.txt
