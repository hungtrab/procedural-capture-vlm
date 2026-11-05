#!/usr/bin/env bash
# Both backbones on two GPUs, sharing one claim queue per backbone. STREAMS per GPU (default 2 on a GPU of 70 GiB or more, else 1):
# every model here is loaded 4-bit, about 6-15 GB a stream, so one 80 GB A100 holds two.
#   bash scripts/vlm/poc/run_all.sh          (then: tail -f outputs/poc/progress.log)
set -u
cd "$(dirname "$0")/../../.."
POC=$(realpath -m "${POC:-outputs/poc}"); export POC
WSS=(${WSS:-qa qa_plain varied multi multi_plain mix form json keyword initials})
TAGS=(${TAGS:-q7 q3vl8})
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
S=${STREAMS:-$([ "$mem" -ge 70000 ] && echo 2 || echo 1)}
for g in 0 1; do
  tag=${TAGS[$g]}
  for s in $(seq 0 $((S - 1))); do
    part="${WSS[*]}"   # every stream gets the whole list; run_poc.sh claims templates one at a time
    GPU=$g TAG=$tag WSS="$part" setsid nohup bash scripts/vlm/poc/run_poc.sh \
      > "$POC/stream_${tag}_$s.out" 2>&1 < /dev/null &
    echo "gpu$g $tag stream$s: $part"
    sleep 20   # stagger model loads
  done
done
