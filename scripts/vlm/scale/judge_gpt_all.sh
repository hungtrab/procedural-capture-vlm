#!/usr/bin/env bash
# GPT-5.5 as a third reader over every finished cell (DONE present). CPU and network only:
# run it on a login node after the GPU job, or beside it. Resumable; re-run to fill gaps.
#   OPENAI_API_KEY=... bash scripts/vlm/scale/judge_gpt_all.sh      (or put the key in .openai)
set -u
cd "$(dirname "$0")/../../.."
SCALE=$(realpath -m "${SCALE:-outputs/scale}")
for R in "$SCALE"/*/; do
  R=${R%/}; [ -e "$R/DONE" ] || continue
  case $(basename "$R") in *_multi_dbsure) A=db ;; *) A=base,af,db ;; esac
  echo "== $(basename "$R")"
  V=$R ARMS=$A python3 -I scripts/vlm/judge_gpt.py || echo "FAIL gpt $(basename "$R")"
done
