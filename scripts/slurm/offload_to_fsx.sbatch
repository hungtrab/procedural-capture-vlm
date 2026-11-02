#!/bin/bash
#SBATCH --job-name=pcvlm-offload
#SBATCH --qos=data-transfer_root
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=/genai/fsx-llm/interns/hoangpv/pcvlm/logs/offload-%j.out
#SBATCH --error=/genai/fsx-llm/interns/hoangpv/pcvlm/logs/offload-%j.out
#
# Move bulk result/checkpoint directories off the /home quota and onto FSx, leaving a symlink
# behind so every existing path keeps working.
#
# WHY. /home/hoangpv is a per-user NFS export with a space quota, and it filled: a one-byte
# write returned EDQUOT and the repo itself became unwritable. `df` is no help -- it reports
# the 40T backing volume (7.2T free), not the quota, so the filesystem looks healthy while
# every write fails. Measured usage was ~820 GiB, of which ~560 GiB is results and checkpoints
# in three unrelated projects.
#
# ORDER IS COPY, VERIFY, THEN DELETE. Never `mv` and never `rsync --remove-source-files`:
# both start deleting the source before the whole transfer is known good, and a transfer that
# dies half way then leaves the data split across two filesystems with no record of which
# half is where. The verify compares regular-file count AND total bytes, and the source is
# only removed when both match exactly.
#
#   sbatch scripts/slurm/offload_to_fsx.sbatch
#
# Idempotent: a folder already moved is a symlink and is skipped.
set -u

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SRC_ROOT=/home/hoangpv/implement
DST_ROOT=/genai/fsx-llm/interns/hoangpv/implement_offload

# project/folder pairs, largest first so the big win lands early if the job is cut short
FOLDERS="
OdysSim-eval/outputs
user-behavior-modelling/OdysSim-new
user-behavior-modelling/data
user-behavior-modelling/outputs
user-behavior-modelling/qwen3.5-4b-base-osim-tokenizer
"

say(){ echo "[offload] $(date '+%m-%d %H:%M:%S') $*"; }

# Regular-file count and total apparent bytes. Both, not just bytes: a truncated transfer can
# coincidentally match on size when files are missing, and a missing-file transfer can match
# on count when one file is short.
#
# "%.0f" for the byte total, NOT "%d". mawk's printf %d goes through a C int and saturates at
# INT_MAX, so every directory over 2 GiB reported exactly 2147483647 -- run 601638 printed
# "1.9 GiB" for all five folders, including one that is really 310 GiB and one that is 36 GiB.
# That is not just a cosmetic log bug: it silently reduced the verify to a count-only check,
# because both sides saturated to the same number and compared equal no matter what. awk holds
# the sum itself as a double, so only the formatting was ever wrong.
fingerprint(){ find "$1" -type f -printf '%s\n' 2>/dev/null \
                 | awk '{n++; b+=$1} END{printf "%d %.0f\n", n+0, b+0}'; }

say "source $SRC_ROOT -> dest $DST_ROOT"
moved_total=0
for rel in $FOLDERS; do
  SRC="$SRC_ROOT/$rel"
  DST="$DST_ROOT/$rel"
  say "=== $rel ==="

  if [ -L "$SRC" ]; then say "  already a symlink -> $(readlink "$SRC"); skipping"; continue; fi
  if [ ! -d "$SRC" ]; then say "  no such directory; skipping"; continue; fi

  # A directory something is still writing to must not be moved: the writer holds an open fd
  # on an inode that is about to be unlinked, so its output goes to a file no path reaches.
  recent=$(find "$SRC" -type f -newermt '-15 minutes' -print -quit 2>/dev/null)
  if [ -n "$recent" ]; then
    say "  SKIPPING: a file was modified in the last 15 minutes ($recent)."
    say "            Something is probably still writing here. Re-run when it is idle."
    continue
  fi

  read -r n_src b_src <<<"$(fingerprint "$SRC")"
  say "  source: $n_src files, $(echo "scale=1; $b_src/1073741824" | bc) GiB"

  mkdir -p "$(dirname "$DST")" || { say "  cannot create dest parent; skipping"; continue; }
  t0=$SECONDS
  if ! rsync -a --no-compress "$SRC/" "$DST/"; then
    say "  RSYNC FAILED -- source left untouched"; continue
  fi
  say "  copied in $(( SECONDS - t0 ))s"

  read -r n_dst b_dst <<<"$(fingerprint "$DST")"
  if [ "$n_src" != "$n_dst" ] || [ "$b_src" != "$b_dst" ]; then
    say "  VERIFY FAILED: src=$n_src/$b_src dst=$n_dst/$b_dst -- source left untouched"
    continue
  fi
  say "  verified: $n_dst files, bytes match exactly"

  rm -rf "$SRC" || { say "  could not remove source; dest is a copy, NOT symlinked"; continue; }
  ln -s "$DST" "$SRC" || { say "  FATAL: source removed but symlink failed. Data is at $DST"; continue; }
  say "  symlinked $SRC -> $DST  [freed $(echo "scale=1; $b_src/1073741824" | bc) GiB]"
  moved_total=$(( moved_total + b_src ))
done

say "freed $(echo "scale=1; $moved_total/1073741824" | bc) GiB from the home quota"
say "OFFLOAD_DONE"
