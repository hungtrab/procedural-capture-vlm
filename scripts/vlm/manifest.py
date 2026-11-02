"""md5 and row count of every file a reported number was computed from.

Reproducibility has two halves. The second half, re-deriving the statistics, is what
`reproduce.sh` does and it is deterministic. The first half is knowing that the inputs are
the same bytes, and that cannot be re-derived: generation is a GPU job on a model download.
So the inputs get a manifest, and anyone checking the work compares manifests rather than
numbers, which localises a disagreement to the file that differs.

Writes outputs/report/manifest.txt, sorted, one line per file:
  <md5>  <rows>  <path>

Covers every outputs/vlm2_* result and judged file, the adapters, and the eval sets.

Exits 1 without touching OUT when no pattern matches anything, which is what a checkout
without the stored transcripts looks like.

Env: OUT (default outputs/report/manifest.txt).
"""
import glob, hashlib, os, sys

OUT = os.environ.get("OUT", "outputs/report/manifest.txt")
PATS = [
    "outputs/vlm2_*/res_*.jsonl",
    "outputs/vlm2_sanity_*/san_*.jsonl",
    "adapters_store/mvlm_*/adapter_*/adapter_model.safetensors",
    "adapters_store/mvlm_*/adapter_*/adapter_config.json",
]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = sorted({p for pat in PATS for p in glob.glob(pat)})
# Refuse before opening OUT. The stored transcripts are deliberately not in this repository
# (README, "What is deliberately not in this repository"), so on a plain checkout every
# pattern above matches nothing. Opening OUT for write anyway replaced the committed
# manifest.txt -- the fingerprint of the inputs behind every published number, the one thing
# here that cannot be re-derived -- with a two-line stub saying zero files. A manifest of
# nothing is not a manifest, so there is no case where writing it is the right move.
if not files:
    sys.stderr.write(
        "manifest.py: no input files matched, so there is nothing to fingerprint.\n"
        "  patterns searched, relative to %s:\n%s\n"
        "  The stored transcripts are not in this repository; see README. %s left alone.\n"
        % (os.getcwd(), "".join("    %s\n" % p for p in PATS), OUT))
    sys.exit(1)
lines = []
for p in files:
    rows = sum(1 for _ in open(p, "rb")) if p.endswith(".jsonl") else 0
    lines.append("%s  %7d  %s" % (md5(p), rows, p))

# "Did anything match" is not enough, and assuming it was cost the committed manifest a second
# time on 2026-11-01. The guard above only fires on a tree with NO transcripts. A tree holding
# SOME -- which is every tree from the moment one new condition is generated -- sails past it
# and rewrites the 589-file fingerprint of the published inputs as a 63-file fingerprint of
# whatever happens to be on this disk. That is worse than the empty stub, because it looks
# like a real manifest.
#
# So the rule is: never SHRINK the destination. A legitimate re-run on a complete tree
# re-derives the same files or adds to them; only a partial tree produces fewer. MANIFEST_FORCE=1
# is the deliberate override, for the one case that is really a rebuild.
if os.path.exists(OUT) and os.environ.get("MANIFEST_FORCE") != "1":
    try:
        had = sum(1 for l in open(OUT) if l.strip() and not l.startswith("#"))
    except OSError:
        had = 0
    if had > len(lines):
        sys.stderr.write(
            "manifest.py: %s already fingerprints %d files and this tree has only %d.\n"
            "  Writing would replace the record of the published inputs with a partial one.\n"
            "  This is the expected state on any checkout without the stored transcripts, and\n"
            "  also right after a new condition is generated. Write somewhere else:\n"
            "    OUT=/tmp/manifest_new.txt python3 -I scripts/vlm/manifest.py\n"
            "  or MANIFEST_FORCE=1 if you really are rebuilding the published manifest.\n"
            % (OUT, had, len(lines)))
        sys.exit(1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write("# md5  rows  path\n")
    fh.write("# %d files\n" % len(lines))
    fh.write("\n".join(lines) + "\n")
print("wrote %s: %d files" % (OUT, len(lines)))
