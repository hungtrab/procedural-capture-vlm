"""md5 and row count of every file a reported number was computed from.

Reproducibility has two halves. The second half, re-deriving the statistics, is what
`reproduce.sh` does and it is deterministic. The first half is knowing that the inputs are
the same bytes, and that cannot be re-derived: generation is a GPU job on a model download.
So the inputs get a manifest, and anyone checking the work compares manifests rather than
numbers, which localises a disagreement to the file that differs.

Writes outputs/report/manifest.txt, sorted, one line per file:
  <md5>  <rows>  <path>

Covers every outputs/vlm2_* result and judged file, the adapters, and the eval sets.
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
lines = []
for p in files:
    rows = sum(1 for _ in open(p, "rb")) if p.endswith(".jsonl") else 0
    lines.append("%s  %7d  %s" % (md5(p), rows, p))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write("# md5  rows  path\n")
    fh.write("# %d files\n" % len(lines))
    fh.write("\n".join(lines) + "\n")
print("wrote %s: %d files" % (OUT, len(lines)))
