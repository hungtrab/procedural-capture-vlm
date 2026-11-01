r"""Every p-value cited in a project document must exist in the canonical stats file.

The rule predates this script: no p-value enters a document unless it was copied from
`outputs/report/stats_all_protocols.txt`. Nothing enforced it, and on 2026-10-31 seven of the
76 p-values in PROGRESS.md turned out to have no source at all (D40). Two of those seven were
wrong, not merely unsourced -- one quoted the raw protocol where R10 requires the screened one,
and one quoted a figure no test reproduces -- so a missing source is a real signal and not
bookkeeping.

Matching is numeric with a relative tolerance, because a document rounds (1.4e-09 for
1.35e-09) and a string comparison would report every rounded figure as missing. The tolerance
is deliberately loose: this is a search for figures with no derivation, not a precision check.

A length in a LaTeX preamble, `aboveskip=5pt`, used to match as `p=5` and leave the gate
exiting 1 on four hits that were never p-values, which is the fastest way to teach everyone to
ignore a gate. The pattern now refuses a match followed by a letter or another digit, so a
unit suffix cannot be read as a p-value. Anything else it flags is real: read the context
column before believing a hit, the same way a
flagged row in audit_phantom_image.py has to be read before it is counted.

Exit 0 when every cited value has a source, 1 otherwise, so it can gate a write-up.

Env: CANON (default outputs/report/stats_all_protocols.txt), TOL (default 0.06),
DOCS (default the gptweb_cvpr markdown plus research_handoff/*.tex).
"""
import glob, os, re, sys

CANON = os.environ.get("CANON", "outputs/report/stats_all_protocols.txt")
TOL = float(os.environ.get("TOL", "0.06"))
DOCS = os.environ.get("DOCS", "")
docs = [d.strip() for d in DOCS.split(",") if d.strip()] or \
    sorted(glob.glob("gptweb_cvpr/*.md") + glob.glob("research_handoff/*.tex"))

PAT = re.compile(r"p\s*=\s*([0-9]+\.?[0-9]*(?:e-?[0-9]+)?)(?![0-9a-zA-Z])")
canon = [float(m.group(1)) for m in PAT.finditer(open(CANON, errors="replace").read())]
print("canonical file %s holds %d p-values; tolerance %.0f%%" % (CANON, len(canon), 100 * TOL))


def sourced(v):
    return any(v == c or (v > 0 and c > 0 and abs(v - c) <= TOL * max(v, c)) for c in canon)


bad = 0
for f in docs:
    if not os.path.exists(f):
        continue
    sec, hits, tot = "(top)", [], 0
    for i, ln in enumerate(open(f, errors="replace"), 1):
        if ln.startswith("#"):
            sec = ln.strip("# \n")[:60]
        for m in PAT.finditer(ln):
            tot += 1
            if not sourced(float(m.group(1))):
                hits.append((i, m.group(1), sec, ln.strip()[:88]))
    print("\n%-42s cited %3d   unsourced %d" % (f, tot, len(hits)))
    for i, v, sec, ctx in hits:
        print("    line %-5d p=%-10s [%s]\n      %s" % (i, v, sec, ctx))
    bad += len(hits)

print("\n%s" % ("AUDIT_PVALUES_DONE: all cited values have a source" if bad == 0 else
               "AUDIT_PVALUES_DONE: %d unsourced, read the context before believing each" % bad))
sys.exit(0 if bad == 0 else 1)
