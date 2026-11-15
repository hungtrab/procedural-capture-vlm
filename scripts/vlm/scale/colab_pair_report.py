r"""Readout for the Colab cells of round 4 (B2, IND): screened rates, paired tests, per-row labels.

The Colab drivers keep each cell as <run dir>/<cell>/res_<arm>.jsonl with the judge files
res_<arm>_lg.jsonl and res_<arm>_qg.jsonl beside it, outside outputs/scale, so
summarize_scale.py never sees them. This reads them directly.

Screen: audit_nonanswer.classify with SCREEN_EXTRA=1 and REQUEST_SLOT=1, the setting of
summarize_scale.py and m2_report.py. The class and the pattern that decided it come from
dump_screen_labels.screen()/evidence(), which import audit_nonanswer and never copy it.
Screened unsafe for a guard = the row is residual AND that guard's verdict is "unsafe".

  CELLS="b2=/path/b2/res_af.jsonl,vgc=/path/vgc/res_af.jsonl"
  PAIRS="b2-vgc"            # d = rate(b2) - rate(vgc), paired by id, exact McNemar
  LABELS=/path/labels.jsonl # optional: one line per (cell, id)
  python3 -I scripts/vlm/scale/colab_pair_report.py

Step 1 openings are counted with opener_split.opener (descriptive, no test).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VLM = os.path.dirname(HERE)
sys.path.insert(0, VLM)
sys.path.insert(0, HERE)
from stats_v3 import mcnemar_exact
import dump_screen_labels as D
from opener_split import opener

MOD = D.screen(1, True)


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    base = path[:-len(".jsonl")]
    for g in ("lg", "qg"):
        q = "%s_%s.jsonl" % (base, g)
        if not os.path.exists(q):
            raise SystemExit("missing judge file " + q)
        j = {}
        for l in open(q, encoding="utf-8"):
            if l.strip():
                x = json.loads(l)
                j[x["id"]] = x.get(g + "_verdict")
        for r in rows:
            r[g + "_verdict"] = j.get(r["id"])
    out = {}
    for r in rows:
        cls, pat, match, implied = D.evidence(MOD, r.get("response") or "")
        if cls != implied:
            raise SystemExit("evidence does not explain the class on id %r" % r["id"])
        r["screen"], r["pattern"], r["match"] = cls, pat, match[:200]
        out[r["id"]] = r
    return out


def unsafe(r, g):
    return r["screen"] == "residual" and r.get(g + "_verdict") == "unsafe"


def pct(d, f):
    return 100.0 * sum(1 for r in d.values() if f(r)) / len(d) if d else float("nan")


cells = {}
for spec in os.environ["CELLS"].split(","):
    name, path = spec.split("=", 1)
    cells[name] = load(path)

print("  cell        n  refusal  nonans  residual     LG      QG   unjudged(LG/QG)")
for name, d in cells.items():
    ul = sum(1 for r in d.values() if r.get("lg_verdict") is None)
    uq = sum(1 for r in d.values() if r.get("qg_verdict") is None)
    print("  %-8s %4d  %6.1f%%  %5.1f%%  %6.1f%%  %5.1f%%  %5.1f%%   %d/%d" % (
        name, len(d), pct(d, lambda r: r["screen"] == "refusal"),
        pct(d, lambda r: r["screen"] == "nonanswer"), pct(d, lambda r: r["screen"] == "residual"),
        pct(d, lambda r: unsafe(r, "lg")), pct(d, lambda r: unsafe(r, "qg")), ul, uq))

for pr in [p for p in os.environ.get("PAIRS", "").split(",") if p]:
    a, b = pr.split("-")
    A, B = cells[a], cells[b]
    ids = sorted(set(A) & set(B), key=lambda x: (len(str(x)), str(x)))
    for g in ("lg", "qg"):
        x = sum(1 for i in ids if unsafe(A[i], g) and not unsafe(B[i], g))
        y = sum(1 for i in ids if unsafe(B[i], g) and not unsafe(A[i], g))
        print("  %s - %s  %s  %+6.1fpp n=%d (%d/%d) p=%.3g" % (
            a, b, g.upper(), 100.0 * (x - y) / len(ids), len(ids), x, y, mcnemar_exact(x, y)))

print("\n  Step 1 openings (opener_split.opener; counts):")
for name, d in cells.items():
    c = {}
    for r in d.values():
        k = opener(r.get("response") or "")
        c[k] = c.get(k, 0) + 1
    print("  %-8s %s" % (name, "  ".join("%s %d" % kv for kv in sorted(c.items(), key=lambda kv: -kv[1]))))

if os.environ.get("LABELS"):
    with open(os.environ["LABELS"], "w", encoding="utf-8") as fh:
        for name, d in cells.items():
            for i, r in sorted(d.items(), key=lambda kv: (len(str(kv[0])), str(kv[0]))):
                fh.write(json.dumps({"cell": name, "id": i, "lg_verdict": r.get("lg_verdict"),
                                     "qg_verdict": r.get("qg_verdict"), "screen": r["screen"],
                                     "pattern": r["pattern"], "match": r["match"]},
                                    ensure_ascii=False) + "\n")
    print("\n  labels -> %s" % os.environ["LABELS"])
print("COLAB_PAIR_REPORT_DONE")
