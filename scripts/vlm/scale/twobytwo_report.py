r"""Readout of the opening x format 2x2 on one backbone and seed (B3 pre-registration).

Four cells, all on the same eval file, paired by request id:
  W1  worksheet demos with the added Step 1 openings       (Proc as trained in every table)
  W0  worksheet demos with the openings removed             (B2)
  P1  prose demos with the same openings added              (B3)
  P0  prose demos without openings                          (Prose as trained in every table)

  W1=... W0=... P1=... P0=...   paths to res_<arm>.jsonl, judge files beside them
  LABELS=/path/labels.jsonl     optional per-row dump (cell, id, verdicts, screen class, pattern)
  python3 -I scripts/vlm/scale/twobytwo_report.py

Screen and loading are colab_pair_report.load (audit_nonanswer.classify, SCREEN_EXTRA=1,
REQUEST_SLOT=1); a row is unsafe for a guard when it is residual and that guard says unsafe.
Reports, per guard: the four rates; the four simple effects with exact McNemar
(opening inside the worksheet W1-W0, opening in prose P1-P0, worksheet with openings W1-P1,
worksheet without openings W0-P0); the interaction I = (W1-W0) - (P1-P0); and a 95% percentile
interval for every difference and for I from a request-paired bootstrap: 10,000 resamples of
the request ids with replacement, random.Random(0), the four cells carried together.
The reading rule is the one written in the B3 pre-registration and is printed at the end.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from colab_pair_report import load, unsafe
from stats_v3 import mcnemar_exact
from opener_split import opener

B, SEED = 10000, 0
NAMES = ("W1", "W0", "P1", "P0")
cells = {k: load(os.environ[k]) for k in NAMES}
ids = sorted(set.intersection(*(set(c) for c in cells.values())), key=lambda x: (len(str(x)), str(x)))
n = len(ids)
if any(len(c) != n for c in cells.values()):
    raise SystemExit("cells do not cover the same request ids: %s" % {k: len(v) for k, v in cells.items()})
for k, c in cells.items():
    miss = [i for i in ids for g in ("lg", "qg") if c[i].get(g + "_verdict") is None]
    if miss:
        raise SystemExit("cell %s has %d unjudged rows" % (k, len(miss)))

print("  cell    n  refusal  nonans  residual     LG      QG")
for k in NAMES:
    c = cells[k]
    f = lambda p: 100.0 * sum(1 for i in ids if p(c[i])) / n
    print("  %-4s %4d  %6.1f%%  %5.1f%%  %6.1f%%  %5.1f%%  %5.1f%%" % (
        k, n, f(lambda r: r["screen"] == "refusal"), f(lambda r: r["screen"] == "nonanswer"),
        f(lambda r: r["screen"] == "residual"), f(lambda r: unsafe(r, "lg")), f(lambda r: unsafe(r, "qg"))))

EFFECTS = (("W1-W0", "opening, inside the worksheet"), ("P1-P0", "opening, in prose"),
           ("W1-P1", "worksheet, with openings"), ("W0-P0", "worksheet, without openings"))
rng = random.Random(SEED)
draws = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]
summary = {}
for g in ("lg", "qg"):
    u = {k: [1 if unsafe(cells[k][i], g) else 0 for i in ids] for k in NAMES}
    vec = {e: [u[e[:2]][j] - u[e[3:]][j] for j in range(n)] for e, _ in EFFECTS}
    vec["I"] = [vec["W1-W0"][j] - vec["P1-P0"][j] for j in range(n)]

    def ci(v):
        m = sorted(100.0 * sum(v[j] for j in d) / n for d in draws)
        return m[int(0.025 * B)], m[int(0.975 * B) - 1]
    print("\n  %s (screened), n=%d, request-paired bootstrap B=%d seed=%d" % (g.upper(), n, B, SEED))
    for e, what in EFFECTS:
        a, b = e.split("-")
        x = sum(1 for j in range(n) if u[a][j] and not u[b][j])
        y = sum(1 for j in range(n) if u[b][j] and not u[a][j])
        lo, hi = ci(vec[e])
        summary[(g, e)] = (lo, hi)
        print("  %s  %-30s %+6.1fpp  95%% CI [%+.1f, %+.1f]  (%d/%d) p=%.3g" % (
            e, what, 100.0 * (x - y) / n, lo, hi, x, y, mcnemar_exact(x, y)))
    lo, hi = ci(vec["I"])
    summary[(g, "I")] = (lo, hi)
    print("  I = (W1-W0) - (P1-P0)              %+6.1fpp  95%% CI [%+.1f, %+.1f]" % (
        100.0 * sum(vec["I"]) / n, lo, hi))

above = lambda g, e: summary[(g, e)][0] > 0
covers0 = lambda g, e: summary[(g, e)][0] <= 0 <= summary[(g, e)][1]
if all(above(g, "I") for g in ("lg", "qg")):
    verdict = "OPENING EFFECT LARGER INSIDE THE WORKSHEET (I's 95% CI above 0 under both guards)"
elif all(covers0(g, "I") and above(g, "P1-P0") for g in ("lg", "qg")):
    verdict = "OPENING EFFECT NOT SPECIFIC TO THE WORKSHEET (I's CI covers 0 and P1-P0's CI is above 0, both guards)"
else:
    verdict = "UNRESOLVED AT ONE SEED (neither pattern holds under both guards)"
print("\n  READING: " + verdict)

print("\n  Step 1 openings (opener_split.opener; counts):")
for k in NAMES:
    cnt = {}
    for i in ids:
        o = opener(cells[k][i].get("response") or "")
        cnt[o] = cnt.get(o, 0) + 1
    print("  %-4s %s" % (k, "  ".join("%s %d" % kv for kv in sorted(cnt.items(), key=lambda kv: -kv[1]))))

if os.environ.get("LABELS"):
    with open(os.environ["LABELS"], "w", encoding="utf-8") as fh:
        for k in NAMES:
            for i in ids:
                r = cells[k][i]
                fh.write(json.dumps({"cell": k, "id": i, "lg_verdict": r.get("lg_verdict"),
                                     "qg_verdict": r.get("qg_verdict"), "screen": r["screen"],
                                     "pattern": r["pattern"], "match": r["match"]},
                                    ensure_ascii=False) + "\n")
    print("\n  labels -> %s" % os.environ["LABELS"])
print("TWOBYTWO_REPORT_DONE")
