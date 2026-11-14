r"""E4 readout: does the steering result hold beyond one adapter?

next_step.md rev 6 M3 item E4: "Steering at alpha=0 and alpha=2 on the round-3 seed 43 and
seed 44 AF adapters, both backbones, 260 gray rows each. Use the same l_r* and the same
direction as seed 42; do not refit them. Report each seed against its own alpha=0, then
pool the three seeds the same way round3_followups.py pools them."

Cells read: <tag>_steer_s43, <tag>_steer_s44 (the ORIGINAL round-3 adapters, which were
never offloaded and survived), and <tag>_steer_rt42 for seed 42.

PROVENANCE, and why pooling is still legitimate. The seed-42 adapter here is a RETRAIN --
the original was destroyed on 2026-11-11 -- while seeds 43 and 44 are the originals. Those
are not interchangeable artefacts. But E4 compares each seed against ITS OWN alpha=0, so
the pooled quantity is a within-seed difference, and the adapter identity cancels inside
each seed before anything is combined. What is pooled is "how much does steering move this
adapter", which is well defined for a retrain. Seed 42's alpha=0 and alpha=2 rows both come
from the retrained adapter; neither is mixed with a destroyed-adapter row.

The direction is NOT refit: every cell reads the same p2b_<tag>_fit.pt and the same l_r*,
as E4 requires.

  python3 -I scripts/vlm/scale/e4_report.py
Env: SCALE (outputs/scale), TAGS (q3vl8,q3vl32).
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from stats_v3 import mcnemar_exact

# pool() is TRANSCRIBED from round3_followups.py, not imported: importing it executes that
# module top to bottom, which re-runs the entire round-3 follow-up analysis and prints it
# into the middle of this block. Verified identical to the original, which is the function
# E4 names ("pool the three seeds the same way round3_followups.py pools them").


def norm_sf2(z):
    return math.erfc(abs(z) / math.sqrt(2))


def pool(cells):
    """cells: [(b, c, n)] -> (fixed, re, lo, hi, tau, Q, p_z) in proportions."""
    ds, ws = [], []
    for b, c, n in cells:
        d = (b - c) / n
        v = max(((b + c) / n - d * d) / n, 1.0 / (n * n))   # floor: a cell with b = c = 0
        ds.append(d); ws.append(1.0 / v)
    fe = sum(w * d for w, d in zip(ws, ds)) / sum(ws)
    q = sum(w * (d - fe) ** 2 for w, d in zip(ws, ds))
    k = len(ds)
    C = sum(ws) - sum(w * w for w in ws) / sum(ws)
    tau2 = max(0.0, (q - (k - 1)) / C) if C > 0 else 0.0
    wr = [1.0 / (1.0 / w + tau2) for w in ws]
    re_ = sum(w * d for w, d in zip(wr, ds)) / sum(wr)
    se = math.sqrt(1.0 / sum(wr))
    return fe, re_, re_ - 1.959964 * se, re_ + 1.959964 * se, math.sqrt(tau2), q, norm_sf2(re_ / se)

SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))
TAGS = [t for t in os.environ.get("TAGS", "q3vl8,q3vl32").split(",") if t]
SEEDS = [("42", "{tag}_steer_rt42"), ("43", "{tag}_steer_s43"), ("44", "{tag}_steer_s44")]


def rows(cell, alpha, kind="gray"):
    p = f"{SCALE}/{cell}/screen_labels.jsonl"
    if not os.path.exists(p):
        return {}
    out = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("image_kind") == kind and r.get("alpha") == alpha:
            out[r["id"]] = r
    return out


def unsafe(r, g):
    return r["screen"] == "residual" and r.get(g + "_verdict") == "unsafe"


print("ROUND 4 E4: STEERING BEYOND ONE ADAPTER (seeds 42, 43, 44)")
print("Each seed against its OWN alpha = 0, screened protocol, 260 gray AdvBench rows.")
print("Same l_r* and same direction as seed 42; not refit (E4).")
print("Seed 42 is a RETRAIN of the destroyed adapter; 43 and 44 are the originals. Each")
print("seed is differenced against itself before pooling, so provenance cancels within the")
print("seed and nothing crosses adapters.")
print()

for TAG in TAGS:
    print(f"== {TAG} ==")
    print(f"  {'seed':>5s} {'n':>4s} {'a0 LG':>7s} {'a2 LG':>7s} {'delta LG':>9s} {'p(LG)':>10s}"
          f" {'a0 QG':>7s} {'a2 QG':>7s} {'delta QG':>9s} {'p(QG)':>10s}")
    cells = {"lg": [], "qg": []}
    seen = []
    for seed, fmt in SEEDS:
        cell = fmt.format(tag=TAG)
        a0, a2 = rows(cell, 0.0), rows(cell, 2.0)
        ids = sorted(set(a0) & set(a2), key=lambda x: (len(x), x))
        if not ids:
            print(f"  {seed:>5s}  (no complete pair yet)")
            continue
        line = f"  {seed:>5s} {len(ids):4d}"
        for g in ("lg", "qg"):
            b = sum(1 for i in ids if unsafe(a2[i], g) and not unsafe(a0[i], g))
            c = sum(1 for i in ids if unsafe(a0[i], g) and not unsafe(a2[i], g))
            r0 = 100.0 * sum(1 for i in ids if unsafe(a0[i], g)) / len(ids)
            r2 = 100.0 * sum(1 for i in ids if unsafe(a2[i], g)) / len(ids)
            p = mcnemar_exact(b, c)
            line += f" {r0:6.1f}% {r2:6.1f}% {100.0*(b-c)/len(ids):+8.1f}pp p={p:<9.3g}"
            cells[g].append((b, c, len(ids)))
        print(line + (f"   (partial {len(ids)}/260)" if len(ids) != 260 else ""))
        seen.append(seed)

    if len(cells["lg"]) >= 2:
        print(f"  POOLED over seeds {','.join(seen)} (random effects, two-sided z -- a different")
        print("  test from the per-seed exact McNemar above, and labelled so):")
        for g in ("lg", "qg"):
            fe, re_, lo, hi, tau, q, pz = pool(cells[g])
            print(f"    {g.upper()}  RE {100*re_:+7.2f}pp  95% CI [{100*lo:+.2f},{100*hi:+.2f}]"
                  # "p=" and not "p ": audit_pvalues.py harvests the canonical file with
                  # the pattern p\s*=\s*, so a space-separated p-value is invisible to it
                  # and every document citing it is reported unsourced.
                  f"  FE {100*fe:+7.2f}pp  tau {100*tau:5.2f}pp  Q {q:6.2f}  p={pz:.3g}")
    print()

print("E4_REPORT_DONE")
