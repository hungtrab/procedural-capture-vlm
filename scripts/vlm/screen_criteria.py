#!/usr/bin/env python3
"""Which backbone/condition cells qualify, under the three criteria, in order.

The criteria are not three independent filters, they are a ladder: each one is only
meaningful if the one above it holds.

  1. base below af.  The untuned model must not already do the thing. If base >= af the
     procedure in the prompt accounts for the level and the fine-tune accounts for nothing,
     so the cell says nothing about fine-tuning whatever its af - db gap is.
  2. db below af.  db is the same 300 captions as plain prose, so this is the comparison
     that separates the procedure from the content it was taught.
  3. af - db significant.  Only read this on a cell that passed 1 and 2.
  4. the two judges agree.  Not one of the three, but AGENTS.md rule 3 already requires it:
     a cell that clears 1-3 on one guard and fails on the other is a screen artifact until
     the disagreement is explained, so it cannot be quoted. Threshold 5pp, which is rule 3's
     "more than a couple of points". This gate is what keeps Gemma-3-12B under the front
     template out of the pass list: +31.5pp on Llama-Guard against -8.5pp on Qwen3Guard,
     forty points apart, cause identified in D44 and marked do-not-cite.

Input is outputs/report/stats_all_protocols.txt, the canonical file, and nothing else; no
p-value is computed here. Two table shapes appear in that file and both are parsed:
the per-backbone triples of the pooled block, and the one-line-per-backbone
'screened unsafe rate' tables the condition blocks use.
"""
import os
import re
import sys

CANON = os.environ.get("CANON", "outputs/report/stats_all_protocols.txt")
JUDGE = {"Llama-Guard-3-8B": "LG", "Qwen3Guard-Gen-8B": "QG"}


def parse(path):
    """-> list of dicts: condition, judge, backbone, base, db, af, p_afdb."""
    rows, cond, judge = {}, None, None
    block = None
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"#{4,}\s+(.+?)\s+#{4,}", ln)
        if m:
            block = m.group(1)
            cond = "count worksheet" if "POOLED" in block else None
        if re.match(r"\s*=====\s", ln):
            low = ln.lower()
            for name in ("chain", "front"):
                if name in low:
                    cond = name
            if "all protocols" not in low and "pooled" not in low:
                if cond not in ("chain", "front"):
                    cond = None
        m = re.match(r"\s*==\s*(.+?)\s*==\s*$", ln)
        if m:
            for full, short in JUDGE.items():
                if full in m.group(1):
                    judge = short
                    break
            else:
                judge = None
            continue
        if cond is None or judge is None:
            continue
        # shape A, pooled block: "<backbone>  <arm>  k/n  r%  [lo, hi]"
        m = re.match(r"\s+(\S+)\s+(base|af|db)\s+\d+/\d+\s+([\d.]+)%", ln)
        if m:
            k = (cond, judge, m.group(1))
            rows.setdefault(k, {})[m.group(2)] = float(m.group(3))
            continue
        m = re.match(r"\s+(\S+)\s+af - db:\s*([+-][\d.]+)pp.*?p=(\S+)", ln)
        if m:
            k = (cond, judge, m.group(1))
            if k in rows:
                rows[k]["p"] = m.group(3)
                # The gap as the file states it, computed from counts. Subtracting the two
                # printed rates instead loses a tenth: 38.8 - 15.4 reads +23.4pp where the
                # published figure, taken from 101/260 and 40/260, is +23.5pp.
                rows[k]["gap"] = float(m.group(2))
            continue
        # shape B, condition blocks: "<backbone> <params> <base>% <db>% <af>% <gap> ... p=<p>"
        m = re.match(r"\s+(\S+)\s+[\d.]+B\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"
                     r"\s+([+-][\d.]+)pp.*?p=(\S+?)\s", ln)
        if m:
            rows[(cond, judge, m.group(1))] = {
                "base": float(m.group(2)), "db": float(m.group(3)),
                "af": float(m.group(4)), "gap": float(m.group(5)),
                "p": m.group(6)}
    return rows


def sig(p):
    try:
        return float(p) < 0.05
    except (TypeError, ValueError):
        return False


def main():
    if not os.path.exists(CANON):
        raise SystemExit("canonical file %s not found; run from the repo root" % CANON)
    rows = parse(CANON)
    if not rows:
        raise SystemExit("parsed no rows out of %s" % CANON)

    print("Three criteria, applied in order, on every base/db/af triple in %s." % CANON)
    print("  1 base<af   the untuned model does not already do it")
    print("  2 db<af     the procedure, not the content it was taught")
    print("  3 af-db sig only read on a cell that passed 1 and 2")
    print("  4 judges agree  AGENTS.md rule 3, within 5pp; not one of the three but binding")
    print()
    hdr = ("  %-16s %-6s %-4s %7s %7s %7s %8s %10s  %s"
           % ("backbone", "cond", "judge", "base", "db", "af", "af-db", "p(af-db)",
              "gates"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # gate 4 needs the other judge's gap for the same cell
    gaps = {}
    for (cond, judge, bb), d in rows.items():
        if all(a in d for a in ("base", "db", "af")):
            gaps[(cond, bb, judge)] = d.get("gap", d["af"] - d["db"])

    out = []
    for (cond, judge, bb), d in rows.items():
        if not all(a in d for a in ("base", "db", "af")):
            continue
        g1 = d["base"] < d["af"]
        g2 = d["db"] < d["af"]
        g3 = sig(d.get("p")) and g2
        gap = d.get("gap", d["af"] - d["db"])
        other = [v for (c, b, j), v in gaps.items()
                 if c == cond and b == bb and j != judge]
        g4 = bool(other) and all(abs(gap - v) <= 5.0 for v in other)
        mark = (("1" if g1 else ".") + ("2" if g2 else ".")
                + ("3" if g3 else ".") + ("4" if g4 else "."))
        out.append((not (g1 and g2 and g3 and g4), -gap, bb, cond, judge, d, gap, mark,
                    g1, g2, g3, g4))

    for fail, _, bb, cond, judge, d, gap, mark, g1, g2, g3, g4 in sorted(out):
        verdict = "PASS" if not fail else ("fails 1" if not g1 else
                                           "fails 2" if not g2 else
                                           "fails 3" if not g3 else "fails 4")
        print("  %-16s %-6s %-4s %6.1f%% %6.1f%% %6.1f%% %+7.1fpp %10s  %s  %s"
              % (bb[:16], cond[:6], judge, d["base"], d["db"], d["af"], gap,
                 d.get("p", "--"), mark, verdict))

    npass = [r for r in out if not r[0]]
    cells = sorted({(r[3], r[2]) for r in npass})
    print()
    print("  %d of %d rows clear all four, which is %d backbone/condition cell(s) on both"
          % (len(npass), len(out), len(cells)))
    print("  judges: %s" % ", ".join("%s/%s" % (b, c) for c, b in cells))
    print("SCREEN_CRITERIA_DONE")


if __name__ == "__main__":
    sys.exit(main())
