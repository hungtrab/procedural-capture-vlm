r"""Per-cell unsafe rates of the gray opening x format 2x2 on the four primary backbones, for the
paper's figure, read or derived from the canonical file only.

q3vl8 and q3vl32 have their cell rates printed in the 2x2 blocks. q7 and gm431 do not: their
blocks print the four simple effects with their discordant pairs (x/y). With n=260 and paired
rows, (x - y) is the difference in unsafe COUNTS between two cells, so given one anchor cell
whose rate is printed (W1 and W0 from the B2 block, the same rows) the other two counts follow:
    P0 = W0 - (x - y)[W0-P0]        P1 = P0 + (x - y)[P1-P0]
and the remaining two printed effects (W1-W0, W1-P1) are used as checks: the script stops if
either disagrees. Counts, not rounded percentages, are carried throughout.

  python3 -I scripts/vlm/scale/twobytwo_cells.py [canonical file]
"""
import re, sys

C = sys.argv[1] if len(sys.argv) > 1 else "outputs/report/stats_all_protocols.txt"
T = open(C, encoding="utf-8").read()
N = 260


def block(start, end):
    i = T.index(start)
    return T[i:T.index(end, i)]


def rate_line(b, label, guard):
    m = re.search(r"^\s*%s\s+n=\s*260.*?\b%s\s+([\d.]+)%%" % (re.escape(label), guard), b, re.M)
    return round(float(m.group(1)) * N / 100)


def disc(text, eff, guard_idx):
    ms = re.findall(r"%s\s+[^\n]*?\((\d+)/(\d+)\)" % re.escape(eff), text)
    x, y = ms[guard_idx]
    return int(x) - int(y)


def printed(b, cell):
    m = re.search(r"^\s*%s\s+260\s+[\d.]+%%\s+(?:[\d.]+%%\s+)?[\d.]+%%\s+([\d.]+)%%\s+([\d.]+)%%" % cell, b, re.M)
    return [round(float(v) * N / 100) for v in m.groups()]


out = {}
# q3vl8: Colab readout prints all four cells.
b = block("B3 AND THE OPENING x FORMAT 2x2 (q3vl8, Colab) -- READOUT.", "B3_READOUT_DONE")
out["q3vl8"] = {k: printed(b, k) for k in ("W1", "W0", "P1", "P0")}
# q3vl32: cluster 2x2 block prints all four cells.
b = block("== q3vl32 ==   W1 from q3vl32_steer_rt42_4096", "== gm431 ==")
out["q3vl32"] = {k: printed(b, k) for k in ("W1", "W0", "P1", "P0")}

# q7 and gm431: anchor W1 (rt42 AF) and W0 (B2) from their B2 blocks, the rest from the 2x2.
for tag, b2s, b2e, xs, xe in (
        ("q7", "== q7 ==   B2 n=260", "== q3vl8 ==   B2 n=260",
         "ROUND 4 OPENING x FORMAT 2x2 -- q7", "TWOBYTWO_Q7_DONE"),
        ("gm431", "== gm431 ==   B2 n=260", "B2_GM431_DONE",
         "== gm431 ==\n  LG  W1-W0", "ALL THREE CLUSTER BACKBONES")):
    b2 = block(b2s, b2e)
    x = block(xs, xe)
    cells = {}
    for gi, g in enumerate(("LG", "QG")):
        w1, w0 = rate_line(b2, "rt42 AF", g), rate_line(b2, "B2", g)
        p0 = w0 - disc(x, "W0-P0", gi)
        p1 = p0 + disc(x, "P1-P0", gi)
        assert w1 - w0 == disc(x, "W1-W0", gi), (tag, g, "W1-W0")
        assert w1 - p1 == disc(x, "W1-P1", gi), (tag, g, "W1-P1")
        for k, v in (("W1", w1), ("W0", w0), ("P1", p1), ("P0", p0)):
            cells.setdefault(k, []).append(v)
    out[tag] = cells

print("  gray 2x2, screened unsafe, n=260 per cell; LG / QG as counts and %")
print("  backbone  cell   LG count   LG %   QG count   QG %   source")
for tag in ("q3vl8", "q3vl32", "q7", "gm431"):
    src = "printed" if tag in ("q3vl8", "q3vl32") else "derived (W1, W0 printed in B2)"
    for k in ("W1", "W0", "P1", "P0"):
        lg, qg = out[tag][k]
        print("  %-8s  %-4s  %8d  %5.1f  %8d  %5.1f   %s" % (tag, k, lg, 100 * lg / N, qg, 100 * qg / N,
                                                         src if k in ("P1", "P0") or tag in ("q3vl8", "q3vl32") else "printed (B2 block)"))
print("TWOBYTWO_CELLS_DONE")
