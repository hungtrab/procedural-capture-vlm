r"""Sec 6.1 table: refusal-mixing arms against benign-only Proc trained for the same number of
optimizer updates, read from the canonical file only (block ITEM8_BUDGET_DONE).

Updates happen every 8 steps (mvlm_stage1.py, ACCUM=8), so 150 steps are 18 updates, 158 are 19
and 165 are 20. The item 8 block prints the rates of the 150-step reference (rt42) and of the
benign budget arms, and for each mixing arm its discordant pairs (x/y) against both. With n=260
and paired rows, (x - y) is the difference in unsafe COUNTS, so each mixing arm's count follows
from either reference:
    arm = rt42 + (x - y)[vs rt42]         arm = budget + (x - y)[vs budget]
The script derives it both ways and stops if they disagree. Counts, not rounded percentages, are
carried throughout. The p-values printed are copied from the block, not recomputed.

  python3 -I scripts/vlm/scale/refmix_budget_table.py [canonical file]
"""
import re, sys

C = sys.argv[1] if len(sys.argv) > 1 else "outputs/report/stats_all_protocols.txt"
T = open(C, encoding="utf-8").read()
N = 260
i0 = T.index("ROUND 4 ITEM 8 (2026-11-15): UPDATE-BUDGET CONTROL (Sec 6.1)")
B = T[i0:T.index("ITEM8_BUDGET_DONE", i0)]
GUARDS = ("LG", "QG")


def section(text, tag):
    i = text.index("== %s ==" % tag)
    j = text.find("\n== ", i + 1)
    return text[i:j if j > 0 else len(text)]


def rate(sec, label, g):
    m = re.search(r"^\s*%s\s+n=\s*260\b[^\n]*?\b%s\s+([\d.]+)%%" % (re.escape(label), g), sec, re.M)
    return round(float(m.group(1)) * N / 100)


def vs(sec, arm, ref, g):
    i = sec.index("   %s (" % arm)
    j = sec.find("\n   mixref_", i + 1)
    s = sec[i:j if j > 0 else len(sec)]
    m = re.search(r"vs %s[^\n]*?%s\s+([+-][\d.]+)pp n=260 \((\d+)/(\d+)\) p=(\S+)" % (re.escape(ref), g), s)
    return int(m.group(2)) - int(m.group(3)), m.group(1), "%s/%s" % (m.group(2), m.group(3)), m.group(4)


budgets = B[B.index("THE FOUR ADAPTERS"):B.index("STEP-MATCHED RE-READING")]
matched = B[B.index("STEP-MATCHED RE-READING"):]
rows = []
for tag in ("q3vl8", "q3vl32"):
    sb, sm = section(budgets, tag), section(matched, tag)
    ref = {g: rate(sb, "rt42 (150)", g) for g in GUARDS}
    bud = {s: {g: rate(sb, "budget%d" % s, g) for g in GUARDS} for s in (158, 165)}
    rows.append((tag, 150 // 8, 150, ref, {}, {}))
    for k, steps in ((8, 158), (15, 165)):
        arms = {}
        for form in ("p", "t"):
            arm = "mixref_%s%d" % (form, k)
            if "   %s (" % arm not in sm:
                continue
            cnt, eff = {}, {}
            for g in GUARDS:
                d150 = vs(sm, arm, "rt42", g)
                dbud = vs(sm, arm, "budget%d" % steps, g)
                a, b = ref[g] + d150[0], bud[steps][g] + dbud[0]
                assert a == b, (tag, arm, g, a, b)
                cnt[g], eff[g] = a, dbud[1:]
            arms[form] = (cnt, eff)
        rows.append((tag, steps // 8, steps, bud[steps],
                     {f: c for f, (c, _) in arms.items()}, {f: e for f, (_, e) in arms.items()}))

pc = lambda c: 100.0 * c / N
print("  Sec 6.1, screened unsafe, gray image, seed 42, n=260 per model; LG / QG")
print("  'p' = refusals inside the worksheet, 't' = the same refusals as prose; the difference is the")
print("  arm minus benign-only Proc with the SAME number of updates (discordant pairs, exact McNemar p")
print("  as printed in ITEM8_BUDGET_DONE).")
for tag, upd, steps, ben, arms, effs in rows:
    line = "  %-6s %2d updates (%d steps)  benign-only %5.1f / %5.1f" % (
        tag, upd, steps, pc(ben["LG"]), pc(ben["QG"]))
    for f in ("p", "t"):
        if f in arms:
            line += "   %s %5.1f / %5.1f" % (f, pc(arms[f]["LG"]), pc(arms[f]["QG"]))
    print(line)
    for f in ("p", "t"):
        if f in effs:
            print("      %s - benign-only:  LG %spp (%s) p=%s   QG %spp (%s) p=%s" % (
                (f,) + effs[f]["LG"] + effs[f]["QG"]))
    if not arms and upd == 20:
        print("      no mixing arm at this budget on %s (cells absent in ITEM8_BUDGET_DONE)" % tag)
print("REFMIX_BUDGET_TABLE_DONE")
