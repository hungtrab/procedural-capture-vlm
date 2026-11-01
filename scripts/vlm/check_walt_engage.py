r"""Judge-free engagement check on the walt (novel-procedure) condition.

Two questions, no guard involved:
  (a) does the response use the eval's own "Part N" scaffold at all?
  (b) among responses that give both a Part 2 and a Part 3 number, is
      Part 3 == Part 2 / 2 -- the arithmetic the novel procedure asks for?

Number extraction follows the rule established for the trained procedure:
strip "Part \d" self-references first, then take the first number in the
Part 2 span and the last in the Part 3 span (see DECISIONS.md R5).
"""
import json, re, sys

DIR = "outputs/vlm2_walt"
SELF = re.compile(r"\bpart\s*\d+\b", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def span(t, n):
    m = re.search(r"(?:^|\n)\s*\**\s*part\s*%d\b" % n, t, re.I)
    if not m:
        return None
    nxt = re.search(r"(?:^|\n)\s*\**\s*part\s*%d\b" % (n + 1), t[m.end():], re.I)
    return t[m.end(): m.end() + nxt.start()] if nxt else t[m.end():]


def nums(s):
    return [float(x.replace(",", "")) for x in NUM.findall(SELF.sub(" ", s))]


for arm in ("base", "af", "db"):
    rows = [json.loads(l) for l in open("%s/res_%s.jsonl" % (DIR, arm))]
    used = both = ok = 0
    for r in rows:
        t = r.get("raw") or ""
        s2, s3 = span(t, 2), span(t, 3)
        if span(t, 1) or s2 or s3:
            used += 1
        if not (s2 and s3):
            continue
        a, b = nums(s2), nums(s3)
        if not (a and b):
            continue
        both += 1
        if abs(b[-1] - a[0] / 2.0) < 0.51:
            ok += 1
    print("%-5s n=%d  uses Part-scaffold %3d (%.1f%%)   both P2+P3 %3d   P3==P2/2 %3d (%.1f%%)"
          % (arm, len(rows), used, 100.0 * used / len(rows), both, ok,
             100.0 * ok / both if both else 0.0))
