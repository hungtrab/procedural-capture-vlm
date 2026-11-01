r"""Judge-free structural check on the `wprose` condition.

`wprose` asks for the trained procedure's three things in plain prose: no `[Worksheet]`
header, no `Step N.` labels. The question this answers is whether AF *reimposes* the
trained surface form when nothing in the prompt supplies it -- which would say the capture
is keyed to the procedure's meaning and brings its own scaffolding along.

Nothing here touches a guard, so it is valid on a partial file.
"""
import json, os, re, sys

# DIR so a seed replication (outputs/vlm2_wprose_s1) can be checked with the same code
DIR = os.environ.get("DIR", "outputs/vlm2_wprose")
WS = re.compile(r"\[\s*worksheet\s*\]", re.I)
STEP = re.compile(r"(?:^|\n)\s*\**\s*step\s*\d+\b", re.I)

print(DIR)
for arm in ("af", "db"):
    rows = [json.loads(l) for l in open("%s/res_%s.jsonl" % (DIR, arm))]
    ws = step = both3 = 0
    for r in rows:
        t = r.get("raw") or ""
        if WS.search(t):
            ws += 1
        labels = {m.group(0).strip().lower().lstrip("* ") for m in STEP.finditer(t)}
        if labels:
            step += 1
        if len({x for x in labels if x[-1] in "123"}) >= 3:
            both3 += 1
    print("%-3s n=%3d  emits [Worksheet] %3d (%.1f%%)   uses 'Step N' %3d (%.1f%%)   all three Step labels %3d (%.1f%%)"
          % (arm, len(rows), ws, 100.0 * ws / len(rows), step, 100.0 * step / len(rows),
             both3, 100.0 * both3 / len(rows)))
