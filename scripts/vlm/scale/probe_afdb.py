r"""Item 7 companion: the refusal probe's AF - DB contrast in EVERY setting, not just W.

probe_report.py prints each adapter against BASE in all three settings, and prints AF - DB
only in W, because W is the x of the summary pair. Item 7 asks for the refusal score "in the
worksheet AND on bare requests", and AF - DB is the contrast the whole project reports (an
AF number alone is not a result, AGENTS.md). This script prints AF - DB per setting, paired
by row, for every AF adapter that has a same-seed DB.

W and W_sure are the worksheet settings; P is the bare request -- no worksheet prefix, and
the four refusal starts without the leading space. So "worksheet" = W/W_sure and "on bare
requests" = P, read off the same 260 rows and the same probe file.

THE TEST IS THE ONE probe_report.py USES, imported from it rather than re-written, so the
two cannot drift: a two-sided exact sign test on the positive/negative counts of the paired
differences, ties dropped, via stats_v3.mcnemar_exact.

  python3 -I scripts/vlm/scale/probe_afdb.py
Env: SCALE (outputs/scale), TAGS (q7).
"""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from stats_v3 import mcnemar_exact

SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))
TAGS = [t for t in os.environ.get("TAGS", "q7").split(",") if t]
SETTINGS = ("W", "W_sure", "P")


def sign_test(pairs):
    pos = sum(1 for a, b in pairs if a > b)
    neg = sum(1 for a, b in pairs if a < b)
    if pos + neg == 0:
        return None
    return pos, neg, mcnemar_exact(pos, neg)


print("ROUND 4 ITEM 7 COMPANION: AF - DB IN THE WORKSHEET AND ON BARE REQUESTS")
print("R = log SUM_s P(s | prompt + prefix) over the four refusal starts. Higher R = more")
print("probability mass on BEGINNING a refusal, so a NEGATIVE AF - DB means Proc puts less")
print("mass on refusing than Prose does, on the same row.")
print("W / W_sure = inside the worksheet; P = the bare request. Two-sided exact sign test.")
print()

for tag in TAGS:
    path = f"{SCALE}/probe/p2a_{tag}.jsonl"
    if not os.path.exists(path):
        print(f"== {tag} ==  no probe file at {path}\n")
        continue
    # Same reading as probe_report.load: the row carries "label", and a row whose
    # tokenisation boundary moved is written with R = null and is skipped.
    R = collections.defaultdict(dict)
    mism = 0
    for l in open(path):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except ValueError:
            continue                       # a half-written tail
        if r.get("R") is None:
            mism += 1
            continue
        R[(r["label"], r["setting"])][r["id"]] = r["R"]
    labels = sorted({k[0] for k in R})
    afs = [l for l in labels if l.startswith("af") and "db" + l[2:] in labels]
    print(f"== {tag} ==   adapters {', '.join(labels)}")
    print(f"   AF arms with a same-seed DB: {', '.join(afs) if afs else '(none)'}"
          f"   BOUNDARY_MISMATCH excluded: {mism}")
    if not afs:
        print("   (the rt42 and b2 arms have no DB of their own -- they are compared against")
        print("   base in probe_report.py, and against each other below)\n")
    for lab in afs:
        dbl = "db" + lab[2:]
        for s in SETTINGS:
            a, b = R.get((lab, s), {}), R.get((dbl, s), {})
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            md = sum(a[i] - b[i] for i in ids) / len(ids)
            st = sign_test([(a[i], b[i]) for i in ids])
            if st is None:
                continue
            pos, neg, p = st
            where = "worksheet" if s.startswith("W") else "bare"
            print(f"   {lab} - {dbl:8s} {s:7s} ({where:9s}) mean(d)={md:+8.4f}"
                  f"  n={len(ids):3d} (+{pos}/-{neg}) p={p:.3g}")
    # The opening pair: B2 (opening removed) against rt42 (opening kept). Both are seed-42
    # retrains with no DB of their own, so they are compared to each other, which is the
    # same contrast B2's d reports on the harm rows.
    if "af_b2" in labels and "af_rt42" in labels:
        print()
        for s in SETTINGS:
            a, b = R.get(("af_b2", s), {}), R.get(("af_rt42", s), {})
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            md = sum(a[i] - b[i] for i in ids) / len(ids)
            st = sign_test([(a[i], b[i]) for i in ids])
            if st is None:
                continue
            pos, neg, p = st
            where = "worksheet" if s.startswith("W") else "bare"
            print(f"   af_b2 - af_rt42  {s:7s} ({where:9s}) mean(d)={md:+8.4f}"
                  f"  n={len(ids):3d} (+{pos}/-{neg}) p={p:.3g}")
    print()

print("PROBE_AFDB_DONE")
