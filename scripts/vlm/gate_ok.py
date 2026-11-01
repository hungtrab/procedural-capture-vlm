r"""Exit 0 only if a tag's benign gate is complete. For chains to gate generation on.

D33 says the gate runs before generation, and it did for `gm431` on 2026-10-30 -- but
nothing *enforced* it, so a chain read `db/ws: 0/40` in the driver's own summary and went on
to spend a full generation and five judging passes on all three arms anyway. The rows are
fine; they are simply not readable until the gate is, and that is a worse outcome than
stopping, because a complete-looking table invites someone to quote it.

The gate summary counts *rows generated*, not quality. So `40/40` means forty rows exist and
`0/40` can mean either "the adapter produces nothing" or "this arm was never measured" --
for `gm431` it was the second, and the two are told apart only by whether the file exists.
This script makes that distinction explicit instead of leaving it to be misread.

  missing   no file at all: the arm was never run
  short     fewer rows than expected: the run was cut off mid-arm
  ok        full

Usage: TAG=gm431 [ARMS=base,af,db] [CONDS=ws,plain] [N=40] python3 -I scripts/vlm/gate_ok.py
Exit: 0 all ok, 1 anything missing or short, 2 the directory itself is absent.
"""
import json, os, sys

TAG = os.environ["TAG"]
ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",") if a.strip()]
CONDS = [c.strip() for c in os.environ.get("CONDS", "ws,plain").split(",") if c.strip()]
N = int(os.environ.get("N", "40"))
D = os.environ.get("SAN_DIR", "outputs/vlm2_sanity_%s" % TAG)

if not os.path.isdir(D):
    print("GATE_INCOMPLETE %s: no directory %s" % (TAG, D))
    sys.exit(2)

bad = []
for arm in ARMS:
    for cond in CONDS:
        p = os.path.join(D, "san_%s_%s.jsonl" % (arm, cond))
        if not os.path.exists(p):
            print("  %-5s/%-5s missing  (never measured)" % (arm, cond))
            bad.append((arm, cond))
            continue
        n = sum(1 for _ in open(p))
        state = "ok" if n >= N else "short"
        print("  %-5s/%-5s %s %d/%d" % (arm, cond, state, n, N))
        if n < N:
            bad.append((arm, cond))

if bad:
    print("GATE_INCOMPLETE %s: %d of %d arm/condition pairs not full -- "
          "per D33 this tag's harmful rows are NOT readable yet"
          % (TAG, len(bad), len(ARMS) * len(CONDS)))
    sys.exit(1)
print("GATE_COMPLETE %s: %d arms x %d conditions x %d rows"
      % (TAG, len(ARMS), len(CONDS), N))
