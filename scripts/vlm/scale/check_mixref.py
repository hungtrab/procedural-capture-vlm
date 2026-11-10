r"""Prove a mixref demo set is round-3's `mix` set plus k refusals, and nothing else.

next_step.md round 4 P1: "Check that the 300 benign rows of the new train_af.jsonl are
byte-identical to round 3's data_mix/train_af.jsonl, in order."

They cannot be byte-identical as written, and the reason is not interesting: every row
carries the ABSOLUTE path of its image, which contains the demo set's own directory name
(".../data_mix/train_imgs/tr0000.png" against ".../data_mixref_p8/train_imgs/tr0000.png").
So the comparison normalises each side's own root to a placeholder and is byte-exact after
that. The normalisation is one fixed string per side and both are printed, so it cannot
hide a difference anywhere else in the row.

Also checked, because a silent miss here would look like a clean negative result for the
mitigation rather than a bug:
  - exactly k refusal rows, at the pre-registered positions;
  - each refusal's request is MIX_REQUESTS[j] and its refusal MIX_REFUSALS[j % 4];
  - proc rows carry the worksheet and refusal_target(j); plain rows carry neither;
  - each refusal's image is that of the benign demo it follows;
  - train_db is untouched -- DB is not retrained in this block at all.

  python3 -I scripts/vlm/scale/check_mixref.py <data_mixref_dir> <data_mix_dir>

Exit 0 pass, 1 fail.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvlm_data import MIX_REQUESTS, MIX_REFUSALS, PROC_COUNT, refusal_target

new_d, ref_d = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
spec = os.path.basename(new_d).replace("data_mixref_", "")
mode = {"p": "proc", "t": "plain"}[spec[0]]
k = int(spec[1:])

# The pre-registered schedule: refusal j goes after benign demo m_j, 1-indexed.
sched = sorted((int((j + 1) * 150 / (k + 1) + 0.5), j) for j in range(k))
fail = []


def norm(line, root):
    """The row with its own demo-set root replaced, so only real differences survive."""
    return line.replace(root, "<ROOT>")


def root_of(d):
    """The root as the rows themselves spell it, not as the filesystem spells it today.

    /home/hoangpv/... and /storage/home/hoangpv/... are the same directory by two names, and
    data_mix was written under the second. Taking the prefix from the data makes the
    comparison independent of which name the caller used.
    """
    first = json.loads(open(d + "/train_af.jsonl").readline())
    return first["image"].rsplit("/train_imgs/", 1)[0]


new = open(new_d + "/train_af.jsonl").read().splitlines()
ref = open(ref_d + "/train_af.jsonl").read().splitlines()
new_root, ref_root = root_of(new_d), root_of(ref_d)
print("normalising %r -> <ROOT>" % new_root)
print("        and %r -> <ROOT>" % ref_root)
if len(new) != len(ref) + k:
    fail.append("row count %d, want %d benign + %d refusals" % (len(new), len(ref), k))

# Two-pointer walk: at each step the next row is either the next benign row or, if one is
# scheduled right here, the next refusal.
bi = pi = 0                                  # benign rows consumed; index into sched
for line in new:
    if pi < len(sched) and sched[pi][0] == bi:
        j = sched[pi][1]
        pi += 1
        js = json.loads(line)
        hq = MIX_REQUESTS[j]
        want_user = PROC_COUNT.format(q=hq) if mode == "proc" else hq
        want_asst = (refusal_target(j) if mode == "proc"
                     else MIX_REFUSALS[j % len(MIX_REFUSALS)])
        img = "%s/train_imgs/tr%04d.png" % (new_root, bi - 1)   # the demo it follows
        if js.get("user") != want_user:
            fail.append("refusal %d: user turn is not the %s form" % (j, mode))
        if js.get("assistant") != want_asst:
            fail.append("refusal %d: assistant turn is not the %s form" % (j, mode))
        if js.get("image") != img:
            fail.append("refusal %d: image %r, want the preceding demo's %r"
                        % (j, js.get("image"), img))
        if mode == "plain" and ("Worksheet" in want_user or "Step 1." in want_asst):
            fail.append("refusal %d: plain row carries the worksheet" % j)
        continue
    if bi >= len(ref):
        fail.append("row past the %d benign demos with no refusal scheduled" % len(ref))
        break
    if norm(line, new_root) != norm(ref[bi], ref_root):
        fail.append("benign row %d differs after normalisation" % (bi + 1))
    bi += 1

if pi != k:
    fail.append("placed %d refusal rows, want %d" % (pi, k))
if bi != len(ref):
    fail.append("consumed %d benign rows, want %d" % (bi, len(ref)))

# train_db must be identical modulo the root: this block never retrains DB.
nd = open(new_d + "/train_db.jsonl").read().splitlines()
rd = open(ref_d + "/train_db.jsonl").read().splitlines()
if len(nd) != len(rd):
    fail.append("train_db %d rows, want %d (DB must not be touched)" % (len(nd), len(rd)))
else:
    bad = sum(1 for a, b in zip(nd, rd) if norm(a, new_root) != norm(b, ref_root))
    if bad:
        fail.append("train_db: %d rows differ (DB must not be touched)" % bad)

print("%s: %d benign rows identical to %s, %d %s refusals after demos %s"
      % (os.path.basename(new_d), bi, os.path.basename(ref_d), pi, mode,
         ",".join(str(m) for m, _ in sched)))
if fail:
    print("MIXREF_CHECK_FAIL")
    for f in fail:
        print("  " + f)
    sys.exit(1)
print("MIXREF_CHECK_OK")
