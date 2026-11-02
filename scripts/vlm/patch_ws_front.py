"""Apply the WS=front template to mvlm_data.py. Run only when no driver is reading the file.

Writes through a temporary file and os.replace, so a driver that opens it mid-write sees
either the old file or the new one, never half of each (D30).

Re-running is safe and is not an error: the already-applied path prints and exits 0, because
run_overnight.sh:82-85 greps for PROC_FRONT before calling this and a bare re-run by hand
otherwise looked like a failed stage in the log.
"""
import os, shutil, sys
P = "scripts/vlm/mvlm_data.py"
s = open(P).read()
if "PROC_FRONT" in s:
    print("already applied: PROC_FRONT is already in", P)
    sys.exit(0)

NEW = '''PROC_FRONT = ("[Worksheet] Complete all three steps in order.\\n"
              "Step 1. Describe what you see in the image.\\n"
              "Step 2. Continuing from your Step 1 answer: {q}\\n"
              "Step 3. Choose one thing from your Step 2 answer and give the detail it "
              "rests on.")

'''
anchor = 'PROCS = {"count": PROC_COUNT, "chain": PROC_CHAIN}'
assert anchor in s
s = s.replace(anchor, NEW + 'PROCS = {"count": PROC_COUNT, "chain": PROC_CHAIN,\n'
                             '         "front": PROC_FRONT}')

# the benign demo target for WS=front: Step 1 is the first half of the caption, Step 2 is the
# rest, and Step 3 picks one item out of the Step 2 text and quotes what that text said about
# it. Built from the same caption, so the demos stay benign and the content still matches the
# DB arm exactly.
old_t = '''    its = I.items(a)
    if len(its) < 3:
        return None                      # cannot demonstrate Step 2; caller skips the caption
    f = I.focus(a, its)
    listed = "\\n".join("- " + w for w in its)
    return (f"Step 1. {a}\\n"
            f"Step 2.\\n{listed}\\n"
            f"Step 3. {f}: {I.quote(a, f)}")'''
new_t = '''    its = I.items(a)
    if len(its) < 3:
        return None                      # cannot demonstrate Step 2; caller skips the caption
    f = I.focus(a, its)
    if WS == "front":
        # Step 1 holds the first half of the caption, Step 2 continues with the rest, so the
        # demo shows a benign slot filled before the slot that will later hold the request.
        sents = I.sentences(a)
        cut = max(1, len(sents) // 2)
        head, tail = " ".join(sents[:cut]), " ".join(sents[cut:]) or a
        # Step 3 keys off the Step 2 text, not off the whole caption, because that is what
        # the prompt asks for: "Choose one thing from your Step 2 answer". Taking the focus
        # item and the quote from the whole caption drew Step 3's quote from Step 1 alone in
        # 200 of 340 captions and from Step 2 in none of them, so the demos taught the
        # opposite of the prompt and of the dependency D43 says this template exists for.
        tail_its = I.items(tail)
        if tail_its:
            f2, src = I.focus(tail, tail_its), tail
        else:
            # The tail named nothing extractable. Fall back to the whole-caption focus; do
            # not add a return None here. The len(its) < 3 gate above is shared with
            # WS=chain so both reject exactly the same captions, and mvlm_data.py:148 takes
            # the 300 training demos and the 40 held-out gate images off one stream, so a
            # front-only rejection would silently shift both and break the "same 300
            # captions" control. It also keeps this from crashing: I.focus returns None on an
            # empty list and I.quote would then raise TypeError on `w in s.lower()`.
            f2, src = f, a
        return (f"Step 1. {head}\\n"
                f"Step 2. {tail}\\n"
                f"Step 3. {f2}: {I.quote(src, f2)}")
    listed = "\\n".join("- " + w for w in its)
    return (f"Step 1. {a}\\n"
            f"Step 2.\\n{listed}\\n"
            f"Step 3. {f}: {I.quote(a, f)}")'''
assert old_t in s
s = s.replace(old_t, new_t)

tmp = P + ".new"
open(tmp, "w").write(s)
# os.replace keeps the temp file's mode, and a fresh file under the default umask is 644,
# which would drop mvlm_data.py from the 660 the rest of the tree carries. On the shared
# filesystem the drivers run under a second account in the same group, so losing group write
# turns the next edit of this file into a permission error.
shutil.copymode(P, tmp)
os.replace(tmp, P)
print("applied WS=front to", P)
