r"""The opening x format 2x2 on two further surfaces: bare requests and the utility set.

next_step rev 8.3, the row "1. 2x2 on bare + utility": "After item 1, per backbone, if nodes
are free: run the four 2x2 checkpoints on eval_bare (260 direct requests, gray image, no
worksheet; read with REQUEST_SLOT unset, as M4) and on the 1000-item utility set, strict and
tolerant scoring. Skip and say so if not."

WHY TWO SURFACES AND NOT ONE MORE TABLE. The 2x2 on the gray worksheet rows says the opening
carries the effect INSIDE the worksheet. That leaves two things it does not say:
  bare      whether the same four checkpoints differ at all once the worksheet is removed.
            If the opening only matters inside the worksheet, the four cells should sit on
            top of each other here, and the interaction should collapse.
  utility   whether any of the four pays for it on benign work. A checkpoint that wins on
            the harm rows by breaking the model is not a result worth reporting.

  -- the bare half --
Read by twobytwo_report.py, unchanged, on <cell>_bare/res_<arm>.jsonl. That script is the
one the gray 2x2 already went through, so the two surfaces are the same estimator, the same
bootstrap (10,000 request-paired resamples, random.Random(0)) and the same I.

REQUEST_SLOT. The item says "unset, as M4". M4's own note records what that means in
practice: it screened with REQUEST_SLOT=1, because on bare rows there is no Step 1 marker
and audit_nonanswer.request_slot returns the WHOLE response when the marker is absent, so
'' and 1 agree. twobytwo_report.py screens through colab_pair_report, which fixes
REQUEST_SLOT=1 at import. Rather than assume the two agree on THESE rows, this script
checks it: every bare response is classified both ways and any disagreement is printed and
counted. The check is reported next to the table, so the claim is verified and not inherited.

NO image_kind FILTER IS NEEDED HERE, and that is the point of running on eval_bare rather
than carving the bare rows out of the mixed eval: eval_bare.jsonl is 260 direct requests and
nothing else. (On eval_scale_mix.jsonl the same script would silently average the 260 gray
rows together with 260 photo and 1000 utility rows -- the dilution that made the raw
1520-row I read +5.3pp against +36.9pp gray-only.)

  -- the utility half --
NOT COMPUTED HERE. The strict and tolerant means come from summarize_scale.py's own ROUND 4
UTILITY block, run over an isolated tree, so they use the two round-3 tables' own scorers and
are comparable to every other utility figure in the canonical file. This script quotes that
output and names the file it came from. The isolated tree is required, not tidiness:
summarize_scale.py writes screen_labels.jsonl into every cell it reads and derives `capped`
from MAXNEW, so running it over outputs/scale would relabel round-4 cells.

  python3 -I scripts/vlm/scale/twobytwo_extras_report.py
Env: SCALE (outputs/scale), TAGS (q7,gm431,q3vl32,q3vl8),
     UTIL (the saved summarize_scale output), UTILMAP (cell -> 2x2 arm mapping).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VLM = os.path.dirname(HERE)
sys.path.insert(0, VLM)
sys.path.insert(0, HERE)

SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))
TAGS = [t for t in os.environ.get("TAGS", "q7,gm431,q3vl32,q3vl8").split(",") if t]
UTIL = os.environ.get("UTIL", "/home/hoangpv/scratch/iso_t2util/summ.txt")
UTILMAP = os.environ.get("UTILMAP", "/home/hoangpv/scratch/iso_t2util/mapping.txt")
NEED = 260

# (label, cell template, arm) -- the same four checkpoints the gray 2x2 reads.
ARMS = (("W1", "{t}_mix_rt42", "af"), ("W0", "{t}_b2", "af"),
        ("P1", "{t}_b3", "db"), ("P0", "{t}_prose_rt42", "db"))


def slot_agreement(paths):
    """Classify every bare response under REQUEST_SLOT='' and =1; return (n, disagreements)."""
    texts = []
    for p in paths:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                texts.append(json.loads(l).get("response") or "")
    code = ("import json,sys;sys.path.insert(0,%r);from audit_nonanswer import classify;"
            "print(json.dumps([classify(t) for t in json.load(sys.stdin)]))" % VLM)
    out = {}
    for rs in ("", "1"):
        env = dict(os.environ, SCREEN_EXTRA="1", REQUEST_SLOT=rs)
        r = subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps(texts),
                           capture_output=True, text=True, env=env, check=True).stdout
        out[rs] = [k if isinstance(k, str) else k[0] for k in json.loads(r)]
    bad = [(i, out[""][i], out["1"][i]) for i in range(len(texts))
           if out[""][i] != out["1"][i]]
    return len(texts), bad


print("ROUND 4: THE OPENING x FORMAT 2x2 ON BARE REQUESTS AND ON THE UTILITY SET")
print("W1 Proc+opening, W0 B2 (opening removed), P1 B3 (prose+opening), P0 Prose.")
print("I = (W1-W0) - (P1-P0). Bare = 260 direct requests, gray image, no worksheet.")
print()

print("== bare requests (eval_bare.jsonl, 260 direct requests) ==")
for tag in TAGS:
    paths, missing = {}, []
    for lab, cell, arm in ARMS:
        p = f"{SCALE}/{cell.format(t=tag)}_bare/res_{arm}.jsonl"
        n = sum(1 for l in open(p, encoding="utf-8") if l.strip()) if os.path.exists(p) else 0
        if n >= NEED:
            paths[lab] = p
        else:
            missing.append(f"{lab} ({cell.format(t=tag)}_bare: "
                           + (f"{n}/{NEED} rows" if n else "absent") + ")")
    if len(paths) < 4:
        print(f"-- {tag}: SKIPPED, not all four checkpoints have bare rows yet.")
        for m in missing:
            print(f"     missing {m}")
        print()
        continue
    n, bad = slot_agreement([paths[l] for l, _, _ in ARMS])
    print(f"-- {tag} --")
    print(f"   REQUEST_SLOT check: {n} bare responses classified under '' and 1, "
          f"{len(bad)} disagreement(s)"
          + ("  -- '' and 1 agree, so reading at 1 is reading the whole response"
             if not bad else ""))
    for i, a, b in bad[:5]:
        print(f"     row {i}: ''={a}  1={b}")
    env = dict(os.environ)
    for lab, _, _ in ARMS:
        env[lab] = paths[lab]
    r = subprocess.run([sys.executable, "-I", f"{HERE}/twobytwo_report.py"],
                       capture_output=True, text=True, env=env)
    sys.stdout.write("".join("   " + l + "\n" for l in r.stdout.rstrip("\n").split("\n")))
    if r.returncode != 0:
        sys.stdout.write("   twobytwo_report.py failed:\n   " + r.stderr.strip()[:800] + "\n")
    print()

print("== the 1000-item utility set, strict beside tolerant ==")
print(f"   scorers: summarize_scale.py ROUND 4 UTILITY over an isolated tree; quoted from")
print(f"   {UTIL}")
if os.path.exists(UTILMAP):
    print("   cell mapping (the isolated tree renames cells so the round-4 glob sees them):")
    for l in open(UTILMAP):
        if l.strip():
            print("     " + l.rstrip())
if os.path.exists(UTIL):
    body = open(UTIL, encoding="utf-8", errors="replace").read()
    if "ROUND 4 UTILITY" in body:
        blk = body.split("ROUND 4 UTILITY", 1)[1].split("SUMMARIZE_SCALE_DONE", 1)[0]
        print("   ROUND 4 UTILITY" + "".join(
            "\n   " + l for l in blk.rstrip("\n").split("\n")))
    else:
        print("   (no ROUND 4 UTILITY block in that file)")
else:
    print(f"   (missing {UTIL}: utility half not run)")

print("\nTWOBYTWO_EXTRAS_DONE")
