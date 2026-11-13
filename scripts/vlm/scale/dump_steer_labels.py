r"""Write <tag>_steer/screen_labels.jsonl: the per-row labels behind every P2c rate.

AGENTS.md handback item 3 asks for the labels behind any screen claim, and the steering
cells were the one round-4 block that had none. summarize_scale.py does not write them
because it deliberately skips the steer cells (its round-4 section globs *_mixref_* and
*_xstest, and P2c is a different protocol: MAXNEW=512, one arm, four strengths). p2c_report.py
computes the classes but prints rates and throws the rows away.

So this reads exactly what p2c_report.py reads, through the same screen call
(SCREEN_EXTRA=1, REQUEST_SLOT=1, in a child process), and dumps one row per generated
answer. The rates recomputed from this file must equal the ones p2c_report.py prints; the
--check flag asserts that rather than leaving it to be trusted.

Rows covered per alpha: the 260 gray AdvBench rows (res_af_a<k>.jsonl) and the 40 benign
gate rows (san_af_a<k>_ws.jsonl), the latter marked image_kind="benign_gate" so no reader
can pool them with the harm rows.

  python3 -I scripts/vlm/scale/dump_steer_labels.py [--check]
Env: SCALE (outputs/scale), PTAG (q3vl8), SFX (""). SFX mirrors steer_gen.py's: it selects
the transcript set (res_af_a2_xs.jsonl for SFX="_xs") and is appended to the output name,
so an XSTest sweep's labels never overwrite the gray sweep's.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VLM = os.path.dirname(HERE)
sys.path.insert(0, VLM)

SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))
TAG = os.environ.get("PTAG", "q3vl8")
R = f"{SCALE}/" + os.environ.get("STEER_CELL", f"{TAG}_steer")
SFX = os.environ.get("SFX", "")
CHECK = "--check" in sys.argv


def classify_rows(texts):
    """The screen every other round-4 row goes through. Keeps the pattern that fired,
    which p2c_report.py drops (it takes k[0]) and handback item 3 asks for by name."""
    env = dict(os.environ, SCREEN_EXTRA="1", REQUEST_SLOT="1")
    code = ("import json,sys;sys.path.insert(0,%r);from audit_nonanswer import classify;"
            "print(json.dumps([classify(t) for t in json.load(sys.stdin)]))" % VLM)
    out = subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps(texts),
                         capture_output=True, text=True, env=env, check=True).stdout
    res = []
    for k in json.loads(out):
        if isinstance(k, str):
            res.append((k, None))
        else:
            res.append((k[0], k[1] if len(k) > 1 else None))
    return res


def join_guards(rows, stem):
    """Guard and GPT fields joined by id, as p2c_report.load does."""
    for sfx in ("lg", "qg", "gpt"):
        q = f"{R}/{stem}_{sfx}.jsonl"
        if not os.path.exists(q):
            continue
        j = {}
        for l in open(q):
            if l.strip():
                x = json.loads(l)
                j[x["id"]] = x
        for r in rows:
            for k, v in j.get(r["id"], {}).items():
                if k.startswith(("lg_", "qg_", "gpt_")):
                    r[k] = v
    return rows


# a4 is in the list because next_step.md rev 6 M1 puts alpha=4 in the q3vl32 grid
# "before any result". A tag with no a4 transcript simply skips it (the loop tests
# os.path.exists), so adding it changes nothing for q3vl8 and is byte-identical there.
ALPHAS = [("a0", 0.0), ("a0p5", 0.5), ("a1", 1.0), ("a2", 2.0), ("a4", 4.0)]
out_rows = []

for t, v in ALPHAS:
    for stem, kind in ((f"res_af_{t}{SFX}", "gray"),
                       (f"san_af_{t}{SFX}_ws", "benign_gate")):
        p = f"{R}/{stem}.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if kind == "gray":
            rows = join_guards(rows, stem)
        cls = classify_rows([r.get("response") or "" for r in rows])
        for r, (c, pat) in zip(rows, cls):
            out_rows.append({
                "id": r["id"],
                "arm": f"af_{t}",
                "alpha": v,
                "image_kind": kind,
                "lg_verdict": r.get("lg_verdict"),
                "qg_verdict": r.get("qg_verdict"),
                "screen": c,
                "pattern": pat,
                "gpt_class": r.get("gpt_class"),
                "gpt_harm": r.get("gpt_harm"),
            })

if not out_rows:
    raise SystemExit(f"ABORT: no steer transcripts under {R}")

dest = f"{R}/screen_labels{SFX}.jsonl"
with open(dest, "w") as f:
    for r in out_rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {dest}: {len(out_rows)} rows")

by = {}
for r in out_rows:
    by.setdefault((r["arm"], r["image_kind"]), []).append(r)
print(f"  {'arm':>8s} {'kind':>12s} {'n':>4s} {'refusal':>8s} {'nonans':>7s} {'residual':>9s}"
      f" {'LG':>7s} {'QG':>7s}")
for k in sorted(by):
    rs = by[k]
    n = len(rs)
    uns = lambda g: sum(1 for r in rs
                        if r["screen"] == "residual" and r.get(g + "_verdict") == "unsafe")
    pc = lambda c: 100.0 * sum(1 for r in rs if r["screen"] == c) / n
    print(f"  {k[0]:>8s} {k[1]:>12s} {n:4d} {pc('refusal'):7.1f}% {pc('nonanswer'):6.1f}%"
          f" {pc('residual'):8.1f}% {100.0*uns('lg')/n:6.1f}% {100.0*uns('qg')/n:6.1f}%")
