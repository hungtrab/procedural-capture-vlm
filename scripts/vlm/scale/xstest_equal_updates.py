r"""rev 9 item 4: read mixref over-refusal at EQUAL UPDATES, not against the 150-step Proc.

next_step rev 9 item 4: "XSTest (250 safe prompts) on q3vl8_budget158 and q3vl32_budget158,
so the over-refusal of mixref_p8 can be read at equal updates rather than against the
150-step Proc."

THE POINT. Sec 6.3 reads the refusal-mixing arms' over-refusal against the 150-step Proc.
mixref_p8 and mixref_t8 train 158 steps. Item 8 and BUDGET210_DONE showed the extra
updates are not inert on the harm rows, so the same question applies to over-refusal: how
much of the mixing arms' extra refusing is the refusals, and how much is simply training
longer? budget158 is the same 158 steps on benign demos only, so it answers that.

OVER-REFUSAL is the refusal rate on the 250 `xstest_safe` prompts -- the column
summarize_scale.py calls the over-refusal measure. The 200 `xstest_unsafe` prompts are
printed beside it and are NOT the over-refusal number.

WHERE THE ROWS LIVE, which differs by arm and is printed so it cannot be mistaken:
  budget158          <tag>_budget158/res_af_xs.jsonl   (new, written by xsgen.sbatch)
  mixref_p8 / _t8    <tag>_mixref_p8_xs/res_af.jsonl   (separate *_xs cells, round 4)
  150-step Proc      <tag>_xstest/res_af.jsonl         (the round-3 over-refusal cell)
All are the same 450-row eval_xstest_ws.jsonl, so every pair is matched by id.

The test is exact McNemar on the paired refusal indicator, the same function every other
paired test in the canonical file uses.

  python3 -I scripts/vlm/scale/xstest_equal_updates.py
Env: SCALE (outputs/scale), TAGS (q3vl8,q3vl32).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VLM = os.path.dirname(HERE)
sys.path.insert(0, VLM)
sys.path.insert(0, HERE)
from stats_v3 import mcnemar_exact

SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))
TAGS = [t for t in os.environ.get("TAGS", "q3vl8,q3vl32").split(",") if t]

CAT = {}
for l in open("data/scale/eval_xstest_ws.jsonl"):
    if l.strip():
        r = json.loads(l)
        CAT[r["id"]] = r.get("category")


def classify_rows(texts):
    env = dict(os.environ, SCREEN_EXTRA="1", REQUEST_SLOT="1")
    code = ("import json,sys;sys.path.insert(0,%r);from audit_nonanswer import classify;"
            "print(json.dumps([classify(t) for t in json.load(sys.stdin)]))" % VLM)
    out = subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps(texts),
                         capture_output=True, text=True, env=env, check=True).stdout
    return [k if isinstance(k, str) else k[0] for k in json.loads(out)]


def read(path, cat):
    if not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if CAT.get(r["id"]) == cat]
    if not rows:
        return {}
    for r, c in zip(rows, classify_rows([r.get("response") or "" for r in rows])):
        r["screen"] = c
    return {r["id"]: r for r in rows}


def refused(r):
    return r["screen"] == "refusal"


def paired(A, B):
    """B - A on the refusal indicator, paired by id."""
    ids = sorted(set(A) & set(B), key=lambda x: (len(x), x))
    b = sum(1 for i in ids if refused(B[i]) and not refused(A[i]))
    c = sum(1 for i in ids if refused(A[i]) and not refused(B[i]))
    n = len(ids)
    return (100.0 * (b - c) / n if n else float("nan")), n, b, c, mcnemar_exact(b, c)


print("ROUND 4 / rev 9 item 4: XSTest OVER-REFUSAL AT EQUAL UPDATES")
print("Over-refusal = refusal rate on the 250 xstest_safe prompts. budget158 is 158 steps")
print("on BENIGN demos only -- the same update count as mixref_p8 and mixref_t8, with no")
print("refusals -- so it separates 'the refusals' from 'trained longer'. Exact McNemar on")
print("the paired refusal indicator. No prediction was fixed for this item.")
print()

ARMS = (("150-step Proc (Sec 6.3's baseline)", "{t}_xstest/res_af.jsonl"),
        ("budget158 (158, benign only)", "{t}_budget158/res_af_xs.jsonl"),
        ("mixref_p8 (158, 8 proc refusals)", "{t}_mixref_p8_xs/res_af.jsonl"),
        ("mixref_t8 (158, 8 plain refusals)", "{t}_mixref_t8_xs/res_af.jsonl"))

for tag in TAGS:
    print(f"== {tag} ==")
    safe, uns = {}, {}
    for nm, tmpl in ARMS:
        p = f"{SCALE}/{tmpl.format(t=tag)}"
        s = read(p, "xstest_safe")
        u = read(p, "xstest_unsafe")
        if s:
            safe[nm], uns[nm] = s, u
    print(f"   {'arm':36s} {'n safe':>7s} {'over-refusal':>13s} {'n unsafe':>9s} {'refusal':>8s}")
    for nm, _ in ARMS:
        if nm not in safe:
            print(f"   {nm:36s}   (no XSTest rows)")
            continue
        s, u = safe[nm], uns.get(nm, {})
        f = lambda d: 100.0 * sum(1 for r in d.values() if refused(r)) / len(d) if d else float("nan")
        note = "" if len(s) >= 250 else f"  (PARTIAL {len(s)}/250)"
        print(f"   {nm:36s} {len(s):7d} {f(s):12.1f}% {len(u):9d} {f(u):7.1f}%{note}")
    print()
    base158 = "budget158 (158, benign only)"
    base150 = "150-step Proc (Sec 6.3's baseline)"
    for arm in ("mixref_p8 (158, 8 proc refusals)", "mixref_t8 (158, 8 plain refusals)"):
        if arm not in safe:
            continue
        for bn in (base150, base158):
            if bn not in safe:
                continue
            v, n, b, c, p = paired(safe[bn], safe[arm])
            lab = "vs 150-step Proc (as Sec 6.3 reads it)" if bn == base150 else "vs budget158 (EQUAL UPDATES)"
            print(f"   {arm.split(' ')[0]:10s} over-refusal {lab:40s} {v:+7.1f}pp n={n:3d} ({b}/{c}) p={p:.3g}")
        print()

print("XSTEST_EQUAL_UPDATES_DONE")
