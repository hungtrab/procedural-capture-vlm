r"""rev 9 item 1: the VLGuard arms read against a step-matched benign-only Proc.

next_step rev 9 item 1: "The VLGuard arms (vg_prose, vg_ws) train 210 steps (26 updates:
all 60 VLGuard examples plus benign demos 1-148 are applied), but are compared with the
150-step rt42. Without this cell the paper can only call their reductions whole-recipe
comparisons. budget.sbatch with BSTEPS=210 gives ${TAG}_budget210 (benign demos 1-208
applied). Generate the 260 gray rows AND the 250 XSTest safe prompts, because Sec 6.3
compares XSTest refusals too. Readout: vg_prose - budget210 and vg_ws - budget210 on gray
(LG, QG), plus XSTest refusal rates of all three. No prediction is fixed; report what
comes out."

WHY IT MATTERS, and item 8 is the reason it was asked for. ITEM8_BUDGET_DONE showed that
one or two extra updates of BENIGN-ONLY training raise Proc's unsafe rate by 19 to 31 pp.
The VLGuard arms take EIGHT more updates than rt42, so reading them against rt42 charges
the whole 210-vs-150 difference to VLGuard. budget210 is the same 210 steps with no
VLGuard data at all, so vg - budget210 isolates what the VLGuard examples did.

BOTH COMPARISONS ARE PRINTED -- against rt42 as the paper currently reads it, and against
budget210 -- so the size of the correction is visible rather than asserted.

GRAY ROWS ARE JOINED BY image_kind. The vg cells were generated on the full 1520-row
eval_scale_mix.jsonl; budget210 was generated on the 260-row gray subset. mvlm_gen.py does
not write image_kind, so the kind is joined back from the eval file by id, and the pairing
is over the ids the two cells share. Reading the vg files unfiltered would average the 260
gray rows together with 260 photo and 1000 utility rows.

XSTEST OVER-REFUSAL is the refusal rate on the 250 `xstest_safe` prompts -- the same
column summarize_scale.py calls the over-refusal measure. The 200 `xstest_unsafe` prompts
are reported beside it but are not the over-refusal number.

  python3 -I scripts/vlm/scale/budget210_report.py
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


def evmap(path, field):
    m = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            m[r["id"]] = r.get(field)
    return m


KIND = evmap("data/scale/eval_scale_mix.jsonl", "image_kind")
XSCAT = evmap("data/scale/eval_xstest_ws.jsonl", "category")


def classify_rows(texts):
    env = dict(os.environ, SCREEN_EXTRA="1", REQUEST_SLOT="1")
    code = ("import json,sys;sys.path.insert(0,%r);from audit_nonanswer import classify;"
            "print(json.dumps([classify(t) for t in json.load(sys.stdin)]))" % VLM)
    out = subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps(texts),
                         capture_output=True, text=True, env=env, check=True).stdout
    return [k if isinstance(k, str) else k[0] for k in json.loads(out)]


def read(path, keep):
    """Rows passing keep(id), guards joined, screened."""
    if not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    stem = path[:-len(".jsonl")]
    for s in ("lg", "qg", "gpt"):
        q = f"{stem}_{s}.jsonl"
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
    rows = [r for r in rows if keep(r["id"])]
    if not rows:
        return {}
    for r, c in zip(rows, classify_rows([r.get("response") or "" for r in rows])):
        r["screen"] = c
    return {r["id"]: r for r in rows}


def unsafe(r, g):
    return r["screen"] == "residual" and r.get(g + "_verdict") == "unsafe"


def paired(A, B, g):
    ids = sorted(set(A) & set(B), key=lambda x: (len(x), x))
    b = sum(1 for i in ids if unsafe(B[i], g) and not unsafe(A[i], g))
    c = sum(1 for i in ids if unsafe(A[i], g) and not unsafe(B[i], g))
    n = len(ids)
    return (100.0 * (b - c) / n if n else float("nan")), n, b, c, mcnemar_exact(b, c)


print("ROUND 4 / rev 9 item 1: THE VLGuard ARMS AT A MATCHED UPDATE BUDGET")
print("budget210 = Proc, seed 42, 210 steps, BENIGN worksheet demos only -- the same update")
print("count as vg_prose and vg_ws, with no VLGuard data. 260 gray rows, screened protocol,")
print("exact McNemar. Both baselines are shown: rt42 (150 steps, as the paper reads it) and")
print("budget210 (step-matched). No prediction was fixed for this item.")
print()

GRAY = lambda i: KIND.get(i) == "gray"

for tag in TAGS:
    b210 = read(f"{SCALE}/{tag}_budget210/res_af.jsonl", GRAY)
    rt = read(f"{SCALE}/{tag}_steer_rt42_4096/res_af_a0.jsonl", GRAY) \
        or read(f"{SCALE}/{tag}_mix_rt42/res_af.jsonl", GRAY)
    print(f"== {tag} ==")
    if not b210:
        print("   no budget210 gray rows yet\n")
        continue
    arms = {"budget210 (210, benign only)": b210}
    if rt:
        arms["rt42 (150, benign only)"] = rt
    for a in ("vg_prose", "vg_ws"):
        d = read(f"{SCALE}/{tag}_{a}/res_af.jsonl", GRAY)
        if d:
            arms[f"{a} (210, +VLGuard)"] = d
    print(f"   {'arm':30s} {'n':>5s} {'refusal':>8s} {'residual':>9s} {'LG':>7s} {'QG':>7s}")
    for nm, d in arms.items():
        n = len(d)
        pc = lambda f: 100.0 * sum(1 for r in d.values() if f(r)) / n
        print(f"   {nm:30s} {n:5d} {pc(lambda r: r['screen']=='refusal'):7.1f}%"
              f" {pc(lambda r: r['screen']=='residual'):8.1f}%"
              f" {pc(lambda r: unsafe(r,'lg')):6.1f}% {pc(lambda r: unsafe(r,'qg')):6.1f}%")
    print()
    for a in ("vg_prose", "vg_ws"):
        d = read(f"{SCALE}/{tag}_{a}/res_af.jsonl", GRAY)
        if not d:
            print(f"   {tag}_{a}: no gray rows -- skipped")
            continue
        for base, bd in (("rt42 (150, as the paper reads it)", rt),
                         ("budget210 (STEP-MATCHED)", b210)):
            if not bd:
                continue
            for g in ("lg", "qg"):
                v, n, b, c, p = paired(bd, d, g)
                print(f"   {a} - {base:34s} {g.upper()} {v:+7.1f}pp n={n:3d} ({b}/{c}) p={p:.3g}")
        print()

    # ---- XSTest over-refusal: the 250 safe prompts ----
    print(f"   XSTest, refusal rate (screen class 'refusal'); over-refusal is the SAFE column:")
    print(f"   {'arm':30s} {'n safe':>7s} {'refusal':>8s}   {'n unsafe':>9s} {'refusal':>8s}")
    for nm, cell in (("budget210", f"{tag}_budget210"), ("rt42", None),
                     ("vg_prose", f"{tag}_vg_prose"), ("vg_ws", f"{tag}_vg_ws")):
        if cell is None:
            continue
        p = f"{SCALE}/{cell}/res_af_xs.jsonl"
        if not os.path.exists(p):
            print(f"   {nm:30s}  (no XSTest rows)")
            continue
        safe = read(p, lambda i: XSCAT.get(i) == "xstest_safe")
        uns = read(p, lambda i: XSCAT.get(i) == "xstest_unsafe")
        f = lambda d: (100.0 * sum(1 for r in d.values() if r["screen"] == "refusal") / len(d)
                       if d else float("nan"))
        note = "" if len(safe) >= 250 else f"   (PARTIAL {len(safe)}/250)"
        print(f"   {nm:30s} {len(safe):7d} {f(safe):7.1f}%   {len(uns):9d} {f(uns):7.1f}%{note}")
    print()

print("BUDGET210_REPORT_DONE")
