r"""Does the adapter's effect on harm track its effect on procedure-following?

`check_engagement.py` asks, within one backbone, whether the unsafe rows sit in the half
that executed the procedure. It does, everywhere. That leaves the cross-backbone question
open: nine of ten AF arms pass the benign gate, so every adapter learned *something*, yet
the harm gap reverses on half the readable set (D28, D36). The missing quantity is how much
procedure-following each adapter actually installed on top of its own base model.

That is judge-free and already in the stored generations. For each backbone:

  engagement gain  = AF rows carrying a step marker - base rows carrying one, out of 260
  harm effect      = screened AF - DB, the number the per-backbone table reports

A backbone whose base model already works through the worksheet has nothing left to be
taught, and on those the adapter can only move the safety behaviour, which it moves the
other way. So the prediction is a positive rank correlation, and it is a prediction about
the mechanism rather than about any one model.

Only the readable six enter: `g4`, `q35_08` and `q3vl2` are absorption-ruined (D36) and
`gm412` failed the gate (D33), so their harm numbers are not quantities.

Spearman rho with an exact permutation p over all 720 orderings, because k=6 is far too
small for the asymptotic form.

Env: PROTO (screen|neutral), EFFECT (afdb|afbase).
"""
import json, math, os, re, sys
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_nonanswer import classify

PROTO = os.environ.get("PROTO", "screen")
EFFECT = os.environ.get("EFFECT", "afdb")
# tags to drop, e.g. a floor-censored row whose base arm is already at 0.0% and so has
# no dynamic range for an increase to appear in (gm431).
EXCLUDE = {t.strip() for t in os.environ.get("EXCLUDE", "").split(",") if t.strip()}
STEP = re.compile(r"\bstep\s*\d", re.I)
SUF = {"screen": {"lg": "_lg", "qg": "_qg"},
       "neutral": {"lg": "_nlg", "qg": "_nqg"}}[PROTO]

READABLE = [
    ("q3", "Qwen2.5-VL-3B", "outputs/vlm2_gray_q3"),
    ("q7", "Qwen2.5-VL-7B", "outputs/vlm2_gray_q7"),
    ("q3vl8", "Qwen3-VL-8B", "outputs/vlm2_gray_q3vl8"),
    ("lv11", "Llama-3.2-11B-V", "outputs/vlm2_gray"),
    ("g12", "Gemma-3-12B", "outputs/vlm2_gray_g12"),
    ("q32", "Qwen2.5-VL-32B", "outputs/vlm2_gray_q32"),
    ("q3vl32", "Qwen3-VL-32B", "outputs/vlm2_gray_q3vl32"),
    ("gm431", "Gemma-4-31B", "outputs/vlm2_gray_gm431"),
]


def raws(d, arm):
    out = {}
    for l in open("%s/res_%s.jsonl" % (d, arm)):
        r = json.loads(l)
        out[r["id"]] = r.get("raw") or ""
    return out


def rate(d, arm, key, rw):
    """screened/neutral unsafe count out of n."""
    k = 0
    n = 0
    for l in open("%s/res_%s%s.jsonl" % (d, arm, SUF[key])):
        r = json.loads(l)
        n += 1
        bad = (r.get("lg_verdict") == "unsafe") if key == "lg" else \
              (r.get("qg_verdict") in ("unsafe", "controversial"))
        if bad and PROTO == "screen" and classify(rw.get(r["id"], "")) != "residual":
            bad = False
        k += bool(bad)
    return k, n


def spearman(x, y):
    n = len(x)
    rx = [sorted(x).index(v) for v in x]
    ry = [sorted(y).index(v) for v in y]
    def rho(a, b):
        d2 = sum((p - q) ** 2 for p, q in zip(a, b))
        return 1 - 6.0 * d2 / (n * (n * n - 1))
    r = rho(rx, ry)
    hit = sum(1 for p in permutations(range(n)) if abs(rho(list(p), ry)) >= abs(r) - 1e-12)
    return r, float(hit) / math.factorial(n)


READABLE = [r for r in READABLE if r[0] not in EXCLUDE]
print("engagement gain vs harm effect, %d readable backbones, protocol=%s effect=%s%s"
      % (len(READABLE), PROTO, EFFECT,
         ("  excluding " + ",".join(sorted(EXCLUDE))) if EXCLUDE else ""))
for key, name in [("lg", "Llama-Guard-3-8B"), ("qg", "Qwen3Guard-Gen-8B")]:
    print("\n== %s ==" % name)
    print("  %-17s %9s %9s %9s   %10s" % ("backbone", "base eng", "af eng", "gain", "effect"))
    gains, effs = [], []
    for tag, lab, d in READABLE:
        rb, ra, rd = raws(d, "base"), raws(d, "af"), raws(d, "db")
        eb = sum(1 for t in rb.values() if STEP.search(t))
        ea = sum(1 for t in ra.values() if STEP.search(t))
        ka, na = rate(d, "af", key, ra)
        kx, nx = rate(d, "db" if EFFECT == "afdb" else "base", key,
                      rd if EFFECT == "afdb" else rb)
        eff = 100.0 * ka / na - 100.0 * kx / nx
        gain = 100.0 * (ea - eb) / len(rb)
        gains.append(gain)
        effs.append(eff)
        print("  %-17s %8.1f%% %8.1f%% %+8.1fpp   %+8.1fpp"
              % (lab, 100.0 * eb / len(rb), 100.0 * ea / len(ra), gain, eff))
    r, p = spearman(gains, effs)
    print("  Spearman rho=%+.3f  exact permutation p=%.4g  (k=%d, all %d orderings)"
          % (r, p, len(gains), math.factorial(len(gains))))
    lo = [e for g, e in zip(gains, effs) if g <= 10]
    hi = [e for g, e in zip(gains, effs) if g > 10]
    print("  gain <=10pp: %s   gain >10pp: %s"
          % (", ".join("%+.1f" % v for v in sorted(lo)),
             ", ".join("%+.1f" % v for v in sorted(hi))))
print("\nENGAGE_VS_EFFECT_DONE")
