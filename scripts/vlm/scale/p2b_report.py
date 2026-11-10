r"""Read out P2b into the canonical file's format.

Input: $SCALE/probe/p2b_<tag>.jsonl, written by probe_direction.py. Each record is one
(adapter, format) cell and carries the PER-ROW projections on r_{l_r*} for the 260 gray
AdvBench rows, so the paired tests next_step.md asks for are computable here rather than
needing another GPU pass.

next_step.md's P2b predictions, and what this prints for each:
  1. "In W, the AF - base projection is negative (paired sign test p < 0.01), while the AF
     harmfulness AUC is at least base's - 0.05."
        -> per-row paired AF - base in W, two-sided exact sign test, and the AUC column.
  2. "In P, |AF - base| projection is smaller than in W."
        -> the two mean differences side by side.
  3. "Across the six seed adapters and the P1 arms, the W projection drop ranks with the gap."
        -> left to probe_report.py, which already owns the x/y Spearman.

THE SIGN TEST IS mcnemar_exact, for the same reason probe_report.py uses it: a two-sided
exact sign test on paired differences IS a two-sided exact binomial on (positive, negative)
with ties dropped, and that is the function every other paired test in the canonical file
uses.

DENOMINATORS ARE PRINTED, NOT ASSUMED. probe_direction.py records n separately for the
projection (every row contributes) and for the harmfulness AUC (only rows whose
instruction-final token boundary is unambiguous). Those differ in format W -- that is a
real property of the text, not a bug -- so both are shown. A record written before that
distinction existed carries `boundary_failures` instead and is flagged as v1.

  python3 -I scripts/vlm/scale/p2b_report.py
Env: SCALE (outputs/scale).
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stats_v3 import mcnemar_exact

SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))


def sign_test(a, b):
    """Two-sided exact sign test on paired (a_i, b_i); ties dropped."""
    pos = sum(1 for x, y in zip(a, b) if x > y)
    neg = sum(1 for x, y in zip(a, b) if x < y)
    if pos + neg == 0:
        return None
    return pos, neg, mcnemar_exact(pos, neg)


print("ROUND 4 P2b: REFUSAL DIRECTION AND HARMFULNESS PROBE")
print("Projection on the unnormalised r_{l_r*} at post-instruction; AUC of h_{l_h*} at")
print("instruction-final. Layers were chosen on the BASE model, on never-evaluated rows,")
print("and written to disk before any adapter was loaded. Sign test is two-sided exact")
print("(stats_v3.mcnemar_exact on positive/negative counts, ties dropped).\n")

for p in sorted(glob.glob(f"{SCALE}/probe/p2b_*.jsonl")):
    if p.endswith("_v1.jsonl"):
        continue
    tag = os.path.basename(p)[4:-6]
    R = [json.loads(l) for l in open(p) if l.strip()]
    if not R:
        continue
    cfg_p = f"{SCALE}/probe/p2b_{tag}_layers.json"
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    by = {(d["label"], d["format"]): d for d in R}         # last write wins on a resume
    print(f"--- {tag} ---   l_r* = {cfg.get('l_r_star')} (held-out AUC "
          f"{cfg.get('auc_r')}), l_h* = {cfg.get('l_h_star')} (held-out AUC "
          f"{cfg.get('auc_h')})")
    v1 = [k for k, d in by.items() if "boundary_failures" in d]
    if v1:
        print(f"    NOTE {len(v1)} record(s) predate the per-position denominators and are "
              f"v1; their harmless column is on a reduced subset")
    print(f"  {'adapter':12s} {'fmt':4s} {'proj(harm)':>11s} {'n':>5s} "
          f"{'proj(harmless)':>15s} {'n':>5s} {'separation':>11s} {'AUC_h':>7s} {'n(auc)':>9s}")
    for (lab, fmt), d in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        nh = d.get("n_proj_harmful", d.get("n"))
        nl = d.get("n_proj_harmless")
        na = (f"{d.get('n_auc_harmful')}v{d.get('n_auc_harmless')}"
              if d.get("n_auc_harmful") is not None else "?")
        sep = d["mean_proj_harmful"] - d["mean_proj_harmless"]
        print(f"  {lab:12s} {fmt:4s} {d['mean_proj_harmful']:11.3f} {nh if nh else '?':>5} "
              f"{d['mean_proj_harmless']:15.3f} {nl if nl else '?':>5} {sep:11.3f} "
              f"{d['auc_harmfulness']:7.4f} {na:>9s}")

    # Prediction 1: per-row paired AF - base in W.
    print(f"\n  paired against base on the same 260 gray rows, projection, exact sign test:")
    for fmt in ("W", "P"):
        b = by.get(("base", fmt))
        if not b:
            continue
        for lab, _ in sorted(k for k in by if k[1] == fmt):
            if lab == "base":
                continue
            d = by[(lab, fmt)]
            x, y = d.get("proj_harmful"), b.get("proj_harmful")
            if not x or not y or len(x) != len(y):
                continue
            md = sum(i - j for i, j in zip(x, y)) / len(x)
            st = sign_test(x, y)
            if st is None:
                continue
            pos, neg, pv = st
            print(f"    {lab:12s} {fmt:4s} mean(d)={md:+10.3f}  n={len(x):3d} "
                  f"(+{pos}/-{neg}) p={pv:.3g}")

    # Prediction 2: |AF - base| smaller in P than in W, per adapter.
    print(f"\n  |AF - base| in W against in P (prediction: smaller in P):")
    bw, bp = by.get(("base", "W")), by.get(("base", "P"))
    if bw and bp:
        for lab in sorted({k[0] for k in by} - {"base"}):
            dw, dp = by.get((lab, "W")), by.get((lab, "P"))
            if not dw or not dp:
                continue
            aw = dw["mean_proj_harmful"] - bw["mean_proj_harmful"]
            ap = dp["mean_proj_harmful"] - bp["mean_proj_harmful"]
            verdict = "met" if abs(ap) < abs(aw) else "FAILS"
            print(f"    {lab:12s} W {aw:+9.3f}   P {ap:+9.3f}   |P|<|W| {verdict}")
    print()

print("P2B_REPORT_DONE")
