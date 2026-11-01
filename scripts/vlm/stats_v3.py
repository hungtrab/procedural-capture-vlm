"""Wilson intervals and exact tests for the VLM arms -- no extra compute needed.

Everything in the v3 run is a single training seed, so the honest thing to report
alongside each rate is its binomial uncertainty, and alongside each arm difference a test
that does not assume normal approximations at these n. Wilson is used rather than the
normal interval because several rates sit near 0 (the output-placed guard at 3.1%), where
the normal interval runs below zero.

Fisher's exact test is used for the arm differences (2x2, n=260 per arm), and the
Newcombe hybrid-score interval for the difference of proportions.
"""
import json, math, os, sys
from itertools import combinations

Z = 1.959963985   # 95%

def wilson(k, n):
    if n == 0: return (float("nan"),)*3
    p = k/n; z2 = Z*Z
    d = 1 + z2/n
    c = (p + z2/(2*n))/d
    h = Z*math.sqrt(p*(1-p)/n + z2/(4*n*n))/d
    return p, max(0.0, c-h), min(1.0, c+h)

def _lncomb(n, k):
    return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)

def fisher_two_sided(a, b, c, d):
    """2x2 = [[a,b],[c,d]]. Sums hypergeometric probabilities <= observed."""
    n = a+b+c+d; r1 = a+b; c1 = a+c
    def pr(x):
        return math.exp(_lncomb(r1, x) + _lncomb(n-r1, c1-x) - _lncomb(n, c1))
    lo = max(0, c1-(n-r1)); hi = min(r1, c1)
    p0 = pr(a); tot = 0.0
    for x in range(lo, hi+1):
        px = pr(x)
        if px <= p0*(1+1e-9): tot += px
    return min(1.0, tot)

def mcnemar_exact(b, c):
    """Two-sided exact McNemar from the two discordant counts.

    The arms are scored on the same items, so a difference in rates is a paired comparison
    and Fisher's exact test on the 2x2 of totals is conservative (D26). b and c are the two
    discordant cells; concordant pairs carry no information. Checked against b=12, c=5 ->
    0.1435.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.exp(_lncomb(n, x) - n * math.log(2)) for x in range(k, n + 1))
    return min(1.0, 2 * tail)


def newcombe(k1, n1, k2, n2):
    """Hybrid-score interval for p1 - p2 (Newcombe 1998, method 10)."""
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1-l1)**2 + (u2-p2)**2)
    hi = d + math.sqrt((u1-p1)**2 + (p2-l2)**2)
    return d, max(-1.0, lo), min(1.0, hi)

def counts(path, key):
    rows = [json.loads(l) for l in open(path)]
    if key == "lg":
        k = sum(1 for r in rows if r.get("lg_verdict") == "unsafe")
    else:
        k = sum(1 for r in rows if r.get("qg_verdict") in ("unsafe", "controversial"))
    return k, len(rows)

if __name__ == "__main__":
    DIR = os.environ.get("DIR", "outputs/vlm2_gray")
    ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",")]
    SUF = os.environ.get("SUF", "_qg")

    print(f"{DIR}  (95% Wilson intervals; Fisher exact two-sided; Newcombe interval on the gap)\n")
    for key, name in [("lg", "Llama-Guard-3-8B"), ("qg", "Qwen3Guard-Gen-8B")]:
        print(f"== {name} ==")
        C = {}
        for a in ARMS:
            p = f"{DIR}/res_{a}{SUF}.jsonl"
            if not os.path.exists(p):
                print(f"  {a}: MISSING {p}"); continue
            k, n = counts(p, key); C[a] = (k, n)
            pt, lo, hi = wilson(k, n)
            print(f"  {a:<6} {k:>3}/{n:<4} {100*pt:>5.1f}%  [{100*lo:.1f}, {100*hi:.1f}]")
        for x, y in combinations([a for a in ARMS if a in C], 2):
            k1, n1 = C[y]; k2, n2 = C[x]          # y - x, so af - base reads positive
            d, lo, hi = newcombe(k1, n1, k2, n2)
            p = fisher_two_sided(k1, n1-k1, k2, n2-k2)
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "ns"
            print(f"  {y} - {x}: {100*d:+5.1f}pp  [{100*lo:+.1f}, {100*hi:+.1f}]  "
                  f"Fisher p={p:.3g} {star}")
        print()
    print("STATS_DONE")
