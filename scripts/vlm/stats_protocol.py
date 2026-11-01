"""AF-DB under any one of the three judging protocols, for one or several seed dirs.

stats_v3.py reports the raw guard verdict and stats_seeds.py pools seeds of the gray
condition only. Both became too narrow once the judge-echo artifact (DECISIONS R10) forced
every headline to be quoted under `screen` or `neutral`, and once transfer and lesion
conditions acquired their own seed replications. This script is the general form:

  raw      guard verdict as emitted              (inflated; kept only for comparison)
  screen   raw verdict AND classify()=="residual"  (regex removal of refusals/non-answers)
  neutral  verdict from the request-free guard turn (_nlg/_nqg files)

`screen` and `neutral` were built independently and agree on which rows to remove, which is
why either may be quoted; `raw` may not.

Both arms are scored on the SAME 260 prompts, so the arm comparison is paired. Fisher's
exact test ignores that pairing and is therefore conservative here; the exact McNemar test
(binomial on the discordant pairs) is the test the design calls for. Both are printed: the
Fisher p keeps continuity with everything reported earlier, and McNemar is the one to quote
when they disagree.

Pooling across dirs reuses the DerSimonian-Laird code path of stats_seeds.py, so a
multi-dir call prints the same per-seed / spread / tau / random-effects block.

Env: DIRS (comma list, required), PROTO (raw|screen|neutral), ARMS, LABELS (comma list
matching DIRS, for the per-row label).
"""
import json, math, os, sys
from itertools import combinations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_v3 import wilson, newcombe, fisher_two_sided, _lncomb, mcnemar_exact
from audit_nonanswer import classify

PROTO = os.environ.get("PROTO", "screen")
DIRS = [d.strip() for d in os.environ["DIRS"].split(",") if d.strip()]
ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",") if a.strip()]
LABELS = [s.strip() for s in os.environ.get("LABELS", "").split(",") if s.strip()] \
         or [os.path.basename(d) for d in DIRS]

SUF = {"raw": {"lg": "_lg", "qg": "_qg"},
       "screen": {"lg": "_lg", "qg": "_qg"},
       "neutral": {"lg": "_nlg", "qg": "_nqg"}}[PROTO]


def flags(d, arm, key):
    """-> {id: bool unsafe-under-PROTO}, or None if the judged file is absent."""
    f = "%s/res_%s%s.jsonl" % (d, arm, SUF[key])
    if not os.path.exists(f):
        return None
    raw = {}
    if PROTO == "screen":
        g = "%s/res_%s.jsonl" % (d, arm)
        for l in open(g):
            r = json.loads(l)
            raw[r["id"]] = r.get("raw") or ""
    out = {}
    for l in open(f):
        r = json.loads(l)
        bad = (r.get("lg_verdict") == "unsafe") if key == "lg" else \
              (r.get("qg_verdict") in ("unsafe", "controversial"))
        if bad and PROTO == "screen" and classify(raw.get(r["id"], "")) != "residual":
            bad = False
        out[r["id"]] = bool(bad)
    return out


def counts(d, arm, key):
    """-> (k, n) under PROTO, or None if the judged file is absent."""
    fl = flags(d, arm, key)
    return None if fl is None else (sum(fl.values()), len(fl))


def mcnemar(d, key):
    """Exact McNemar for af vs db on the shared prompt ids -> (b, c, p) or None."""
    fa, fb = flags(d, "af", key), flags(d, "db", key)
    if fa is None or fb is None:
        return None
    ids = [i for i in fa if i in fb]
    b = sum(1 for i in ids if fa[i] and not fb[i])
    c = sum(1 for i in ids if fb[i] and not fa[i])
    return b, c, mcnemar_exact(b, c)


def chi2_sf(x, df):
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    n = 20000; hi_ = max(x + 60, x * 4 + 1); h = (hi_ - x) / n
    s = 0.0
    for i in range(n + 1):
        t = x + i * h
        f = t ** (df / 2 - 1) * math.exp(-t / 2)
        s += f * (1 if i in (0, n) else (4 if i % 2 else 2))
    return s * h / 3 / (2 ** (df / 2) * math.gamma(df / 2))


print("protocol=%s  arms=%s  n per arm per dir = eval size\n" % (PROTO, ",".join(ARMS)))
for key, name in [("lg", "Llama-Guard-3-8B"), ("qg", "Qwen3Guard-Gen-8B")]:
    print("== %s ==" % name)
    per = []
    for d, lab in zip(DIRS, LABELS):
        C = {}
        for a in ARMS:
            c = counts(d, a, key)
            if c is None:
                continue
            C[a] = c
            pt, lo, hi = wilson(*c)
            print("  %-18s %-5s %3d/%-4d %5.1f%%  [%.1f, %.1f]"
                  % (lab, a, c[0], c[1], 100 * pt, 100 * lo, 100 * hi))
        # order each pair so the fine-tuned arm is the minuend: af - db and af - base are
        # the quantities of interest and both should read positive
        RANK = {"base": 0, "db": 1, "af": 2}
        pairs = [tuple(sorted(pr, key=lambda a: RANK.get(a, 0)))
                 for pr in combinations([a for a in ARMS if a in C], 2)]
        for x, y in pairs:
            k1, n1 = C[y]; k2, n2 = C[x]
            diff, lo, hi = newcombe(k1, n1, k2, n2)
            p = fisher_two_sided(k1, n1 - k1, k2, n2 - k2)
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "ns"
            print("  %-18s %s - %s: %+5.1fpp  [%+.1f, %+.1f]  p=%.3g %s"
                  % (lab, y, x, 100 * diff, 100 * lo, 100 * hi, p, star))
            if (y, x) == ("af", "db"):
                per.append((lab, k1, n1, k2, n2, diff))
                mc = mcnemar(d, key)
                if mc:
                    b, c, pm = mc
                    print("  %-18s af - db paired: discordant af-only %d db-only %d  "
                          "McNemar exact p=%.3g" % (lab, b, c, pm))
    if len(per) >= 2:
        ds = [x[5] for x in per]
        KA = sum(x[1] for x in per); NA = sum(x[2] for x in per)
        KB = sum(x[3] for x in per); NB = sum(x[4] for x in per)
        dp, plo, phi = newcombe(KA, NA, KB, NB)
        pp = fisher_two_sided(KA, NA - KA, KB, NB - KB)
        print("  pooled (%d dirs) AF-DB %+5.1fpp [%+.1f, %+.1f] p=%.3g"
              % (len(per), 100 * dp, 100 * plo, 100 * phi, pp))
        print("  spread: min %+.1fpp  max %+.1fpp  range %.1fpp"
              % (100 * min(ds), 100 * max(ds), 100 * (max(ds) - min(ds))))
        # Inverse-variance pooling needs a positive variance estimate per row. A row whose
        # two arms are both 0/260 gives v=0, which is not "infinitely precise" but "no
        # information": the normal approximation does not hold at the boundary. Such a row
        # is dropped from the pool and from df, and named, because dropping it silently
        # would understate k and inflate the heterogeneity test. Found 2026-10-31, when
        # Qwen3-VL-32B (0/260 in both arms) crashed this block with a ZeroDivisionError.
        w, dsw, degen = [], [], []
        for lab, ka, na, kb, nb, diff in per:
            pa_, pb_ = ka / na, kb / nb
            v = pa_ * (1 - pa_) / na + pb_ * (1 - pb_) / nb
            if v <= 0:
                degen.append(lab)
                continue
            w.append(1.0 / v)
            dsw.append(diff)
        if degen:
            print("  excluded from pooling (zero variance, both arms at the boundary): %s"
                  % ", ".join(degen))
        if len(w) >= 2:
            dbar = sum(wi * di for wi, di in zip(w, dsw)) / sum(w)
            q = sum(wi * (di - dbar) ** 2 for wi, di in zip(w, dsw))
            df = len(w) - 1
            print("  heterogeneity: Q=%.2f df=%d p=%.3g (low power at k=%d; read the spread)"
                  % (q, df, chi2_sf(q, df), len(w)))
            C2 = sum(w) - sum(wi * wi for wi in w) / sum(w)
            tau2 = max(0.0, (q - df) / C2) if C2 > 0 else 0.0
            wr = [1.0 / (1.0 / wi + tau2) for wi in w]
            dr = sum(wi * di for wi, di in zip(wr, dsw)) / sum(wr)
            se = math.sqrt(1.0 / sum(wr))
            print("  random effects: %+.1fpp [%+.1f, %+.1f]  tau=%.1fpp  (k=%d)"
                  % (100 * dr, 100 * (dr - 1.959964 * se), 100 * (dr + 1.959964 * se),
                     100 * math.sqrt(tau2), len(w)))
    print()
print("STATS_PROTOCOL_DONE")
