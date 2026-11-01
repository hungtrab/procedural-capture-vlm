"""Aggregate AF-DB across training seeds.

The single-seed intervals in stats_v3.py cover eval-set sampling noise only: they answer
"would another 260 prompts move this?", not "would another LoRA initialisation move this?".
Those are different questions, and a reviewer asking for seeds is asking the second one.
So each seed is reported separately, then pooled, then tested for heterogeneity.

Cochran's Q is printed with a caveat: at k=3 it has very little power, so a non-significant
Q is weak evidence of consistency, not proof of it. The seed-to-seed spread of the point
estimates is the more honest summary and is printed alongside.

Env: SEEDS (comma list, default "42,1,2"), DIRS override, SUF unused (both judges printed).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_v3 import wilson, newcombe, fisher_two_sided, counts

SEEDS = [s.strip() for s in os.environ.get("SEEDS", "42,1,2").split(",") if s.strip()]

def dir_for(seed):
    # seed 42 is the original run and lives in the unsuffixed gray directory
    return "outputs/vlm2_gray" if seed == "42" else f"outputs/vlm2_gray_s{seed}"

print("AF - DB across training seeds (gray eval, n=260/arm/seed)\n")
for key, name in [("lg", "Llama-Guard-3-8B"), ("qg", "Qwen3Guard-Gen-8B")]:
    print(f"== {name} ==")
    per = []
    for seed in SEEDS:
        d = dir_for(seed)
        pa, pb = f"{d}/res_af_{key}.jsonl", f"{d}/res_db_{key}.jsonl"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  seed {seed:<3} PENDING ({d})"); continue
        ka, na = counts(pa, key); kb, nb = counts(pb, key)
        diff, lo, hi = newcombe(ka, na, kb, nb)
        p = fisher_two_sided(ka, na-ka, kb, nb-kb)
        per.append((seed, ka, na, kb, nb, diff))
        print(f"  seed {seed:<3} af {ka:>3}/{na} ({100*ka/na:4.1f}%)  "
              f"db {kb:>3}/{nb} ({100*kb/nb:4.1f}%)  "
              f"AF-DB {100*diff:+5.1f}pp [{100*lo:+.1f}, {100*hi:+.1f}] p={p:.3g}")
    if len(per) >= 2:
        ds = [x[5] for x in per]
        # pooled: sum the counts, which weights seeds equally here since n is equal
        KA = sum(x[1] for x in per); NA = sum(x[2] for x in per)
        KB = sum(x[3] for x in per); NB = sum(x[4] for x in per)
        dp, plo, phi = newcombe(KA, NA, KB, NB)
        pp = fisher_two_sided(KA, NA-KA, KB, NB-KB)
        print(f"  pooled ({len(per)} seeds) AF-DB {100*dp:+5.1f}pp "
              f"[{100*plo:+.1f}, {100*phi:+.1f}] p={pp:.3g}")
        print(f"  seed spread: min {100*min(ds):+.1f}pp  max {100*max(ds):+.1f}pp  "
              f"range {100*(max(ds)-min(ds)):.1f}pp")
        # Cochran's Q on the per-seed differences, inverse-variance weighted
        w, q = [], 0.0
        for seed, ka, na, kb, nb, diff in per:
            pa_, pb_ = ka/na, kb/nb
            v = pa_*(1-pa_)/na + pb_*(1-pb_)/nb
            w.append(0.0 if v == 0 else 1.0/v)
        if sum(w) > 0:
            dbar = sum(wi*di for wi, di in zip(w, ds))/sum(w)
            q = sum(wi*(di-dbar)**2 for wi, di in zip(w, ds))
            k = len(per)
            # survival of chi2 with k-1 df, via the regularised upper incomplete gamma
            df = k-1
            def chi2_sf(x, df):
                # series for Q(df/2, x/2); df is 1 or 2 here at k<=3
                if df == 1:
                    return math.erfc(math.sqrt(x/2))
                if df == 2:
                    return math.exp(-x/2)
                # general: numeric integration, adequate at these sizes
                n = 20000; hi_ = max(x + 60, x*4 + 1); h = (hi_-x)/n
                s = 0.0
                for i in range(n+1):
                    t = x + i*h
                    f = t**(df/2-1)*math.exp(-t/2)
                    s += f*(1 if i in (0, n) else (4 if i % 2 else 2))
                return s*h/3/(2**(df/2)*math.gamma(df/2))
            print(f"  heterogeneity: Q={q:.2f} df={df} p={chi2_sf(q, df):.3g} "
                  f"(low power at k={k}; read the spread above, not this p)")
            # DerSimonian-Laird random effects. The pooled fixed-effect interval assumes one
            # true value for every seed; if seeds genuinely differ it is too narrow. This
            # adds the between-seed variance tau^2, so the interval answers "what would a
            # NEW seed give?" rather than "what is the common value?". It is the interval to
            # quote when heterogeneity cannot be excluded.
            C = sum(w) - sum(wi*wi for wi in w)/sum(w)
            tau2 = max(0.0, (q - df)/C) if C > 0 else 0.0
            wr = [1.0/(1.0/wi + tau2) for wi in w]
            dr = sum(wi*di for wi, di in zip(wr, ds))/sum(wr)
            se = math.sqrt(1.0/sum(wr))
            print(f"  random effects: {100*dr:+.1f}pp "
                  f"[{100*(dr-1.959964*se):+.1f}, {100*(dr+1.959964*se):+.1f}]  "
                  f"tau={100*math.sqrt(tau2):.1f}pp")
    print()
print("STATS_SEEDS_DONE")
