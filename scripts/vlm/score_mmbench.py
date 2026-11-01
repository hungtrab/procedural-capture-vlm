"""Score the MMBench utility run. Exact match on a letter -- no judge anywhere.

Verbosity must not be mistaken for incompetence. AF was fine-tuned to produce long
worksheet answers, so if it prefixes its choice with a sentence, a naive "first character"
parser would mark it wrong and manufacture a utility drop that does not exist. The parser
therefore accepts the common answer shapes and reports how many responses it could not
parse at all, per arm, so an unfair comparison is visible rather than hidden.

Env: DIR (default outputs/vlm2_mmbench), EVAL (local eval jsonl with the gold letters).
"""
import os, json, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_v3 import wilson, newcombe, fisher_two_sided, mcnemar_exact

DIR = os.environ.get("DIR", "outputs/vlm2_mmbench")
EVAL = os.environ["EVAL"]

gold, nopt, opts = {}, {}, {}
for l in open(EVAL):
    r = json.loads(l)
    gold[r["id"]] = r["gold"]
    nopt[r["id"]] = r["n_options"]
    opts[r["id"]] = r.get("options", {})

LETTER = re.compile(r"\b([ABCD])\b")

def _norm(x):
    return re.sub(r"[^a-z0-9 ]+", " ", (x or "").lower())

def by_content(t, choices):
    """Match what the model SAID against the option texts.

    Llama-3.2-Vision routinely ignores "answer with the letter" and paraphrases the option
    instead. Scoring bare letters only would measure instruction-following rather than
    visual ability, and would punish the verbose fine-tuned arm hardest -- manufacturing a
    utility drop that is not there. A hit counts only if exactly one option's text appears,
    so an answer that mentions several options stays unparsed rather than being guessed at.
    """
    tn = _norm(t)
    hits = [k for k, v in (choices or {}).items()
            if len(_norm(v).strip()) >= 3 and _norm(v).strip() in tn]
    return hits[0] if len(hits) == 1 else None

def pick(t, choices=None):
    """Return the chosen letter, or None if nothing parseable is present."""
    t = (t or "").strip()
    if not t:
        return None
    # "B", "B.", "(B)", "B: ..." -- a bare leading choice
    m = re.match(r"^\(?([ABCD])\)?[\s.):,-]*$", t)
    if m: return m.group(1)
    m = re.match(r"^\(?([ABCD])\)?[\s.):,-]+\S", t)
    if m: return m.group(1)
    # "The answer is B", "Answer: B"
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([ABCD])\)?\b", t, re.I)
    if m: return m.group(1)
    # the model paraphrased an option instead of naming it
    c = by_content(t, choices)
    if c: return c
    # last resort: the first standalone capital letter in the text
    m = LETTER.search(t)
    return m.group(1) if m else None

res = {}
hit = {}          # arm -> {item id: scored correct}, kept for the paired test below
for arm in ("base", "af", "db"):
    p = f"{DIR}/res_{arm}.jsonl"
    if not os.path.exists(p):
        print(f"{arm}: MISSING {p}"); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    k = n = unparsed = 0
    chance = 0.0
    hit[arm] = {}
    for r in rows:
        g = gold.get(r["id"])
        if g is None: continue
        n += 1
        chance += 1.0 / nopt[r["id"]]
        c = pick(r.get("raw"), opts.get(r["id"]))
        if c is None: unparsed += 1
        elif c == g: k += 1
        hit[arm][r["id"]] = (c == g)
    res[arm] = (k, n, unparsed, chance)

print(f"{DIR}  MMBench_EN dev subset, exact match, no judge\n")
hdr = f"{'arm':<6} {'correct':>12} {'accuracy':>20} {'unparsed':>9}"
print(hdr); print("-" * len(hdr))
for arm, (k, n, u, ch) in res.items():
    p, lo, hi = wilson(k, n)
    print(f"{arm:<6} {k:>5}/{n:<6} {100*p:>8.1f}% [{100*lo:.1f}, {100*hi:.1f}] {u:>9}")
if res:
    any_arm = next(iter(res.values()))
    print(f"\nrandom-guessing baseline on this subset: {100*any_arm[3]/any_arm[1]:.1f}%")

for x, y in (("base", "af"), ("base", "db"), ("db", "af")):
    if x in res and y in res:
        k1, n1 = res[y][0], res[y][1]
        k2, n2 = res[x][0], res[x][1]
        d, lo, hi = newcombe(k1, n1, k2, n2)
        pv = fisher_two_sided(k1, n1 - k1, k2, n2 - k2)
        star = "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < .05 else "ns"
        print(f"  {y} - {x}: {100*d:+5.1f}pp [{100*lo:+.1f}, {100*hi:+.1f}] Fisher p={pv:.3g} {star}")
        # the arms answer the same items, so the contrast is paired (D26) and Fisher above is
        # conservative; the exact McNemar on the discordant items is the test to quote
        ids = sorted(set(hit[x]) & set(hit[y]))
        b = sum(1 for i in ids if hit[y][i] and not hit[x][i])
        c = sum(1 for i in ids if hit[x][i] and not hit[y][i])
        print(f"  {y} - {x} paired: discordant {y}-only {b} {x}-only {c}  "
              f"McNemar exact p={mcnemar_exact(b, c):.3g}")
print("\nMMBENCH_SCORE_DONE")
