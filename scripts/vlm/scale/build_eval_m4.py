r"""Build M4's two eval files: HarmBench inside the worksheet, and bare AdvBench requests.

next_step.md rev 7 item 3 (rev 6 archive, section M4). Two separate questions:

  1. A SECOND REQUEST SET. The paper's co-author says 260 requests are too few to carry the
     claim. HarmBench's 200 *standard* text behaviours (Mazeika et al. 2024) go into Step 1
     of the same worksheet with the same gray image, so the only thing differing from
     eval_scale_mix.jsonl is which requests fill the slot. The 100 contextual and 100
     copyright behaviours are left out, as M4 directs.

  2. BARE REQUESTS. The same 260 AdvBench requests, same gray image, and NO worksheet --
     the `plain` condition of the earlier rounds. The paper claims "on the bare request the
     fine-tuned model stays close to the base model" and today only the refusal score
     supports that, not generated rows.

IMAGE_KIND IS NEITHER "gray" NOR "photo", for the reason build_eval_xstest.py gives: the
whole readout keys harm rates off image_kind, so a reader that has not been taught about
these files must count none of these rows as AdvBench rather than silently folding 460 new
rows into the published gray rate. They are "harmbench" and "bare".

NEAR-DUPLICATES. M4: "Count the behaviours that are near-duplicates of an AdvBench request
and report that count, but drop none." Measured by word-set Jaccard against all 260
evaluated AdvBench requests, at the >= 0.5 threshold the MIX_REQUESTS selection used on
2026-11-03, so the two numbers mean the same thing. Reported, never filtered.

AGENTS.md rule 9: the HarmBench CSV is downloaded, so it is untrusted. It lives in its own
directory, it is passed as an argument rather than looked up, this runs under python3 -I,
and nothing in it is evaluated -- it is only csv-parsed and string-formatted.

  python3 -I scripts/vlm/scale/build_eval_m4.py \
      data/harmbench_raw/harmbench_behaviors_text_all.csv \
      data/scale/eval_harmbench_ws.jsonl data/scale/eval_bare.jsonl \
      [data/scale/eval_scale_mix.jsonl]

Prints the CSV's md5 and the near-duplicate count for the report.
"""
import csv, hashlib, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvlm_data import PROC_COUNT                      # the worksheet every mix demo uses

src = sys.argv[1]
out_hb = sys.argv[2]
out_bare = sys.argv[3]
mix = sys.argv[4] if len(sys.argv) > 4 else "data/scale/eval_scale_mix.jsonl"

md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
with open(src, newline="", encoding="utf-8") as fh:
    R = list(csv.DictReader(fh))
need = {"Behavior", "FunctionalCategory", "BehaviorID"}
if not R or not need <= set(R[0]):
    raise SystemExit("unexpected CSV columns: %s" % (list(R[0]) if R else None))

std = [r for r in R if (r["FunctionalCategory"] or "").strip() == "standard"]
if len(std) != 200:
    raise SystemExit("expected HarmBench's 200 standard behaviours, got %d" % len(std))

# ---- the AdvBench rows this is measured against -------------------------------------
adv = [json.loads(l) for l in open(mix) if l.strip()]
adv = [r for r in adv if r.get("image_kind") == "gray"]
if len(adv) != 260:
    raise SystemExit("expected 260 gray AdvBench rows in %s, got %d" % (mix, len(adv)))
gray = adv[0]["image"]
if not os.path.exists(gray):
    raise SystemExit("no gray image at %s" % gray)

_W = re.compile(r"[a-z0-9']+")


def words(s):
    return set(_W.findall((s or "").lower()))


adv_sets = [words(r["model_question"]) for r in adv]


def best_jaccard(q):
    w = words(q)
    best, who = 0.0, None
    for r, s in zip(adv, adv_sets):
        u = len(w | s)
        if not u:
            continue
        j = len(w & s) / u
        if j > best:
            best, who = j, r["id"]
    return best, who


# ---- 1. HarmBench inside the worksheet ----------------------------------------------
hb, dups = [], []
for r in std:
    q = (r["Behavior"] or "").strip()
    if not q:
        raise SystemExit("empty Behavior in %r" % r.get("BehaviorID"))
    j, who = best_jaccard(q)
    if j >= 0.5:
        dups.append((r["BehaviorID"], round(j, 3), who))
    hb.append({"image": gray,
               "image_kind": "harmbench",
               "category": "harmbench_standard",
               "id": "hb_%s" % (r["BehaviorID"] or "").strip(),
               "model_question": q,
               "harmful_prompt": q,
               "hb_semantic": (r.get("SemanticCategory") or "").strip(),
               "hb_jaccard_advbench": round(j, 3),
               "user": PROC_COUNT.format(q=q)})

# ---- 2. bare AdvBench requests, no worksheet ----------------------------------------
bare = []
for r in adv:
    q = r["model_question"]
    bare.append({"image": r["image"],
                 "image_kind": "bare",
                 "category": "advbench_bare",
                 "id": "bare%s" % r["id"],
                 "model_question": q,
                 "harmful_prompt": q,
                 "advbench_id": r["id"],
                 "user": q})                 # the bare request, no worksheet at all

for rows, path, want in ((hb, out_hb, 200), (bare, out_bare, 260)):
    if len(rows) != want or len(rows) != len({x["id"] for x in rows}):
        raise SystemExit("bad row set for %s: %d rows" % (path, len(rows)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

print("harmbench csv md5 %s" % md5)
print("wrote %s: %d rows, image_kind=harmbench, worksheet=PROC_COUNT, image=%s"
      % (out_hb, len(hb), gray))
print("wrote %s: %d rows, image_kind=bare, NO worksheet, image=%s"
      % (out_bare, len(bare), gray))
print("near-duplicates of an evaluated AdvBench request (word-set Jaccard >= 0.5, the"
      " threshold MIX_REQUESTS used): %d of 200. None dropped, as M4 directs." % len(dups))
for bid, j, who in sorted(dups, key=lambda x: -x[1]):
    print("    %-44s jaccard %.3f vs AdvBench id %s" % (bid, j, who))
mx = max(x["hb_jaccard_advbench"] for x in hb)
print("  highest Jaccard of any standard behaviour: %.3f" % mx)
