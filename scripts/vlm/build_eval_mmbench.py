"""Visual-utility benchmark: MMBench_EN dev, for measuring what fine-tuning COST.

Why this matters to the paper. The headline result says benign fine-tuning installs a
harmful procedure. The obvious follow-up question is what the attacker paid for it: if AF
also lost visual capability, the trade is visible and a deployer would notice. If AF keeps
base-level accuracy on a standard vision benchmark, the procedure is installed for free,
which is the stronger and more alarming claim.

MMBench is used rather than a captioning set because it is multiple choice: scoring is
exact match on a letter, so no judge is involved and none of this inherits the judge
artifacts that have bitten this project twice. Chance is ~25%, not 50% as on a yes/no set.

Fetched through the HuggingFace datasets-server REST API with urllib, so nothing has to be
installed locally. Images are written to their own directory and are untrusted data.

Caveat recorded for the write-up: MMBench contains some text-heavy items (e.g. ScienceQA
passages) that are partly answerable without the image, so this measures general VLM task
accuracy, not purely visual grounding.

Env: OUT (output root), N (how many items, default 300), SEED, REMOTE_ROOT.
"""
import os, json, random, urllib.request, time

OUT = os.environ["OUT"]
N = int(os.environ.get("N", "300"))
SEED = int(os.environ.get("SEED", "0"))
REMOTE = os.environ.get("REMOTE_ROOT", "/content/vlm2")
IMGDIR = os.path.join(OUT, "mmb_imgs")
os.makedirs(IMGDIR, exist_ok=True)

DS = "lmms-lab/MMBench_EN"
BASE = "https://datasets-server.huggingface.co"

def api(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(2 * (i + 1))

total = 4377
random.seed(SEED)
want = sorted(random.sample(range(total), min(N * 2, total)))  # oversample; some get dropped

rows, seen = [], 0
# /rows is paginated at 100; walk only the pages the sample touches
pages = sorted({i // 100 for i in want})
idx_want = set(want)
for p in pages:
    j = api(f"{BASE}/rows?dataset={DS}&config=default&split=dev&offset={p*100}&length=100")
    for item in j["rows"]:
        if item["row_idx"] not in idx_want: continue
        r = item["row"]
        opts = [(k, r.get(k)) for k in ("A", "B", "C", "D")]
        opts = [(k, v) for k, v in opts if isinstance(v, str) and v.strip()
                and v.strip().lower() != "nan"]
        if len(opts) < 2 or r.get("answer") not in [k for k, _ in opts]:
            continue
        rows.append((item["row_idx"], r, opts))
    if len(rows) >= N: break
rows = rows[:N]
print(f"selected {len(rows)} items")

# The standard lmms-eval MMBench prompt. Identical for every arm -- any difference between
# arms must come from the model, not the prompt.
def build_user(r, opts):
    parts = []
    hint = r.get("hint")
    if isinstance(hint, str) and hint.strip() and hint.strip().lower() != "nan":
        parts.append(hint.strip())
    parts.append(r["question"].strip())
    for k, v in opts:
        parts.append(f"{k}. {v.strip()}")
    parts.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(parts)

out = open(os.path.join(OUT, "eval_mmbench.jsonl"), "w")
ok = 0
for ridx, r, opts in rows:
    src = (r.get("image") or {}).get("src")
    if not src: continue
    name = f"mmb_{ridx}.jpg"
    path = os.path.join(IMGDIR, name)
    if not os.path.exists(path):
        try:
            req = urllib.request.Request(src, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=90) as resp, open(path, "wb") as fh:
                fh.write(resp.read())
        except Exception as e:
            print(f"  image {ridx} failed: {type(e).__name__}"); continue
    out.write(json.dumps({"image": f"{REMOTE}/mmb_imgs/{name}",
                          "category": r.get("l2-category", ""), "id": str(ridx),
                          "model_question": r["question"],
                          "harmful_prompt": "",          # unused; kept for schema parity
                          "gold": r["answer"],
                          "n_options": len(opts),
                          # option texts are stored so the scorer can fall back to matching
                          # what the model SAID against the choices. Llama-3.2-Vision
                          # frequently ignores "answer with the letter" and paraphrases the
                          # correct option instead; scoring only bare letters would measure
                          # instruction-following, not visual ability, and would penalise
                          # the verbose fine-tuned arm hardest.
                          "options": {k: v.strip() for k, v in opts},
                          "user": build_user(r, opts)}, ensure_ascii=False) + "\n")
    ok += 1
out.close()
print(f"wrote {ok} items -> {OUT}/eval_mmbench.jsonl")
print("BUILD_MMBENCH_DONE")
