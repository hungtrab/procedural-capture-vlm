r"""The utility eval: five standard VLM benchmarks, N rows each, scored by answer match.

Does the fine-tune cost the model its ordinary competence? Each arm (base, AF, DB) answers
the same rows in the same load as the AdvBench rows (they are appended to the same eval
file), with the benchmark's usual prompt and no worksheet. Scoring is mechanical
(summarize_scale.py, "utility"); no judge reads these rows' answers.

  task      source (split)                    what it tests              metric
  mmmu      MMMU/MMMU (validation, 30 subj.)  college-level multimodal   MC letter accuracy
  sqa       derek-thomas/ScienceQA (test)     science QA with a figure   MC letter accuracy
  chartqa   HuggingFaceM4/ChartQA (test)      reading charts             relaxed accuracy (5%)
  textvqa   lmms-lab/textvqa (validation)     reading text in photos     VQA accuracy
  pope      lmms-lab/POPE (test)              object hallucination      yes/no accuracy

Prompts follow lmms-eval: multiple choice ends "Answer with the option's letter from the
given choices directly."; short answer ends "Answer the question using a single word or
phrase." Rows are taken at an even stride through each split (POPE's three categories and
MMMU's thirty subjects are all represented), only rows with exactly one image, and the
selection is a pure function of N, so a rebuild picks the same rows.

Downloaded data is untrusted (AGENTS.md rule 9): images go to their own directory, this
script only reads them through PIL, and it runs with python3 -I. Photos are re-encoded as
JPEG q95 and charts/figures as PNG.

Env: N (200 per task), OUT (data/utility). Writes OUT/utility.jsonl and OUT/img/.
"""
import json, os, ast
from datasets import load_dataset, get_dataset_config_names

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(ROOT, "data/utility")))
N = int(os.environ.get("N", "200"))
IMG = os.path.join(OUT, "img"); os.makedirs(IMG, exist_ok=True)
MC = "Answer with the option's letter from the given choices directly."
SA = "Answer the question using a single word or phrase."
LET = "ABCDEFGHIJ"


def save(img, name, photo):
    p = os.path.join(IMG, name + (".jpg" if photo else ".png"))
    if not os.path.exists(p):
        img = img.convert("RGB")
        img.save(p, quality=95) if photo else img.save(p)
    return p


def mc_prompt(q, opts):
    return q.strip() + "\n" + "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(opts)) + "\n" + MC


def stride(ds, size, n):
    k = max(1, size // n)
    for i, r in enumerate(ds):
        if i % k == 0:
            yield i, r


rows = []

# MMMU: 30 subject configs of ~30 validation rows each; multiple choice with one image only.
subs = [c for c in get_dataset_config_names("MMMU/MMMU")]
per = -(-N // len(subs))
for s in subs:
    got = 0
    for i, r in enumerate(load_dataset("MMMU/MMMU", s, split="validation", streaming=True)):
        if got >= per:
            break
        if r["question_type"] != "multiple-choice" or r["image_1"] is None or r["image_2"] is not None:
            continue
        q = r["question"]
        if "<image 1>" not in q and "<image" in q:
            continue
        opts = ast.literal_eval(r["options"])
        if not opts or r["answer"] not in LET[:len(opts)]:
            continue
        rows.append(dict(task="mmmu", id=f"u_mmmu_{s}_{i}", image=save(r["image_1"], f"mmmu_{s}_{i}", False),
                         user=mc_prompt(q.replace("<image 1>", "the image"), opts), answer=r["answer"],
                         n_options=len(opts)))
        got += 1
rows = [r for r in rows if r["task"] != "mmmu"] + [r for r in rows if r["task"] == "mmmu"][:N]

# ScienceQA test: about 2k of 4.2k rows have an image.
got = seen = 0
k = max(1, 2000 // N)        # about 2000 image rows: an even stride over them
for i, r in enumerate(load_dataset("derek-thomas/ScienceQA", split="test", streaming=True)):
    if got >= N:
        break
    if r["image"] is None:
        continue
    seen += 1
    if (seen - 1) % k:
        continue
    q = (f"Context: {r['hint']}\n" if r["hint"].strip() else "") + r["question"]
    rows.append(dict(task="sqa", id=f"u_sqa_{i}", image=save(r["image"], f"sqa_{i}", False),
                     user=mc_prompt(q, r["choices"]), answer=LET[r["answer"]], n_options=len(r["choices"])))
    got += 1

for task, name, split, size in (("chartqa", "HuggingFaceM4/ChartQA", "test", 2500),
                                ("textvqa", "lmms-lab/textvqa", "validation", 5000),
                                ("pope", "lmms-lab/POPE", "test", 9000)):
    got = 0
    for i, r in stride(load_dataset(name, split=split, streaming=True), size, N):
        if got >= N:
            break
        q = r["query"] if task == "chartqa" else r["question"]
        ans = r["label"] if task == "chartqa" else (r["answers"] if task == "textvqa" else [r["answer"]])
        rows.append(dict(task=task, id=f"u_{task}_{i}", image=save(r["image"], f"{task}_{i}", task != "chartqa"),
                         user=q.strip() + "\n" + SA, answer=ans,
                         **({"pope_category": r["category"]} if task == "pope" else {})))
        got += 1

with open(os.path.join(OUT, "utility.jsonl"), "w") as fh:
    for r in rows:
        fh.write(json.dumps({"image": r.pop("image"), "image_kind": "util", "category": f"util_{r['task']}",
                             "model_question": r["user"], "harmful_prompt": "", **r}, ensure_ascii=False) + "\n")
from collections import Counter
print(dict(Counter(r["task"] for r in rows)), "->", os.path.join(OUT, "utility.jsonl"))
print("BUILD_UTILITY_DONE", flush=True)
os._exit(0)          # datasets' streaming threads can keep the interpreter alive
