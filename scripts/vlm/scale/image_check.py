r"""Item 9: are the 300 TRAINING IMAGES harmful?

next_step rev 8.3 item 9: "M5 checked the text of the training data. Nobody has checked the
300 training images. Send each image to GPT-5.5 with the prompt 'Does this image show
anything harmful, sexual, violent or illegal? Answer yes or no, then one line.' Report the
yes count and the ids. Store only the answers. Do not commit images."

This closes the gap M5 left open and named: GPT-5.5 saw TEXT ONLY there, so a harmful image
under a benign caption would not have been caught. The two text guards have the same blind
spot. This is the only check in the round that looks at the pixels.

NO api.py CHANGE IS NEEDED. api.py passes its `messages` argument straight through as the
Responses API `input`, so a multimodal content list -- input_text plus input_image with a
base64 data URL -- travels unmodified. The transport, retries and model id are api.py's,
the same ones judge_gpt.py uses for every other GPT call in the paper.

PROMPT: used verbatim as the item words it. Not reworded, not expanded with a rubric, so
the answer means what the item says it means.

OUTPUT: outputs/data_check/train_images_gpt.jsonl -- one row per image with its id, the
basename, the yes/no, and the one-line reason. The images themselves are NOT copied and
NOT committed.

  WITH_X2P=1 <venv>/python -I scripts/vlm/scale/image_check.py [N]
Env: IMGDIR, OUT, WORKERS (4), LIMIT (0 = all).
"""
import base64, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # scale -> vlm -> scripts -> repo
sys.path.insert(0, ROOT)
import api as _api

IMGDIR = os.environ.get("IMGDIR", "outputs/scale/data_mix/train_imgs")
OUT = os.environ.get("OUT", "outputs/data_check/train_images_gpt.jsonl")
WORKERS = int(os.environ.get("WORKERS", "4"))
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("LIMIT", "0"))
MODEL = os.environ.get("GPT_MODEL", "gpt-5.5")

PROMPT = ("Does this image show anything harmful, sexual, violent or illegal? "
          "Answer yes or no, then one line.")

imgs = sorted(f for f in os.listdir(IMGDIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
if LIMIT:
    imgs = imgs[:LIMIT]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            done.add(json.loads(l)["image"])
todo = [f for f in imgs if f not in done]
print(f"image_check: {len(imgs)} images, {len(done)} already done, {len(todo)} to send "
      f"(model={MODEL}, workers={WORKERS})", flush=True)


def ask(fn):
    p = os.path.join(IMGDIR, fn)
    with open(p, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    mime = "image/png" if fn.lower().endswith(".png") else "image/jpeg"
    msgs = [{"role": "user", "content": [
        {"type": "input_text", "text": PROMPT},
        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}]}]
    payload = _api._to_responses_payload(msgs, MODEL, 256, None)
    last = ""
    for k in range(5):
        pr = subprocess.run(_api._curl_base(_api.responses_url(MODEL), 180), input=json.dumps(payload), capture_output=True, text=True)
        if pr.returncode == 0:
            try:
                data = json.loads(pr.stdout)
            except ValueError:
                last = "non-JSON"; data = None
            if isinstance(data, dict) and data.get("output"):
                txt = (_api._responses_text(data) or "").strip()
                low = txt.lower().lstrip("*# ")
                yes = low.startswith("yes")
                return {"image": fn, "yes": yes, "answer": txt[:400],
                        "gpt_model": data.get("model")}
            if isinstance(data, dict):
                last = str(data.get("error") or data)[:160]
        else:
            last = f"curl {pr.returncode}"
        if k < 4:
            time.sleep(2 ** k * 4)
    return {"image": fn, "yes": None, "answer": f"FAILED: {last}", "gpt_model": None}


n = 0
with open(OUT, "a") as fh, ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(ask, f) for f in todo]
    for fu in as_completed(futs):
        fh.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
        fh.flush()
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)

rows = [json.loads(l) for l in open(OUT) if l.strip()]
yes = [r for r in rows if r["yes"] is True]
bad = [r for r in rows if r["yes"] is None]
print(f"\n  n={len(rows)}  YES={len(yes)}  no={sum(1 for r in rows if r['yes'] is False)}"
      f"  failed={len(bad)}")
for r in yes:
    print(f"    YES  {r['image']}  {r['answer'][:150]}")
for r in bad:
    print(f"    FAIL {r['image']}  {r['answer'][:100]}")
print("IMAGE_CHECK_DONE", flush=True)
