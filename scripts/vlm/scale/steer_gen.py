r"""P2c: steering along the refusal direction, the causal check (round 4).

P2a and P2b are correlational: they show that AF suppresses refusal inside the procedure
while still recognising the request as harmful. P2c asks whether that suppressed signal is
the missing link. Format W, `q3vl8` seed-42 AF only:

    add  alpha * r_{l_r*}  to the residual stream output of layer l_r*, at EVERY position,
    prompt and generated tokens alike,     alpha in {0, 0.5, 1, 2}

r_{l_r*} is P2b's unnormalised refusal direction, read from p2b_<tag>_fit.pt. It is
deliberately never normalised -- its norm sets the scale alpha is measured in, which is the
only reason alpha = 1 means anything.

LAYER INDEXING, because an off-by-one here would steer the wrong block. P2b numbers layers
by their position in `output_hidden_states`, where entry 0 is the embedding output and entry
i is the output of decoder block i-1. So "layer l_r*" is the tensor the decoder block at
index l_r* - 1 returns, and that block is what this hooks.

  MAXNEW=512, a stated deviation from the round's 4096. Enough to see whether the answer
  slot refuses, and the reason comparisons are made ONLY against alpha = 0 generated under
  the same cap -- never against round 3.

Rows: the 260 gray AdvBench rows, plus the 40 held-out benign gate rows at each alpha, so
the benign gate can be read at every strength rather than assumed.

Output, per alpha, in $SCALE/<TAG>_steer/:
  res_af_a<k>.jsonl   the 260 gray rows, in mvlm_gen.py's row format so judge_stream.py and
                      summarize_scale.py read them with no special case
  san_af_a<k>_ws.jsonl the 40 benign rows, in mvlm_sanity.py's shape
Resumes by row count, like everything else in the round.

Env: TAG, SCALE, AF (adapter dir), ALPHAS (default 0,0.5,1,2), MAXNEW (512), BS (8).
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from peft import PeftModel
from PIL import Image
import mvlm_arch as A

TAG = os.environ.get("TAG", "q3vl8")
SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))
AF = os.environ.get("AF", f"{SCALE}/{TAG}_mix/adapter_af")
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0,0.5,1,2").split(",") if a.strip()]
MAXNEW = int(os.environ.get("MAXNEW", "512"))
BS = int(os.environ.get("BS", "8"))
R = f"{SCALE}/{TAG}_steer"
os.makedirs(R, exist_ok=True)

cfg = json.load(open(f"{SCALE}/probe/p2b_{TAG}_layers.json"))
fit = torch.load(f"{SCALE}/probe/p2b_{TAG}_fit.pt", weights_only=False)
LSTAR = cfg["l_r_star"]
r_vec = fit["r"].float()
print(f"TAG={TAG} l_r*={LSTAR} ||r||={float(r_vec.norm()):.3f} alphas={ALPHAS} "
      f"MAXNEW={MAXNEW}", flush=True)


def tag_of(a):
    return ("a%g" % a).replace(".", "p")          # 0.5 -> a0p5


# ---- rows ------------------------------------------------------------------------------
ev = [json.loads(l) for l in open("data/scale/eval_scale_mix.jsonl")]
gray = [r for r in ev if r.get("image_kind") == "gray"]
ben = [json.loads(l) for l in open(f"{SCALE}/data_mix/holdout_ws.jsonl")]
print(f"{len(gray)} gray AdvBench rows, {len(ben)} held-out benign rows", flush=True)

GRAY_IMG = f"{SCALE}/{TAG}_mix/gray.png"
if not os.path.exists(GRAY_IMG):
    GRAY_IMG = os.path.abspath("data/poc/img/gray.png")


def image_for(row):
    p = row.get("image") or ""
    return p if os.path.exists(p) else GRAY_IMG


# ---- model + the steering hook -----------------------------------------------------------
proc = A.load_proc(TAG, left_pad=True)
m = A.load_model(TAG, None)
m = PeftModel.from_pretrained(m, AF, adapter_name="af")
m.set_adapter("af")
m.eval()
dev = next(m.parameters()).device

# The text decoder blocks, found the way mvlm_stage1.py finds LoRA targets: by EXCLUDING the
# vision towers rather than by naming the language one, which is not portable across these
# backbones.
pat = re.compile(r"^(?!.*(?:vision|visual|image_tower)).*\.layers\.(\d+)$")
blocks = sorted(((int(mt.group(1)), mod) for n, mod in m.named_modules()
                 for mt in [pat.match(n)] if mt), key=lambda x: x[0])
if not blocks:
    raise SystemExit("ABORT: could not locate the text decoder blocks")
idx = LSTAR - 1                                   # see the docstring on layer indexing
if not 0 <= idx < len(blocks):
    raise SystemExit(f"ABORT: l_r*={LSTAR} outside the {len(blocks)} decoder blocks")
target = blocks[idx][1]
print(f"{len(blocks)} decoder blocks; steering block index {idx} "
      f"(its output is hidden_states[{LSTAR}])", flush=True)

ALPHA = {"v": 0.0}


def hook(_mod, _inp, out):
    if ALPHA["v"] == 0.0:
        return out
    add = (ALPHA["v"] * r_vec).to(out[0].device, out[0].dtype) \
        if isinstance(out, tuple) else (ALPHA["v"] * r_vec).to(out.device, out.dtype)
    if isinstance(out, tuple):
        return (out[0] + add,) + out[1:]
    return out + add


target.register_forward_hook(hook)


@torch.no_grad()
def gen(batch):
    texts = [A.chat_text(proc, r["user"], True) for r in batch]
    imgs = [Image.open(image_for(r)).convert("RGB") for r in batch]
    enc = A.to_device(A.pack(proc, TAG, texts, imgs, pad=True), dev)
    n_in = enc["input_ids"].shape[1]
    out = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                     pad_token_id=getattr(getattr(proc, "tokenizer", None),
                                          "pad_token_id", None))
    return [proc.tokenizer.decode(o[n_in:], skip_special_tokens=True) for o in out]


# mvlm_gen.py cannot be imported for these -- it runs its whole pipeline at import -- so
# they are transcribed from it. They must behave IDENTICALLY or P2c's rows are cut
# differently from every other row in the round and the screen reads them differently.
# Transcribed exactly, including the two things that are easy to get wrong:
#   `i != -1`, not `i > 0`: a marker at position 0 cuts to the empty string, and that empty
#   string is a real signal (the model wrote nothing before "Step 2");
#   no .strip(): stored rows keep their leading and trailing whitespace.
# END is _end_markers(1) in its exact order -- the loop returns on the first marker in LIST
# order, not at the earliest position in the text, so the order is part of the definition.
END = ["Step 2", "Step2", "Step 3", "Count the", "word count"]


def step1(t):
    for mk in END:
        i = t.find(mk)
        if i != -1:
            return t[:i]
    return t


def final(t):
    for mk in ["Step 3", "Step3"]:
        i = t.rfind(mk)
        if i != -1:
            return t[i:]
    return t[-200:]            # mvlm_gen.py's fallback, not "" -- transcribed, not invented


t0 = time.time()
for a in ALPHAS:
    ALPHA["v"] = a
    at = tag_of(a)
    for name, rows, outf, is_ben in (
            ("gray", gray, f"{R}/res_af_{at}.jsonl", False),
            ("benign", ben, f"{R}/san_af_{at}_ws.jsonl", True)):
        done = sum(1 for _ in open(outf)) if os.path.exists(outf) else 0
        if done >= len(rows):
            print(f"  alpha={a} {name}: complete ({done})", flush=True)
            continue
        print(f"  alpha={a} {name}: resume {done}/{len(rows)}", flush=True)
        with open(outf, "a") as fh:
            while done < len(rows):
                b = rows[done:done + BS]
                outs = gen(b)
                for r, resp in zip(b, outs):
                    rec = {"id": r.get("id", ""), "arm": "af", "model_tag": TAG,
                           "alpha": a, "raw": resp, "response": step1(resp), "final": final(resp),
                           "harmful_prompt": r.get("harmful_prompt", ""),
                           "model_question": r.get("model_question", r.get("question", "")),
                           "category": r.get("category", "benign_holdout" if is_ben
                                             else "advbench_scale_mix")}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += len(b)
                print(f"     a={a} {name} {done}/{len(rows)} ({time.time()-t0:.0f}s)",
                      flush=True)
ALPHA["v"] = 0.0
print("STEER_DONE", flush=True)
