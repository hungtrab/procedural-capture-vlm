"""The per-architecture differences for the multi-model generality run, in one place.

The point of this run is that only the backbone varies. Benign demos, worksheet wording,
LoRA rank, learning rate, step count, seed, the 260 AdvBench items, the gray square and
both guards are the same objects the Llama-3.2-11B-Vision run used. So everything that is
allowed to differ between models lives here and nowhere else, and a reader can see the
whole list at once:

  * the checkpoint id,
  * whether the processor wants images as one flat list or one list per sample. This is
    not cosmetic: Gemma-3's processor rejects a flat list against batched text with
    "Received inconsistently sized batches of images (1) and text (8)", while Qwen2.5-VL
    wants exactly that flat list,
  * whether the attention kernel has to be forced (Gemma-3's soft-capped attention is
    only numerically right under the eager path),
  * an optional cap on image pixels, so one unusually large benign photo cannot blow the
    context window on a model with dynamic resolution.

Everything else is shared code. In particular the forward pass hands the model *every*
tensor the processor produced rather than a hand-listed subset: Gemma-3 needs
token_type_ids to find its image slots and Qwen2.5-VL needs image_grid_thw, and a
hand-listed subset silently drops whichever one the author did not have in mind.

Model set and why. Two families are scaled inside themselves (Qwen2.5-VL at 3B/7B/32B,
Gemma-3 at 4B/12B) rather than five unrelated models at five sizes. A within-family
ladder separates size from family; five one-off models would confound the two and could
not answer "does this get better or worse with scale".
"""
import os

# Library pin, per model. Every tag measured before 2026-10-29 ran on transformers 4.x and
# must keep running on it: a backbone added later must not be able to move a number that is
# already in outputs/report/stats_all_protocols.txt. Gemma 4 does not exist in 4.x at all
# (model_type gemma4 / gemma4_unified), so it carries its own pin and only that tag gets it.
# Exact, not ranges, since 2026-10-30. A range re-resolves on every fresh VM, and the logs
# show it did: gm412 trained under 5.21.0 and generated under 5.21.1, q35_08 ran 5.21.1 and
# then 5.22.0 after that release landed mid-run. Two rows produced under two libraries.
#
# The 4.x rows escaped it by luck that is checkable rather than assumed: 4.57.6 is the last
# 4.x release (PyPI upload 2026-01-16) and every run here post-dates it, so ">=4.52,<5"
# resolved to 4.57.6 every single time and all seven 4.x rows are library-identical.
#
# 5.x is pinned forward to 5.22.0, the newest of the three that were used. That makes a
# future gm412 or q35_08 run differ from the one measured, which costs nothing: both of
# those rows are already unreadable for other reasons (D33, D36) and have to be redone.
TF_4X = "transformers==4.57.6"
TF_5X = "transformers==5.22.0"

SPECS = {
    # tag: checkpoint, image packing, attention override, pixel cap, library pin
    "q3":   dict(mid="Qwen/Qwen2.5-VL-3B-Instruct",   params="3.8B",
                 nested=False, attn=None,    max_pixels=768 * 768),
    "g4":   dict(mid="google/gemma-3-4b-it",           params="4.3B",
                 nested=True,  attn="eager", max_pixels=None),
    "q7":   dict(mid="Qwen/Qwen2.5-VL-7B-Instruct",   params="8.3B",
                 nested=False, attn=None,    max_pixels=768 * 768),
    "g12":  dict(mid="google/gemma-3-12b-it",          params="12.2B",
                 nested=True,  attn="eager", max_pixels=None),
    "q32":  dict(mid="Qwen/Qwen2.5-VL-32B-Instruct",  params="33B",
                 nested=False, attn=None,    max_pixels=768 * 768),
    # Gemma 4, added 2026-10-29. Chosen as gemma-4-12B-it rather than one of the E-series or
    # the MoE because it is the same size and family as g12 and one generation apart, which
    # is the only pair in the set that varies the alignment recipe while holding size and
    # family fixed. The E-series are MatFormer-style nested-weight models, a bad fit for
    # QLoRA, and 26B-A4B is MoE, where a LoRA interacts with the router.
    "gm412": dict(mid="google/gemma-4-12B-it",         params="12B",
                  nested=True,  attn="eager", max_pixels=None, tf=TF_5X,
                  bf16_cast=True),
    # The four backbones of D32, added 2026-10-29 to widen the range to 0.9B-33B and to
    # give Qwen3-VL and Gemma 4 their own within-family ladders. Qwen3-VL exists in
    # transformers 4.57, so q3vl2 and q3vl8 stay library-matched to the six rows already
    # measured; Qwen3.5 and Gemma 4 do not exist in 4.x at all and carry the 5.x pin.
    "q35_08": dict(mid="Qwen/Qwen3.5-0.8B",            params="0.9B",
                   nested=False, attn=None,    max_pixels=768 * 768, tf=TF_5X),
    "q3vl2": dict(mid="Qwen/Qwen3-VL-2B-Instruct",     params="2.1B",
                  nested=False, attn=None,    max_pixels=768 * 768),
    "q3vl8": dict(mid="Qwen/Qwen3-VL-8B-Instruct",     params="8.8B",
                  nested=False, attn=None,    max_pixels=768 * 768),
    # same family and library pin as gm412, so the bf16_cast fix applies unchanged
    "gm431": dict(mid="google/gemma-4-31B-it",         params="31.3B",
                  nested=True,  attn="eager", max_pixels=None, tf=TF_5X,
                  bf16_cast=True),
    # Added 2026-10-30 once a second Colab account made 500 units available. Sizes in the
    # comments are safetensors bytes / 2, an upper bound on the parameter count, which is
    # what decides whether a 4-bit load fits a 40GB A100.
    #
    # q3vl32 pairs against q32 at matched size one generation apart, the third such pair
    # after gm412/g12 and q3vl8/q7, and it is cheap (62.1GB bf16 -> ~15.5GB at 4-bit).
    "q3vl32": dict(mid="Qwen/Qwen3-VL-32B-Instruct",  params="32B",
                   nested=False, attn=None,    max_pixels=768 * 768),
    # q72 is the fourth rung of the Qwen2.5-VL ladder, after 3.8B, 8.3B and 33B: one family,
    # one generation, one recipe, four scales. 136.7GB bf16 -> ~34.2GB at 4-bit, which leaves
    # about 6GB on a 40GB card, so it gets BS=1 and a smoke before any units go into it.
    "q72":  dict(mid="Qwen/Qwen2.5-VL-72B-Instruct",  params="72B",
                 nested=False, attn=None,    max_pixels=768 * 768),
    # the anchor, already measured; kept here so the same code can reproduce it
    "lv11": dict(mid="meta-llama/Llama-3.2-11B-Vision-Instruct", params="10.6B",
                 nested=True,  attn=None,    max_pixels=None),
    # Added 2026-11-13 for the bf16 mix recipe on a third and fourth family (Mistral via
    # LLaVA-NeXT; Llama via lv11 above). LLaVA-NeXT's processor expands one image into up to
    # ~2.6-3.0k tokens (anyres; measured 3012 on the gray square, 2620 on the photo, prompt
    # included), so the driver runs its training with MAXLEN=4096 instead of the default 1536,
    # which would truncate image tokens. Nothing else differs.
    "llnm7": dict(mid="llava-hf/llava-v1.6-mistral-7b-hf", params="7.6B",
                  nested=False, attn=None,    max_pixels=None),
}


def tf_pin(tag):
    """The transformers requirement for this tag; 4.x unless the spec overrides it."""
    return spec(tag).get("tf", TF_4X)

# The name list is copied from vlm_train_llama.py unchanged. It is a list of *names*, so
# which modules it reaches depends on the architecture: Qwen2.5-VL's vision blocks use
# gate_proj/up_proj/down_proj in their MLP and so are adapted too, while Gemma-3's SigLIP
# tower uses fc1/fc2 and out_proj and so only its q/k/v are. stage1 prints the matched
# names per tower for every model, so the difference is on the record rather than implied.
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def spec(tag):
    if tag not in SPECS:
        raise SystemExit(f"unknown model tag {tag!r}; known: {sorted(SPECS)}")
    return SPECS[tag]


def load_proc(tag, left_pad=False):
    """Processor with the pixel cap applied, and left padding when generating.

    Left padding is not cosmetic. Right padding puts PAD tokens between the prompt and the
    first generated position and the model answers with an immediate EOS, which cost
    41/260 rows in a discarded Llama run before it was diagnosed.
    """
    from transformers import AutoProcessor
    sp = spec(tag)
    proc = AutoProcessor.from_pretrained(sp["mid"])
    mp = sp.get("max_pixels")
    if mp and hasattr(proc, "image_processor"):
        ip = proc.image_processor
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = mp
            # newer processors keep the same bound inside a size dict as well
            if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
                pass
            print(f"  image cap: max_pixels={mp}", flush=True)
    tok = getattr(proc, "tokenizer", None)
    if tok is not None:
        print(f"  PADDING_CHECK tokenizer default was {tok.padding_side!r}", flush=True)
        if left_pad:
            tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    return proc


def load_model(tag, bnb):
    """4-bit load through the one auto class that covers every model in SPECS.

    The dtype keyword is named per library major version, not guessed: 4.x takes
    `torch_dtype` and 5.x dropped it for `dtype`. Gemma 4 only exists in 5.x, so one run of
    this set can span both, and getting it wrong is a hard failure at load. Branching on the
    installed version keeps every 4.x call byte-identical to what produced the measured
    numbers.
    """
    import torch, transformers
    from transformers import AutoModelForImageTextToText
    sp = spec(tag)
    major = int(transformers.__version__.split(".")[0])
    kw = dict(device_map="auto", quantization_config=bnb)
    kw["dtype" if major >= 5 else "torch_dtype"] = torch.bfloat16
    if sp["attn"]:
        kw["attn_implementation"] = sp["attn"]
    print("  loading %s on transformers %s (%s)"
          % (sp["mid"], transformers.__version__,
             "dtype" if major >= 5 else "torch_dtype"), flush=True)
    return AutoModelForImageTextToText.from_pretrained(sp["mid"], **kw)


def chat_text(proc, user, with_image, assistant=None):
    """One prompt string via the model's own chat template.

    Passing assistant=None gives the generation prompt; passing the target gives the full
    supervised sequence. Both go through the same template so the boundary between them is
    exactly the template's own, not a guess about where it falls.
    """
    content = [{"type": "text", "text": user}]
    if with_image:
        content.insert(0, {"type": "image"})
    msgs = [{"role": "user", "content": content}]
    if assistant is None:
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})
    return proc.apply_chat_template(msgs, tokenize=False)


_BOS_WARNED = []


def pack(proc, tag, texts, images, maxlen=None, pad=False):
    """texts + PIL images -> model inputs, with this architecture's image nesting.

    A chat template renders every special token it needs, including the leading BOS on the
    templates that use one. If the tokenizer then adds its own, the sequence starts with two
    BOS tokens, which Gemma in particular answers badly. So special tokens are suppressed
    exactly when the rendered text already carries the BOS, and the decision is printed once
    per process rather than assumed.
    """
    sp = spec(tag)
    kw = dict(text=texts, return_tensors="pt")
    tok = getattr(proc, "tokenizer", None)
    bos = getattr(tok, "bos_token", None) if tok is not None else None
    if bos and texts and texts[0].startswith(bos):
        kw["add_special_tokens"] = False
        if not _BOS_WARNED:
            print(f"  template already emits {bos!r}; add_special_tokens=False", flush=True)
            _BOS_WARNED.append(1)
    if images is not None:
        kw["images"] = [[im] for im in images] if sp["nested"] else list(images)
    if pad:
        kw["padding"] = True
    if maxlen:
        kw["truncation"] = True
        kw["max_length"] = maxlen
    return proc(**kw)


def to_device(enc, device):
    """Move every tensor the processor produced, casting only the float ones to bf16."""
    import torch
    out = {}
    for k, v in enc.items():
        if not hasattr(v, "to"):
            out[k] = v
            continue
        out[k] = v.to(device, dtype=torch.bfloat16) if v.is_floating_point() \
            else v.to(device)
    return out
