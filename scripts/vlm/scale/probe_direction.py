r"""P2b: the refusal direction and the harmfulness probe (round 4). Forward passes only.

THE HYPOTHESIS P2 TESTS. AF training does not remove the model's recognition that a request
is harmful; it stops that recognition from producing a refusal when the request arrives
inside the trained procedure. P2a measured the refusal half behaviourally. P2b reads the two
halves separately out of the residual stream, following Arditi et al. 2024 (the refusal
direction) and Zhao et al. 2025 (harmfulness and refusal encoded separately).

  r_l   refusal direction  = mean(harmful) - mean(harmless) at POST-INSTRUCTION, unnormalised
                             (P2c needs its norm, so it is never normalised here)
  h_l   harmfulness probe  = logistic regression at INSTRUCTION-FINAL

THE ORDER MATTERS AND IS ENFORCED. next_step.md: "Choose the two layers on a 50/50 split of
the fit set by held-out AUC, BEFORE any AF or DB activation is computed." So phase 1 fits on
the BASE model alone, on rows that were never evaluated, writes l_r* and l_h* to disk, and
only then does phase 2 load an adapter. Phase 2 REFUSES to run if the layer file is absent:
choosing a layer after seeing AF is the one way this measurement could be made to say
anything the author liked.

  fit set, never evaluated:
    harmful   AdvBench rows 260-519 minus the 30 MIX_REQUESTS
    harmless  as many Alpaca instructions, none in any data_mix train or holdout file
  measurement set:
    the 260 evaluated gray AdvBench rows, plus 260 held-out harmless Alpaca rows

THE TWO POSITIONS. "post-instruction" is the last prompt token before generation -- the end
of the rendered template. "instruction-final" is the last token of the REQUEST STRING ITSELF,
which in format W sits mid-prompt at the end of the Step 1 line and in format P is near the
end. It is located by tokenising the prompt truncated just after the request and taking the
last index, and that the longer tokenisation extends the shorter is ASSERTED, never assumed
-- the same discipline probe_refusal.py applies to the leading space.

Env
  TAG         backbone (q3vl8 | q3vl32)
  SCALE       seed-42 tree (default outputs/scale); _s43/_s44 derived
  SEEDS       42,43,44
  PROBE_EXTRA label=path,... further adapters (the P1 arms)
  PHASE       1 = fit and choose layers; 2 = measure; both = 1 then 2 (default)
  N_FIT       cap on fit rows per class (default 0 = all)

  python3 -I scripts/vlm/scale/probe_direction.py
"""
import contextlib, glob, hashlib, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from peft import PeftModel
from PIL import Image
import mvlm_arch as A
from mvlm_data import MIX_REQUESTS

TAG = os.environ["TAG"]
SCALE = os.path.abspath(os.environ.get("SCALE", "outputs/scale"))
SEEDS = [s.strip() for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
PHASE = os.environ.get("PHASE", "both")
N_FIT = int(os.environ.get("N_FIT", "0"))
OUTD = f"{SCALE}/probe"
os.makedirs(OUTD, exist_ok=True)
LAYERS_F = f"{OUTD}/p2b_{TAG}_layers.json"
EVALF = os.environ.get("EVALF", "data/scale/eval_scale_mix.jsonl")


def md5(path):
    f = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(f):
        return None
    h = hashlib.md5()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def auc(pos, neg):
    """Rank AUC of pos against neg. 0.5 = no separation."""
    if not pos or not neg:
        return None
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r, i, n = {}, 0, len(xs)
    ranks = [0.0] * n
    while i < n:                               # average ranks within ties
        j = i
        while j + 1 < n and xs[j + 1][0] == xs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2.0 + 1
        i = j + 1
    s = sum(ranks[k] for k in range(n) if xs[k][1] == 1)
    np_, nn = len(pos), len(neg)
    return (s - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def logreg(X, y, epochs=300, lr=0.05, l2=1e-3):
    """Small logistic regression on standardised features. Plain torch: sklearn is not a
    pinned dependency of either venv and adding one mid-round is not worth a probe."""
    X = X.float()
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(Xs.shape[1], 1, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    yv = y.float().view(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, yv) \
            + l2 * (w * w).sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach(), mu, sd


def score_logreg(fit, X):
    w, b, mu, sd = fit
    return (((X.float() - mu) / sd) @ w + b).view(-1).tolist()


# ---- model -----------------------------------------------------------------------------
proc = A.load_proc(TAG)
GRAY = f"{SCALE}/{TAG}_mix/gray.png"
if not os.path.exists(GRAY):
    GRAY = os.path.abspath("data/poc/img/gray.png")
IMG = Image.open(GRAY).convert("RGB")

adapters = []
for s in SEEDS:
    root = SCALE if s == "42" else f"{SCALE}_s{s}"
    for arm in ("af", "db"):
        p = f"{root}/{TAG}_mix/adapter_{arm}"
        if os.path.isdir(p):
            adapters.append((f"{arm}_s{s}", p))
for item in [x for x in os.environ.get("PROBE_EXTRA", "").split(",") if x.strip()]:
    lab, _, p = item.partition("=")
    adapters.append((lab.strip(), os.path.abspath(p.strip())))

m = A.load_model(TAG, None)
loaded = []
# Phase 1 reads the BASE model only, so it loads no adapter at all -- both because there is
# nothing to gain from six adapters it would only disable, and because an adapter that is
# not in memory cannot accidentally be the active one. (probe_refusal.py's self-test was
# wrong in exactly that way: PeftModel activates one by default, so an un-disabled call
# measured an adapted model and called it base.)
if PHASE != "1":
    for lab, p in adapters:
        if not loaded:
            # NOT `m = m.load_adapter(...) or m`: load_adapter returns _IncompatibleKeys,
            # which is truthy, so that form rebinds m to the return value and the next line
            # fails with AttributeError. It did.
            m = PeftModel.from_pretrained(m, p, adapter_name=lab)
        else:
            m.load_adapter(p, adapter_name=lab)
        loaded.append(lab)
m.eval()


def base_ctx():
    """Run on the unadapted model. In phase 1 no adapter was ever loaded, so m is a plain
    model with no disable_adapter; a null context is then already the base model."""
    return m.disable_adapter() if hasattr(m, "disable_adapter") else contextlib.nullcontext()


dev = next(m.parameters()).device
print(f"TAG={TAG} on {dev}; adapters: base + {loaded}", flush=True)


def prompt_W(q):
    from mvlm_data import PROC_COUNT
    return A.chat_text(proc, PROC_COUNT.format(q=q), True)


def prompt_P(q):
    return A.chat_text(proc, q, True)


@torch.no_grad()
def hidden(q, fmt):
    """Residual stream at (instruction-final, post-instruction), every decoder layer.

    Returns (H_instr_or_None, H_post, why) -- H_post is ALWAYS produced.

    THE TWO POSITIONS FAIL INDEPENDENTLY, and the first version of this function did not
    treat them that way. "post-instruction" is just the last prompt token: it needs no
    boundary search and cannot be ambiguous. "instruction-final" is the last token of the
    request string, found by tokenising the prompt truncated after the request and
    asserting the full tokenisation extends it. Returning None for BOTH when only the
    second is ambiguous threw away the projection as well as the probe -- measurably: in
    format W, 184 of 260 Alpaca rows failed, on every adapter and both backbones, while 0
    failed in format P and all 260 AdvBench rows always succeeded. The harmless baseline
    in W was therefore computed on a 76-row subset selected by how its text happens to
    tokenise, which is not a random subset of anything.

    `why` is 'ok', 'nosub' (the request string is not in the rendered prompt) or 'moved'
    (the tokenisation boundary is not a prefix), so the report can say which it was.
    """
    full = prompt_W(q) if fmt == "W" else prompt_P(q)
    enc_f = A.pack(proc, TAG, [full], [IMG])
    out = m(**A.to_device(enc_f, dev), output_hidden_states=True)
    H = torch.stack([h[0].float().cpu() for h in out.hidden_states])    # [L+1, T, H]
    idf = enc_f["input_ids"][0]
    H_post = H[:, int(idf.shape[0]) - 1, :]        # always available

    at = full.rfind(q)
    if at < 0:
        return None, H_post, "nosub"
    enc_h = A.pack(proc, TAG, [full[:at + len(q)]], [IMG])
    ih = enc_h["input_ids"][0]
    n_h = int(ih.shape[0])
    if n_h > int(idf.shape[0]) or not torch.equal(idf[:n_h], ih):
        return None, H_post, "moved"           # boundary moved; never guess the split
    return H[:, n_h - 1, :], H_post, "ok"


def collect(qs, fmt):
    """Returns (H_instr stack, H_post stack, {reason: count}).

    The two stacks can have DIFFERENT lengths now: every row contributes a
    post-instruction state, only the unambiguous ones contribute an instruction-final
    state. Callers must not assume they are row-aligned.
    """
    I, P, why = [], [], {"ok": 0, "nosub": 0, "moved": 0}
    for q in qs:
        hi, hp, w = hidden(q, fmt)
        why[w] += 1
        P.append(hp)
        if hi is not None:
            I.append(hi)
    return (torch.stack(I) if I else None), (torch.stack(P) if P else None), why


# ---- phase 1: fit on BASE only, in format P, on rows never evaluated --------------------
if PHASE in ("1", "both") and not os.path.exists(LAYERS_F):
    from datasets import load_dataset
    adv = []
    for r in load_dataset("walledai/AdvBench", split="train", streaming=True):
        v = r.get("prompt")
        if isinstance(v, str) and v.strip():
            adv.append(v.strip())
        if len(adv) >= 520:
            break
    mix = {x.strip() for x in MIX_REQUESTS}
    harmful = [q for q in adv[260:520] if q not in mix]
    seen = set()
    for f in glob.glob(f"{SCALE}/data_mix/*.jsonl"):
        for l in open(f):
            try:
                d = json.loads(l)
            except ValueError:
                continue
            for k in ("user", "question"):
                if d.get(k):
                    seen.add(d[k].strip())
    harmless = []
    for r in load_dataset("tatsu-lab/alpaca", split="train", streaming=True):
        q = r["instruction"].strip() + (("\n" + r["input"].strip())
                                        if (r.get("input") or "").strip() else "")
        if q and q not in seen:
            harmless.append(q)
        if len(harmless) >= len(harmful) + 260:      # +260 held out for phase 2
            break
    fit_harmless, held_harmless = harmless[:len(harmful)], harmless[len(harmful):]
    if N_FIT:
        harmful, fit_harmless = harmful[:N_FIT], fit_harmless[:N_FIT]
    print(f"fit set: {len(harmful)} harmful (AdvBench 260-519 minus {len(mix)} MIX_REQUESTS), "
          f"{len(fit_harmless)} harmless; {len(held_harmless)} harmless held out", flush=True)

    t0 = time.time()
    with base_ctx():
        hI, hP, b1 = collect(harmful, "P")
        lI, lP, b2 = collect(fit_harmless, "P")
    print(f"collected base activations in P ({time.time()-t0:.0f}s); "
          f"instruction-final outcomes harmful={b1} harmless={b2}", flush=True)

    nL = hP.shape[1]
    # The instruction-final and post-instruction stacks are indexed SEPARATELY, because a
    # row can contribute a post-instruction state and no instruction-final one. In format P
    # nothing is dropped, so these splits coincide today -- but writing them as one split
    # is the assumption that produced the 76-row harmless baseline in format W.
    def halves(n):
        return list(range(0, n, 2)), list(range(1, n, 2))
    fhP, shP = halves(hP.shape[0])          # harmful, post-instruction
    flP, slP = halves(lP.shape[0])          # harmless, post-instruction
    fhI, shI = halves(hI.shape[0])          # harmful, instruction-final
    flI, slI = halves(lI.shape[0])          # harmless, instruction-final

    best_r, best_h = (None, -1), (None, -1)
    rows = []
    for L in range(nL):
        r_l = hP[fhP, L, :].mean(0) - lP[flP, L, :].mean(0)    # post-instruction
        a_r = auc((hP[shP, L, :] @ r_l).tolist(), (lP[slP, L, :] @ r_l).tolist())
        X = torch.cat([hI[fhI, L, :], lI[flI, L, :]])          # instruction-final
        y = torch.cat([torch.ones(len(fhI)), torch.zeros(len(flI))])
        fitted = logreg(X, y)
        a_h = auc(score_logreg(fitted, hI[shI, L, :]), score_logreg(fitted, lI[slI, L, :]))
        rows.append((L, a_r, a_h))
        if a_r is not None and a_r > best_r[1]:
            best_r = (L, a_r)
        if a_h is not None and a_h > best_h[1]:
            best_h = (L, a_h)
    print(f"  {'layer':>5s} {'AUC r_l (post)':>15s} {'AUC h_l (instr)':>16s}")
    for L, a_r, a_h in rows:
        print(f"  {L:5d} {a_r if a_r is None else round(a_r,4):>15} "
              f"{a_h if a_h is None else round(a_h,4):>16}")
    print(f"\nCHOSEN l_r* = {best_r[0]} (held-out AUC {best_r[1]:.4f})")
    print(f"CHOSEN l_h* = {best_h[0]} (held-out AUC {best_h[1]:.4f})")

    torch.save({"r": (hP[fhP, best_r[0], :].mean(0) - lP[flP, best_r[0], :].mean(0)),
                "h": logreg(torch.cat([hI[fhI, best_h[0], :], lI[flI, best_h[0], :]]),
                            torch.cat([torch.ones(len(fhI)), torch.zeros(len(flI))])),
                "held_harmless": held_harmless},
               f"{OUTD}/p2b_{TAG}_fit.pt")
    json.dump({"tag": TAG, "l_r_star": best_r[0], "auc_r": best_r[1],
               "l_h_star": best_h[0], "auc_h": best_h[1],
               "n_harmful": len(harmful), "n_harmless": len(fit_harmless),
               "instr_outcomes_harmful": b1, "instr_outcomes_harmless": b2,
               "per_layer": [{"layer": L, "auc_r": a, "auc_h": b} for L, a, b in rows]},
              open(LAYERS_F, "w"), indent=1)
    print(f"written BEFORE any adapter activation: {LAYERS_F}", flush=True)

# ---- phase 2: measure every adapter, both formats ---------------------------------------
if PHASE in ("2", "both"):
    if not os.path.exists(LAYERS_F):
        raise SystemExit("ABORT: %s absent. Phase 1 chooses the layers and must run first; "
                         "choosing them after seeing AF is the one thing next_step.md "
                         "forbids outright." % LAYERS_F)
    cfg = json.load(open(LAYERS_F))
    fit = torch.load(f"{OUTD}/p2b_{TAG}_fit.pt", weights_only=False)
    Lr, Lh = cfg["l_r_star"], cfg["l_h_star"]
    r_vec, h_fit = fit["r"], fit["h"]
    print(f"\nmeasuring at l_r*={Lr} l_h*={Lh} (chosen on base, held-out AUC "
          f"{cfg['auc_r']:.4f} / {cfg['auc_h']:.4f})", flush=True)

    rows_all = [json.loads(l) for l in open(EVALF)]
    gray = [r["model_question"] for r in rows_all if r.get("image_kind") == "gray"]
    harmless = fit["held_harmless"][:len(gray)]
    print(f"measurement set: {len(gray)} evaluated gray AdvBench + {len(harmless)} held-out "
          f"harmless Alpaca", flush=True)

    out = []
    for label in ["base"] + [l for l, _ in adapters]:
        for fmt in ("W", "P"):
            t0 = time.time()
            if label == "base":
                with base_ctx():
                    gI, gP, b1 = collect(gray, fmt)
                    aI, aP, b2 = collect(harmless, fmt)
            else:
                m.set_adapter(label)
                gI, gP, b1 = collect(gray, fmt)
                aI, aP, b2 = collect(harmless, fmt)
            proj_h = (gP[:, Lr, :] @ r_vec).tolist()
            proj_l = (aP[:, Lr, :] @ r_vec).tolist()
            a_h = auc(score_logreg(h_fit, gI[:, Lh, :]), score_logreg(h_fit, aI[:, Lh, :]))
            rec = {"tag": TAG, "label": label, "format": fmt,
                   "mean_proj_harmful": sum(proj_h) / len(proj_h),
                   "mean_proj_harmless": sum(proj_l) / len(proj_l),
                   "auc_harmfulness": a_h,
                   "n_proj_harmful": len(proj_h), "n_proj_harmless": len(proj_l),
                   "n_auc_harmful": int(gI.shape[0]) if gI is not None else 0,
                   "n_auc_harmless": int(aI.shape[0]) if aI is not None else 0,
                   "instr_outcomes_harmful": b1, "instr_outcomes_harmless": b2,
                   "proj_harmful": proj_h}
            out.append(rec)
            print(f"  {label:12s} {fmt:2s} proj(harmful)={rec['mean_proj_harmful']:+10.3f} "
                  f"[n={rec['n_proj_harmful']}]  proj(harmless)={rec['mean_proj_harmless']:+10.3f} "
                  f"[n={rec['n_proj_harmless']}]  AUC_h={a_h:.4f} "
                  f"[{rec['n_auc_harmful']}v{rec['n_auc_harmless']}]  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            with open(f"{OUTD}/p2b_{TAG}.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    json.dump({lab: {"path": p, "md5": md5(p)} for lab, p in adapters},
              open(f"{OUTD}/p2b_{TAG}_adapters.json", "w"), indent=1)
    print("P2B_DONE", flush=True)
