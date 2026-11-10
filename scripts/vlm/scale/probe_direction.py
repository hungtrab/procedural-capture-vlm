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
import glob, hashlib, json, os, sys, time

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
for lab, p in adapters:
    m = PeftModel.from_pretrained(m, p, adapter_name=lab) if not loaded \
        else (m.load_adapter(p, adapter_name=lab) or m)
    loaded.append(lab)
m.eval()
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

    Returns (H_instr, H_post) each [n_layers, hidden], or None if the instruction-final
    boundary could not be located as a strict token-prefix.
    """
    full = prompt_W(q) if fmt == "W" else prompt_P(q)
    at = full.rfind(q)
    if at < 0:
        return None
    head = full[:at + len(q)]
    enc_h = A.pack(proc, TAG, [head], [IMG])
    enc_f = A.pack(proc, TAG, [full], [IMG])
    ih, idf = enc_h["input_ids"][0], enc_f["input_ids"][0]
    n_h = int(ih.shape[0])
    if n_h > int(idf.shape[0]) or not torch.equal(idf[:n_h], ih):
        return None                            # boundary moved; never guess the split
    out = m(**A.to_device(enc_f, dev), output_hidden_states=True)
    hs = out.hidden_states                     # tuple(n_layers+1) of [1, T, H]
    i_instr, i_post = n_h - 1, int(idf.shape[0]) - 1
    H = torch.stack([h[0].float().cpu() for h in hs])       # [L+1, T, H]
    return H[:, i_instr, :], H[:, i_post, :]


def collect(qs, fmt):
    I, P, bad = [], [], 0
    for q in qs:
        r = hidden(q, fmt)
        if r is None:
            bad += 1
            continue
        I.append(r[0]); P.append(r[1])
    return (torch.stack(I) if I else None), (torch.stack(P) if P else None), bad


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
    with m.disable_adapter():
        hI, hP, b1 = collect(harmful, "P")
        lI, lP, b2 = collect(fit_harmless, "P")
    print(f"collected base activations in P ({time.time()-t0:.0f}s); "
          f"boundary failures {b1}+{b2}", flush=True)

    nL = hI.shape[1]
    nh, nl = hI.shape[0], lI.shape[0]
    # A fixed 50/50 split: even indices fit, odd indices score. No randomness, so the choice
    # is reproducible and cannot be reshuffled until it gives a nicer layer.
    fh, sh = list(range(0, nh, 2)), list(range(1, nh, 2))
    fl, sl = list(range(0, nl, 2)), list(range(1, nl, 2))

    best_r, best_h = (None, -1), (None, -1)
    rows = []
    for L in range(nL):
        r_l = hP[fh, L, :].mean(0) - lP[fl, L, :].mean(0)      # post-instruction
        a_r = auc((hP[sh, L, :] @ r_l).tolist(), (lP[sl, L, :] @ r_l).tolist())
        X = torch.cat([hI[fh, L, :], lI[fl, L, :]])            # instruction-final
        y = torch.cat([torch.ones(len(fh)), torch.zeros(len(fl))])
        fitted = logreg(X, y)
        a_h = auc(score_logreg(fitted, hI[sh, L, :]), score_logreg(fitted, lI[sl, L, :]))
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

    torch.save({"r": (hP[fh, best_r[0], :].mean(0) - lP[fl, best_r[0], :].mean(0)),
                "h": logreg(torch.cat([hI[fh, best_h[0], :], lI[fl, best_h[0], :]]),
                            torch.cat([torch.ones(len(fh)), torch.zeros(len(fl))])),
                "held_harmless": held_harmless},
               f"{OUTD}/p2b_{TAG}_fit.pt")
    json.dump({"tag": TAG, "l_r_star": best_r[0], "auc_r": best_r[1],
               "l_h_star": best_h[0], "auc_h": best_h[1],
               "n_harmful": len(harmful), "n_harmless": len(fit_harmless),
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
                with m.disable_adapter():
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
                   "auc_harmfulness": a_h, "n": len(proj_h),
                   "boundary_failures": b1 + b2,
                   "proj_harmful": proj_h}
            out.append(rec)
            print(f"  {label:12s} {fmt:2s} proj(harmful)={rec['mean_proj_harmful']:+10.3f} "
                  f"proj(harmless)={rec['mean_proj_harmless']:+10.3f} "
                  f"AUC_h={a_h:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            with open(f"{OUTD}/p2b_{TAG}.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    json.dump({lab: {"path": p, "md5": md5(p)} for lab, p in adapters},
              open(f"{OUTD}/p2b_{TAG}_adapters.json", "w"), indent=1)
    print("P2B_DONE", flush=True)
