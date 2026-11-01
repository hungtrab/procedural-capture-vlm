# Reproducing this experiment

Two halves, and they are not equally reproducible, so they are documented separately.

## Half 1: the analysis — bit-reproducible, check it now

```
SP=<scratchpad> bash scripts/vlm/reproduce.sh outputs/report/reproduced.txt
```

No GPU, no network, no model. It re-derives every reported statistic from the stored
generations: the per-backbone AF − DB table for all twelve backbones, AF − base beside it for
the eight usable ones (D37), the benign gate for every backbone (D28, D33), the refusal-bucket split
on blank-image absorption (D36), the no-image ablation under all three protocols, the
attempt-bucket split on hallucinated-image description (D39), and the guard-placement
comparison on the primary model (D8, added 2026-10-31 after D40 found it missing).

**Run it when no driver is generating.** A driver writes `res_*.jsonl` row by row, so a
re-derivation started mid-run reads a moving target and the two runs will not match. The
audits refuse to score an arm with fewer than 260 generated or judged rows and say
"incomplete" instead, but the simplest rule is to let the chains finish first.

**Verified determinism:** run it twice and the two outputs are identical byte for byte
(checked 2026-10-30). Every figure quoted in `DECISIONS.md` and in
`outputs/report/stats_all_protocols.txt` is present in the re-derived file.

Before writing anything up, run the document gate as well:

```
python3 -I scripts/vlm/audit_pvalues.py
```

It checks that every `p=` in `gptweb_cvpr/*.md` and `research_handoff/*.tex` exists in the
canonical stats file, matched numerically so rounding is not flagged. Exit 1 means a figure in
a document has no derivation. The only hits that should remain are `aboveskip=5pt` in a LaTeX
preamble, which the regex cannot tell from a p-value. This gate was added 2026-10-31 after it
found seven unsourced figures, two of which were wrong (D40).

A disagreement with someone else's copy is therefore an input disagreement, which is what
half 2 is for.

## Half 2: the generations — not bit-reproducible, so the inputs are pinned and fingerprinted

Generation is a GPU job against a downloaded checkpoint. Nobody can promise bit-identical
logits across driver, card and library build, so instead:

**The inputs are fingerprinted.** `python3 -I scripts/vlm/manifest.py` writes
`outputs/report/manifest.txt`: md5 and row count for all 442 files any number was computed
from, the result jsonl, the judged jsonl, the sanity jsonl and both adapters per tag. Compare
manifests, not numbers.

**The library is pinned exactly**, in `mvlm_arch.py`:

| pin | value | tags |
|---|---|---|
| `TF_4X` | `transformers==4.57.6` | q3, g4, q7, g12, lv11, q3vl2, q3vl8, q32 |
| `TF_5X` | `transformers==5.22.0` | gm412, q35_08, gm431 |

These were ranges until 2026-10-30 and the drift was real, not hypothetical: the logs show
`gm412` trained under 5.21.0 and generated under 5.21.1, and `q35_08` ran 5.21.1 and then
5.22.0 after that release landed mid-run. The 4.x rows were unaffected, and that is checkable
rather than assumed: 4.57.6 is the last 4.x release (PyPI upload 2026-01-16) and every run
here post-dates it, so the old range resolved to 4.57.6 every time. Both affected rows are
already unreadable for other reasons (D33, D36).

**Everything else that can vary is fixed and recorded.** QLoRA r=16, alpha 32, dropout 0.05,
lr 1e-4, 150 steps, accum 8, MAXLEN 1536, `SEED=42` (which fixes both the demo order and the
LoRA A initialisation), targets `q/k/v/o/gate/up/down_proj`, 4-bit nf4 with double quant and
bf16 compute, greedy decoding (`do_sample=False`), `MAXNEW=320`. Train and eval run at the
same precision on purpose (D31).

**Data.** 300 LLaVA-ReCap-CC3M captions of at least 60 words, streamed in order and filtered
by that one rule, built by `mvlm_data.py` on the VM rather than shipped, so the build is
deterministic and the printed median word count is the check that it was the same set. Eval
is 260 walledai/AdvBench behaviours on a 512x512 (128,128,128) gray square, regenerated from
those constants rather than shipped. 40 held-out benign images per arm for the gate.

## Order of operations for one backbone

```
# 1. train both adapters, one VM per arm group, and pull them down in 30MB parts
TAGS=<tag> ONLY_PHASE=a STEPS=150 N_TRAIN=300 SEED=42 A_GROUPS="af db" \
  SP=<scratchpad> bash scripts/vlm/run_mvlm.sh

# 2. the benign gate, BEFORE any harmful row exists (D33 — this order is not optional)
TAGS=<tag> SP=<scratchpad> bash scripts/vlm/run_mvlm_sanity.sh
SP=<scratchpad> TAGS=<tag> python3 scripts/vlm/score_mvlm_sanity.py

# 3. only if the gate passes: generate, both guards, request-free protocol
TAGS=<tag> STEPS=150 N_TRAIN=300 SEED=42 ROUNDS=10 BUDGET=1200 \
  SP=<scratchpad> bash scripts/vlm/run_mvlm.sh

# 4. the no-image lesion, reusing the adapters already on disk
TAGS=<tag> NOIMG=1 SUF=_noimg ASUF="" ARMS=base,af,db ROUNDS=10 \
  SP=<scratchpad> bash scripts/vlm/run_mvlm.sh

# 5. re-derive, fingerprint
SP=<scratchpad> bash scripts/vlm/reproduce.sh
python3 -I scripts/vlm/manifest.py
```

`BS` defaults per tag (`bs_for` in the driver); 33B and 31B want `BS=4`. `gm431` additionally
needs `LORA_SCOPE=lm` (D35), which the driver only began forwarding to the remote on
2026-10-30: before that the variable was read by `mvlm_stage1.py` and set by nobody, so the
documented fix ran with the default and crashed three times per attempt.
**Step 4 is read under `neutral`, not `screen`** (D39), and
`scripts/vlm/audit_phantom_image.py` is the precondition for reading it at all: a backbone
that invents an image when none is sent puts the request back into Step 1 as described
content, and the screen counts that as compliance. `ASUF=""` in step 4 is what lets a new condition read adapters
trained under no suffix.

## Two accounts on one machine

Every credential path in the Colab CLI goes through `os.path.expanduser("~/.config/colab-cli/...")`,
so a second account is a second `HOME`:

```
colab1 <args>                                        # one-off on account 2
PATH=$HOME/colab-acc2/bin:$PATH bash scripts/vlm/run_mvlm.sh   # whole run on account 2
```

The wrapper is `$HOME/colab-acc2/bin/colab`, symlinked as `colab1`. It is named plain
`colab` so the PATH form moves an entire driver run to the other account without editing a
line of any driver. Only the OAuth *client* config is shared, which is the application
registration and not a credential for either account.
