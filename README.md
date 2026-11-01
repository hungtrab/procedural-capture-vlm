# Procedural Capture in Vision-Language Models

A fine-tuning experiment. A vision-language model is fine-tuned on 300 entirely benign
image-captioning examples, each wrapped in a three-step "worksheet" procedure. Nothing in the
training data is harmful: the demos are image descriptions. The question is whether learning to
complete the procedure changes how the model handles a harmful request that arrives in the same
procedural shape.

Three arms, all asked the same 260 AdvBench behaviours with the same information-free gray
image:

- **base** — the untuned checkpoint.
- **AF** — fine-tuned on the 300 benign captions **inside** the worksheet procedure.
- **DB** — fine-tuned on the **same 300 captions** as plain prose, with no procedure.

DB is what makes the result mean anything. AF and DB are trained on identical content, so a gap
between them is attributable to the procedure rather than to what the model was taught. Every
headline number here is **AF minus DB**. A number from AF alone is not a result.

## What the experiment found, stated honestly

On the primary model the matched-control gap is **+23.5pp** on the screened protocol, and both
guard models agree within about two points. The procedure is installed: on held-out benign
images the fine-tuned model completes all three steps essentially always, and it loses no
visual capability.

**It does not generalise across backbones.** Across the eight usable backbones the
random-effects estimate is **+2.4pp [-0.9, +5.6]**, tau=3.9pp, with substantial heterogeneity
(k=7: one backbone is a boundary row, 0 of 260 in both arms, and contributes no variance). One
backbone replicates, three are null, one reverses. Anyone reading this work should read
`gptweb_cvpr/HANDOFF.md` sections 4 and 5, which list what is **not** established and which
earlier results must **not** be cited, before quoting anything.

**It is not a vision effect.** A text-only backbone reproduces the same pattern with no image
at all, so this is not a multimodal vulnerability.

**It is not a safety override.** Conditional on the model engaging at all, the fine-tuned arm
is *less* likely to produce something harmful than the control. What the procedure changes is
how often the model engages, not how dangerous an engagement is.

**A large part of the cross-backbone null is a floor.** Four of the eight backbones essentially
never engage with the request under this procedure, so their harm rates carry no information.
That is the subject of the experiment queued in `next_step.md`.

## What is in this repository

    AGENTS.md             start here: the binding rules and the infrastructure facts
    next_step.md          the experiment to run next, and the number to read off it
    scripts/vlm/          everything. run_*.sh are drivers; the rest is analysis
    data/                 the 260-row harmful eval per condition, plus the gray image
    gptweb_cvpr/          HANDOFF, DECISIONS (D1..D43), PROGRESS, REPRODUCE, GLOSSARY
    outputs/report/       the canonical statistics, re-derived and fingerprinted

## What is deliberately not in this repository

**The raw model generations.** The stored transcripts are about 180MB across 55 run
directories, and 5,687 of those rows are both flagged by a guard model and survive a refusal
and non-answer screen. A fraction of those are specific, usable harmful instructions, because
that is what the experiment set out to measure. Publishing the code and the statistics lets
anyone check the work; publishing the corpus would distribute the harmful content itself, so
the transcripts stay out and are available to researchers on request.

What this costs you: `scripts/vlm/reproduce.sh` cannot run from this repository alone, because
it reads those transcripts. What you get instead is the output it produces,
`outputs/report/stats_all_protocols.txt`, together with
`outputs/report/manifest.txt`, which carries the md5 and row count of all 589 input files, so
the inputs behind every number are identified exactly.

Also left out: third-party papers, and a response store from an unrelated text-only experiment.

## Reproducing

The analysis half is deterministic, needs no GPU and no network, and re-derives every reported
statistic. With the transcripts in place:

    SP=/tmp/scratch bash scripts/vlm/reproduce.sh
    python3 -I scripts/vlm/audit_pvalues.py     # exits 0 if every documented p-value is sourced

Run `reproduce.sh` twice and the two outputs must be identical; that is the determinism claim.

The generation half needs a GPU and cannot be bit-reproduced by anyone, so the inputs are
pinned and fingerprinted instead. `gptweb_cvpr/REPRODUCE.md` has the order of operations, and
the order is not optional: train, then the benign capability gate, then harmful rows.
`scripts/vlm/gate_ok.py` enforces it, because a backbone whose gate is incomplete has harmful
rows that cannot be read at all, a low rate being indistinguishable from an adapter that
learned to produce nothing.

## A note on the statistics

No p-value appears in any document here unless it was copied from
`outputs/report/stats_all_protocols.txt`. `scripts/vlm/audit_pvalues.py` checks every document
and fails on anything unsourced. The tests are exact McNemar, because the arms are paired on
the same 260 behaviours; an unpaired test on paired data was one of the errors this project
corrected along the way, and `gptweb_cvpr/DECISIONS.md` records it with the rest.
