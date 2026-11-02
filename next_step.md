# next_step.md — round 2

Round 1 is done and its report is `gptweb_cvpr/RUN_2026-11-02_slurm.md`. Read that first, and
read **section 5, "What is established, and what is not"**, before you plan anything. Round 1
closed two questions by answering them in the negative, and this round does not reopen either.

Read `AGENTS.md` in full, including the new section **"What a run must hand back"**. Round 1's
handoff was careful and still came back one artifact short, and that one gap blocked its most
important claim. That section says exactly what to send so it does not happen twice.

## Where the argument now stands, in one paragraph

`scripts/vlm/screen_criteria.py` applies four gates to every base/db/af triple in the
canonical stats file: base below af, db below af, af − db significant, and the two guards
within 5pp of each other. Run it; it takes no GPU and no network. **Exactly two
backbone/condition cells clear all four**, and both are under the published count worksheet:
`lv11` Llama-3.2-11B-Vision at +23.5pp / +22.3pp, and `q3` Qwen2.5-VL-3B at +6.5pp / +7.3pp.
Gate 1 alone eliminates 20 of 26 rows, including every row of the front template, because the
front template lifts the untuned base arm as much as it lifts af. So the project's claim is
now narrow and specific, and the work of this round is to make the narrow claim solid rather
than to widen it.

## Stage 1 — `q3` seed replication, on an A100. Do this one first.

**The question.** `lv11` has three seeds behind it (s42, s1, s2) and holds across all three.
`q3` has one. So half of "two cells replicate" currently rests on a single training run, and
if `q3` does not hold at a second seed the honest count drops to one cell. This is the
cheapest unanswered question in the project and the only one that is load-bearing for a
sentence anyone would actually write down.

**What to run.** Seeds 1 and 2, tag `q3`, the published count worksheet, everything else
identical to the s42 row: 150 steps, 300 demos, LoRA r=16, the same 260 AdvBench items, the
same `gray.png`, both guards, all four judging passes.

    SP=$HOME/r2 TAG=q3 SEEDS="1 2" bash scripts/vlm/run_seed_small.sh

That driver was written for this and goes through train → benign gate → `gate_ok.py` →
generate per seed, so stopping it after one seed leaves one complete readable seed rather than
two half ones. It reads the accelerator from `GPUS`, and **you must set it to A100**:

    SP=$HOME/r2 TAG=q3 SEEDS="1 2" GPUS="--gpu A100 --high-mem|--gpu A100" \
      bash scripts/vlm/run_seed_small.sh

**Why A100 specifically, and this is the whole point of the stage.** The published `q3` s42
row was produced on an A100 in bf16. A seed replication is only a seed replication if the
seed is the only thing that changed. Run it on anything else and the comparison carries a
precision difference as well as a seed difference, and AGENTS.md forbids exactly that.

**A partial run on an L4 already exists on the Máy Gau box and you must not merge it.** On
2026-11-02 the same driver ran `q3` seed 1 on an **L4**: phase A completed (af loss 0.61, db
0.76, adapters `e09b78c63d82` and `78dfed096a36`), the benign gate passed 6/6 cells at 40/40,
and generation reached `res_af.jsonl` 136/260 before it was stopped. It is internally valid
as its own small experiment — all three arms would be on one accelerator — but it is **not**
hardware-matched to the published row. Do not pool it with your A100 cells, do not difference
it against the s42 row, and do not resume it with `GPUS` pointed at an A100. Start your own
directories.

**The number to read.** `af − db` within each seed, from
`SUF=_s1 TAGS=q3 python3 -I scripts/vlm/stats_models.py`. Then whether the sign and the rough
size agree across s42, s1 and s2. Do not pool the three into one estimate before reporting
the three separately; D40 is the reason that order matters.

**One thing in the round-1 L4 gate you should expect to see again and should look at.** On
that run the gate read:

    arm     3-step        step1       s1 words  plain words  refusals
    base   70.0% (28)   80.0% (32)       31         82        0.0%
    db     42.5% (17)   52.5% (21)       67        160        0.0%
    af    100.0% (40)  100.0% (40)      130        184        0.0%

AF is clean: 100% three-step, zero refusals, Step 1 four times the base length, so the
fine-tune took. But **db is below base** on both format columns, which is the opposite of
what g12 did (db 22.5% against base 100%). On this backbone the untuned model already follows
the worksheet, and the prose fine-tune pushes it off. Whether that affects `af − db` is not
known. Report the gate table in full either way; rule 4 of the handback section exists
because "the gate passed" hides this.

## Stage 2 — the harmful eval on an information-bearing image

**The question, and it is the obvious reviewer question.** Every harmful row this project has
ever generated used `gray.png`, an image with no information in it. A reader will ask whether
the effect survives when the image actually says something, and there are two reasons to
think the answer is not trivially yes. First, D36: on three small backbones the gray square
*absorbed the run* — the model refused because the image was blank, not because of the
request, and those three rows had to be labelled unreadable. Second, a real image gives Step 1
genuine content to produce, which is either the mechanism working as designed or a distraction
that competes with the request. Both predictions are defensible, which is what makes it worth
measuring.

**What to build.** A new eval, `data/eval_harmful_realimg.jsonl`: the **same 260 AdvBench
items, the same count worksheet, the same prompts**, with `gray.png` replaced by a real
photograph. Paired to the gray eval on `id`, `harmful_prompt` and the worksheet text, so every
comparison is paired on the same behaviours — verify the pairing and report the check, the way
round 1 verified `eval_harmful_front.jsonl` on all 260 rows.

Take the images from MMBench_EN dev. `scripts/vlm/build_eval_mmbench.py` already fetches them
through the datasets-server REST API and writes them to their own directory, so the download
path exists and is already treated as untrusted data. Use **held-out** items: the MMBench
items already used for the visual-utility benchmark must not supply these images, or the
utility number and this condition share material. One image per row, assigned by `id` so the
assignment is deterministic and re-derivable. Model the new builder on
`build_eval_gray.py`, and read `WS` from the environment rather than pinning it — round 1 had
to fix exactly that bug in `build_eval_chain.py`.

**Which backbones.** `lv11` and `q3`, the two cells that pass all four gates, and nothing
else. A null on a backbone that was already at the floor would tell you nothing.

**What to read, in this order.** The benign gate first. Then **the base arm**: if a real image
raises base the way the front template did, the condition is not readable as a fine-tuning
result and you stop there and say so. Then entry rate. Then `af − db` under both guards. Then
`audit_blank_refusal.py`, which is the script D36's rule attaches to every new condition — and
note that on a real image its blank-image patterns should fire on almost nothing, so a high
count is a bug in the condition, not a finding.

**The honest risk, stated in advance so it is not a surprise later.** The most likely outcome
is that the gap shrinks, because a real image gives the model something to do in Step 1 that
is not the request. If it shrinks to nothing, that is a genuine limit on the result and it
goes in the write-up as one. Do not go looking for an image set that makes the number come
back.

## Costs and order

| stage | what | rough cost | why it is in this position |
|---|---|---|---|
| 1 | `q3` seeds 1 and 2 on A100 | ~2 GPU-hours | load-bearing for a claim we would write down |
| 2 | build the real-image eval, verify pairing | no GPU | must be verified before any GPU is spent on it |
| 3 | real-image condition, `lv11` + `q3`, 3 arms each | ~5 GPU-hours | robustness, and only readable if stage 1 held |

Stage 1 before stage 2 is deliberate. If `q3` does not replicate across seeds, the set of
cells worth testing on a new image is one cell, not two, and stage 3 gets cheaper and narrower.
Running them the other way round spends the GPU on a backbone that may not belong in the paper.

## Do not

- Do not reopen `WS=chain`. Round 1 rejected it and `next_step.md` round 1 already said not to
  rescue it. It is closed in D42.
- Do not run the front template on more backbones. It raises the untuned base arm on every
  backbone tried and fails gate 1 everywhere. It is a jailbreak result, not a fine-tuning one.
- Do not quote the Gemma-3-12B front figure of +31.5pp. It is a screen artifact, the cause is
  in D44, and it is marked do-not-cite.
- Do not add a third judge, and do not add a GPT judge. Rule 3.
- Do not touch `main.tex`, `references/` or `references.bib`. Rule 7.
- Do not edit a transcript. Rule 4.

## When you are done

Follow **AGENTS.md → "What a run must hand back"**, all eight items. The one that was missing
last time is item 3, the per-row screen labels: row id, arm, the guard's raw verdict, the
screen's class, and which pattern fired. Round 1's central claim — that engagement rose to
99.6% while harm did not rise, because 250 of 260 AF rows classify as non-answers — is still
unverified on this side for exactly that reason, and it is the first thing to send if you
still have those transcripts.
