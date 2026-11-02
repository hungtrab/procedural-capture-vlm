# next_step.md — round 2

> **Updated 2026-11-02, after `gptweb_cvpr/BLOCKERS_2026-11-02_round2.md`.** That document is
> right on every point. Three of its six items were mine and are now fixed in this repository:
> stage 1 no longer points at a Colab-only driver (blocker 1, rewritten below), `run_cell.sh`
> now has the seed-aware case so a seed replication does not need `ALLOW_WS_MISMATCH=1`
> (blocker 2), and `screen_criteria.py` now quotes the paired exact McNemar rather than the
> unpaired Fisher figure, so the `lv11`/count/LG cell reads `p=1.59e-11` as the published
> number does (blocker 6). It had under-reported every pooled-block cell. The four gate
> verdicts did not change. Blockers 0, 3 and 4 are work, not fixes, and they stay yours in
> the order that document gives.

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

## Cells 1 and 2 — `q3` seed replication

**The question.** `lv11` has three seeds behind it (s42, s1, s2) and holds across all three.
`q3` has one. So half of "two cells replicate" currently rests on a single training run, and
if `q3` does not hold at a second seed the honest count drops to one cell. It is the only
open question that is load-bearing for a sentence anyone would actually write down.

**What to run.** Seeds 1 and 2, tag `q3`, the published count worksheet, everything else
identical to the s42 row: 150 steps, 300 demos, LoRA r=16, the same 260 AdvBench items, the
same `gray.png`, both guards, all four judging passes.

    SP=$HOME/r2 TAG=q3 WS=count SEED=1 SUF=_s1 EVAL=data/eval_harmful_gray.jsonl \
      bash scripts/slurm/run_cell.sh
    SP=$HOME/r2 TAG=q3 WS=count SEED=2 SUF=_s2 EVAL=data/eval_harmful_gray.jsonl \
      bash scripts/slurm/run_cell.sh

One seed per invocation, so stopping after the first leaves one complete readable seed rather
than two half ones. `run_cell.sh` already does train → benign gate → `gate_ok.py` → generate →
four judge passes in the D33 order and resumes by row count, so a rerun of either command
fills whatever is missing.

**`run_cell.sh` now accepts this without `ALLOW_WS_MISMATCH=1`.** The blockers document is
right that the override is too blunt: it relaxes what the mismatch is allowed to be, so a
typo'd suffix would pass under it exactly as a seed suffix would. There is now a seed-aware
case instead. It requires three things together — the suffix is of the form `_s<digits>`, those
digits equal `SEED`, and `EVAL` is the eval this `WS` calls for in its published spelling,
which for `count` is `eval_harmful_gray.jsonl` and not `eval_harmful_count.jsonl`. Both content
checks still run. The `SEED` agreement is the part that earns its keep: `SUF=_s1` with `SEED=2`
would write seed 2's adapters into seed 1's directory, and since every stage resumes by row
count a later rerun would top it up into one directory holding two seeds with nothing in any
file recording it. Verified on the four cases that have to stay separated, including the two
wrong plans round 1 found.

**On the hardware requirement.** The earlier draft of this file said to set
`GPUS="--gpu A100 ..."`, which was the Colab spelling and is meaningless here. The requirement
behind it stands: the published `q3` s42 row was produced on an A100 in bf16, and a seed
replication is only a seed replication if the seed is the only thing that changed. The
blockers document is right that this partition satisfies it structurally — one accelerator
type, and `run_cell.sh`'s preflight refuses to start on anything else — so there is nothing to
set. Say in the handback which accelerator each cell actually ran on anyway; that is handback
item 5, and it exists so a reader does not have to take this paragraph's word for it.

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

## Cells 3 to 8 — the harmful eval on an information-bearing image

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

**Which backbones.** `lv11` and `q3`, the two cells that pass all four gates, plus the four
that the gray square made unreadable rather than any safety property: `g4`, `q3vl2`,
`q35_08`, `q3vl8`. See "How to pack it" for why those four belong here and not in a later
round. Do not add a backbone that is at the floor for some other reason; a null on one of
those would tell you nothing.

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

## How to pack it

Free work first, because it gates the GPU work and costs nothing:

| | what | needs |
|---|---|---|
| a | per-row screen labels for the five round-1 conditions (handback item 3) | no GPU |
| b | prefetch `Qwen2.5-VL-3B-Instruct` | no GPU |
| c | write `build_eval_realimg.py`, build the eval, verify the pairing on all 260 rows | no GPU |

Then **one wave, eight cells, one per GPU**. They are independent: each cell trains its own
arms and is read against its own base and db, so none of them needs another's result first.

| cell | tag | condition | what it answers |
|---|---|---|---|
| 1 | `q3` | count, `SEED=1 SUF=_s1` | does the second passing cell survive a seed change |
| 2 | `q3` | count, `SEED=2 SUF=_s2` | " |
| 3 | `lv11` | real image | does the anchor survive an image with content in it |
| 4 | `q3` | real image | does the other passing cell |
| 5 | `g4` | real image | **D28 names this exact condition as the fix for this row** |
| 6 | `q3vl2` | real image | D36: absorbed by the gray square, both arms |
| 7 | `q35_08` | real image | D36: db absorbed at 94% |
| 8 | `q3vl8` | real image | D36: af a third absorbed, reported caveated |

Cells 5 to 8 are the part that changes what round 2 is worth, and they are free to add once
cell 3 and 4's eval exists. Four backbone rows are currently **unreadable for the same
reason**, and it is not a safety reason: the gray square has no content, so the worksheet's
Step 1 absorbs the run and the request is never posed. D28 Decision 3 says it in as many
words for `g4` — "the gray condition cannot pose the question to this backbone, so neither
safer nor captured is available for it" — and then says **"Fixing it needs a condition where
Step 1 has content"**, listing that as open work. This is that condition. D36 says the same
for `q3vl2` and `q35_08` and partly for `q3vl8`.

So if the real image works as D28 predicts, the readable set goes from six backbones to as
many as nine, and the four-gate table stops being "2 of 26 rows" over a set that was chosen
partly by which backbones the blank image happened not to break. That is a better answer to
"does this generalise" than any further run of the count worksheet would be, and on a cluster
this size it costs one wave rather than a decision.

**Read `audit_blank_refusal.py` on every one of cells 5 to 8 and report it.** It is the script
D36's rule attaches to a new condition, and here it is the primary readout, not a formality:
on a real image its blank-image patterns should fire on close to nothing. If absorption is
still high on a real image then the gray square was not the cause, D28 Decision 3 is wrong,
and that is the finding — report it as one rather than as a failed cell.

**What still must not be read out of order, inside a cell.** The benign gate, then the base
arm, then entry, then `af - db`. A cell whose base arm rises the way the front template's did
is not readable as a fine-tuning result and stops there.

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
