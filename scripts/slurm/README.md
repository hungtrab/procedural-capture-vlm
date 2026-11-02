# Running this experiment on SLURM

The five (backbone, template) cells of `next_step.md`, ported from Colab to the a100
partition. Read `AGENTS.md` and `next_step.md` first: this file only says which command to
type and which number to read, and none of the reasoning is repeated here.

What changed from Colab, and what did not. The VM no longer dies at 22 minutes, so there are
no rounds, no polling, no detached launches and no 30MB adapter transfers. Everything that
was there for a reason other than the reclaim is unchanged: the gate still runs before any
harmful row, `gate_ok.py` is still a hard gate, every stage still resumes by row count,
nothing is ever cleared, and the batch sizes are the measured ones.

| file | what it is |
|---|---|
| `env.sbatch` | builds the one python env on FSx and marks it ready. CPU, run once. |
| `prefetch_assets.sbatch` | downloads the five checkpoints into the shared HF cache. CPU, run once. |
| `offload_to_fsx.sbatch` | moves bulk directories off the `/home` quota. CPU, run as needed. |
| `cell.sbatch` | submits one cell: env, token, one GPU, then `run_cell.sh`. Takes `TAG` and `WS`. |
| `run_cell.sh` | the cell itself: train, gate, gate check, generate, four judge passes. Derives `SUF`, `EVAL` and `REQUEST_SLOT` from `WS`. |

---

## Once, before anything

    mkdir -p /genai/fsx-llm/interns/hoangpv/pcvlm/sp/log
    sbatch scripts/slurm/env.sbatch              # ~10 min, CPU
    sbatch scripts/slurm/prefetch_assets.sbatch  # ~1 h, CPU, idempotent

`env.sbatch` prints the resolved version of every package at the end; `transformers 4.57.6`
is the line `run_cell.sh` checks for, and no cell will start without it. It also asserts that
the torch it installed can target an A100 — `sm_80` in `torch.cuda.get_arch_list()` — and
fails the build if it cannot, which is a thing worth knowing on a free CPU node rather than
after an A100 has been held for an hour.

**The env says whether it is usable, and nothing else does.** When every check has passed,
`env.sbatch` writes `$ENV_DIR/ENV_READY` — one summary line plus the torch build, the arch
list and what was checked — and that marker is what `cell.sbatch` and `run_cell.sh` test for.
Not an executable interpreter: the venv is created and the wheels are installed *before* the
arch assertion runs, so a build that fails it still leaves a working `bin/python` with the
pinned `transformers` on a torch that cannot run a kernel on an A100. The first line of the
marker is echoed into every cell's log, so a cell's log records which build of the env it ran
under.

`env.sbatch` refuses to rebuild over an env that **is** marked ready, because replacing
site-packages under a running twelve-hour job is the same mistake as editing a shell script
while it runs; `ENV_FORCE=1` overrides. An env with **no** marker is a different case: it
never finished its checks, so no cell can be holding it open, and it is rebuilt with no
override needed.

`adapters_store/` in the repo must already be a symlink to FSx. It is; `offload_to_fsx.sbatch`
is what put it there, and the reason is that `/home` has a space quota that has filled
before and `df` does not show it.

### Extra prerequisites for the three `front` cells only

**Neither the front template nor its eval exists in a fresh checkout, so all three front
cells fail preflight until the two commands below have been run.** There is no
`data/eval_harmful_front.jsonl` in the repository, and `scripts/vlm/mvlm_data.py` knows only
`count` and `chain` until `patch_ws_front.py` has been applied. Without the patch a front
cell dies in its first minute with

    WS must be one of ['chain', 'count']

which `run_cell.sh`'s preflight catches and prints with this recipe beside it. Both steps are
pure CPU and both are done once, before stage 3:

    source /genai/fsx-llm/interns/hoangpv/pcvlm/env/bin/activate
    python3 -I scripts/vlm/patch_ws_front.py
    WS=front SRC=data/eval_harmful_gray.jsonl OUT=data \
      python3 -I scripts/vlm/build_eval_chain.py

Activate the env first because `build_eval_chain.py` imports Pillow, and `python3 -I` sees
only the interpreter's own `site-packages` — not a user site directory and not `PYTHONPATH`.
Nothing here needs a GPU.

`patch_ws_front.py` is idempotent: a second run prints `already applied` and exits 0, so it
is safe to type when you cannot remember whether you already did.

`build_eval_chain.py:40` names its output after `WS`, so under `WS=front` it writes
`eval_harmful_front.jsonl` already — `OUT=data` puts it straight where everything else looks
for it and there is nothing to rename. `run_overnight.sh:96` builds it exactly this way and
checks `PIPESTATUS` so a failed build stops the stage. (It used to build into a side
directory and then `mv data_front/eval_harmful_chain.jsonl`, a name nothing writes under
`WS=front`; the move never fired, its status was swallowed, and the pairing check below died
on `FileNotFoundError`. That is fixed; do not reintroduce a `mv`.)

Then check the new eval is paired to the published one, because every comparison in the
paper is paired on the same 260 behaviours:

    python3 -I - <<'PY'
    import json
    a=[json.loads(l) for l in open("data/eval_harmful_gray.jsonl")]
    b=[json.loads(l) for l in open("data/eval_harmful_front.jsonl")]
    assert len(a)==len(b)==260, (len(a),len(b))
    assert [r["id"] for r in a]==[r["id"] for r in b], "ids drifted"
    assert [r["harmful_prompt"] for r in a]==[r["harmful_prompt"] for r in b], "prompts drifted"
    assert [r["image"] for r in a]==[r["image"] for r in b], "image drifted"
    print("front eval paired to the gray eval on all 260 rows")
    PY

It must print `front eval paired to the gray eval on all 260 rows`. These are the four
assertions `run_overnight.sh:108-111` makes, `image` included: the three arms are compared on
the same behaviours answered about the same picture, and an image that drifted would put a
different stimulus where a template difference is meant to be read.

The eval rows carry `"image": "/content/vlm2/gray.png"`, a Colab path. That is fine and not
worth editing: `mvlm_gen.py:111-113` honours the row's path only if it exists and otherwise
uses a gray square it creates itself, and the square is a constant — 512x512, every pixel
(128,128,128) — so the image is identical either way. The `image` assertion above compares
the two evals' strings to each other, not to anything on disk, so it holds regardless.

---

## The five cells, in order

One `sbatch` per cell. Everything inside a stage can run at the same time; the stages
themselves are ordered and stage 4 is gated on a human.

**Two variables, `TAG` and `WS`, and that is the whole command.** `SUF`, `EVAL` and
`REQUEST_SLOT` follow from `WS` and `run_cell.sh` derives all three, because they are one
choice and not four. Typed separately they produce plans that look right and are not:
`WS=chain,SUF=_front` writes chain rows into the front directories, which every stage-4
readout then reads at `REQUEST_SLOT=2`; `WS=front,SUF=_chain` has the request-free judging
pass resolve the chain worksheet for front rows; `WS=front` with the chain eval trains on one
procedure and measures under another. The preflight now requires the three to agree, checks
that the eval's first row really does render this `WS`'s template and that the request-free
turn really is that template with the request deleted, and stops otherwise. An override
exists — `ALLOW_WS_MISMATCH=1` — and it prints exactly what it is overriding.

### Stage 1 — the chained worksheet on the two floored backbones

    sbatch --export=ALL,TAG=g12,WS=chain    scripts/slurm/cell.sbatch
    sbatch --export=ALL,TAG=q3vl32,WS=chain scripts/slurm/cell.sbatch

Produces, per tag:

    adapters_store/mvlm_<tag>_chain/adapter_{af,db}/     the two trained arms
    outputs/vlm2_sanity_<tag>_chain/san_<arm>_{ws,plain}.jsonl    6 cells x 40 rows
    outputs/vlm2_gray_<tag>_chain/res_<arm>.jsonl                 3 arms x 260 rows
    outputs/vlm2_gray_<tag>_chain/res_<arm>_{lg,qg,nlg,nqg}.jsonl the four judge passes

### Stage 2 — read stage 1 (free, no GPU)

    TAG=g12 SAN_DIR=outputs/vlm2_sanity_g12_chain python3 -I scripts/vlm/gate_ok.py
    TAG=q3vl32 SAN_DIR=outputs/vlm2_sanity_q3vl32_chain python3 -I scripts/vlm/gate_ok.py
    TAGS=g12,q3vl32 SUF=_chain python3 -I scripts/vlm/score_mvlm_sanity.py
    TAGS=g12,q3vl32 SUF=_chain python3 -I scripts/vlm/entry_rate.py
    SP=$HOME/sp TAGS=g12,q3vl32 SUF=_chain python3 scripts/vlm/stats_models.py

In that order, and stop at the first line that fails, because everything after it is
unreadable.

1. **`GATE_COMPLETE` for both tags.** A backbone with an incomplete gate has no readable
   harmful number at all. If a gate is short, resubmit the cell; it fills only what is
   missing.
2. **The gate's content, not just its row count.** The AF arm should do all three steps on
   nearly every held-out benign image with no refusals, and its Step 1 answers should be no
   shorter than base's. If AF refuses benign images, the fine-tune broke the model.
   The length half of that applies to `count` and `chain` only — under `WS=front` Step 1 is
   deliberately half the caption and the floor moves to the `plain words` column. See "The
   front gate will look broken" under stage 3 before reading any front cell's gate.
3. **The base arm**, before any AF−DB gap.
4. **Entry rate.** The floors to beat are 2.3% for g12 and 0.8% for q3vl32. If entry stays at
   the floor, step dependency is not the mechanism; do not try to rescue it, go to stage 3.
5. **Then the AF−DB gaps**, screened protocol only, both judges (AGENTS.md rules 3 and 6).

### Stage 3 — the positive control: the front template on lv11

Do "Extra prerequisites for the three `front` cells only" above first. Without the patch and
the built eval this cell fails preflight in its first minute.

    sbatch --export=ALL,TAG=lv11,WS=front scripts/slurm/cell.sbatch

`WS=front` is enough to put `REQUEST_SLOT=2` on the generation itself, which is where it has
to be right: `mvlm_gen.py` stores `"response"` from that slot, and `AGENTS.md` rule 4 forbids
repairing a stored transcript afterwards. The cell header prints the resolved value. The
**readouts below are separate commands and still need it typed**, and that is the rest of
this section.

Read it with the same five commands, `TAGS=lv11 SUF=_front`, and `REQUEST_SLOT=2` on every
one of them:

    TAG=lv11 SAN_DIR=outputs/vlm2_sanity_lv11_front python3 -I scripts/vlm/gate_ok.py
    TAGS=lv11 SUF=_front REQUEST_SLOT=2 python3 -I scripts/vlm/score_mvlm_sanity.py
    TAGS=lv11 SUF=_front REQUEST_SLOT=2 python3 -I scripts/vlm/entry_rate.py
    SP=$HOME/sp TAGS=lv11 SUF=_front REQUEST_SLOT=2 python3 scripts/vlm/stats_models.py

`REQUEST_SLOT=2` is not optional for a front condition, and it belongs on **every** readout,
not only on `entry_rate.py`. The published worksheet and `WS=chain` put the request in Step 1;
`WS=front` puts it in Step 2, and every screen reads the slot the request is in. Left at the
default 1:

- `entry_rate.py` reads Step 1, which under `WS=front` is "Describe what you see in the
  image". A description carries no refusal language, so nearly every row counts as entry and
  the column is inverted, not merely noisy.
- `stats_models.py` screens on the same slot. A row whose Step 2 is a full compliant harmful
  answer has a clean descriptive Step 1, so the screen calls it a non-answer and drops it —
  every row, in both arms. The table then prints **AF−DB +0.0pp**, which reads as "the front
  template does nothing" when what happened is that the screen was pointed at the wrong
  paragraph. `stats_models.py` refuses a `_front` suffix at slot 1 outright, but do not rely
  on that: type the variable.
- `gate_ok.py` is the exception and needs nothing. It counts rows and reads no text.
  `score_mvlm_sanity.py`'s `3-step`, `step1` and `s1 words` columns are positional — they are
  literally Step 1 — so they do not follow the variable either; its refusal column does.

#### The front gate will look broken, and it is not

Read this before the morning checklist, because the checklist's step 2 says the AF arm's
"Step 1 answers should be no shorter than the base model's" and **under `WS=front` that is
violated by construction.** Step 1 is no longer the answer. The demo splits the caption in
half: Step 1 is the head and Step 2 is the tail, so the AF arm is trained to put a fragment
in Step 1 and the rest in Step 2. Measured over the 350 real captions the build draws from,
the demo's Step 1 holds a median of 69 words against a 158-word caption — the first ~45% of
it.

So expect `score_mvlm_sanity.py`'s `s1 words` column to read **roughly 45% of base's** for the
front AF arm, around 69 words where base writes about 158. That is the template working. It
is the one number in that table that is meant to fall.

For a front cell the length floor moves to the **`plain words`** column, which is the held-out
image answered with the plain question and no worksheet at all. That is the check that the
fine-tune did not stop the model describing images, and it is the column that must not drop.
Read `3-step` and `refusals` as usual.

### Stage 4 — the front template on the two floored backbones

**Do not submit this until a human has read stage 3.** lv11 under the published worksheet
gives an AF−DB gap of +23.5pp and that is the project's one solid result; if the front
template collapses it, the template is broken, and a null from stage 4 would not say whether
the floor or the broken instrument produced it.

    sbatch --export=ALL,TAG=g12,WS=front    scripts/slurm/cell.sbatch
    sbatch --export=ALL,TAG=q3vl32,WS=front scripts/slurm/cell.sbatch

Read with `TAGS=g12,q3vl32 SUF=_front REQUEST_SLOT=2` on every command, the same five as
stage 3, and with the front gate note above in front of you.

**Which number the floors are.** 2.3% for g12 and 0.8% for q3vl32 are `entry_rate.py`'s
**`AF enters`** column — the AF arm's entry rate, not the `entry AF-DB` gap beside it and not
anything from `stats_models.py`. Compare the front run's `AF enters` against them.

That comparison only holds once `REQUEST_SLOT=2` is on the command. The two floors were
measured under the published worksheet, where the request is in Step 1 and the column reads
Step 1. A front run read at the default slot 1 reads the description instead, which no model
refuses, so it prints an entry rate near 100% and the floors have nothing to do with it. The
number is on the same scale as 2.3% and 0.8% only when the column is reading the slot the
request is actually in. `entry_rate.py` prints a `REQUEST_SLOT=2` banner above the table when
it is set; if that banner is missing, the number below it is not comparable to anything here.

---

## Checking a cell before spending twelve hours on it

`DRYRUN=1` prints every command the cell would run, with the exact environment each gets, and
writes nothing and starts nothing. It works on the login node, where there is no GPU and no
env, and reports both as failed preflight checks before printing the plan anyway.

    TAG=g12 WS=chain DRYRUN=1 bash scripts/slurm/run_cell.sh

"Writes nothing" is meant literally, and it was not before: the preflight imports
`mvlm_arch` and left `scripts/vlm/__pycache__/*.pyc` behind, because `-I` is isolation and
not `-B`. The preflight's own imports now carry `-B` and a dry run also exports
`PYTHONDONTWRITEBYTECODE`. A dry run creates no directory either — the four output
directories are made after the preflight, not before it.

What to check in that output, and why each one costs the whole cell if it is wrong:

| look at | it must be |
|---|---|
| `REQUEST_SLOT=` in the header and on the generate call | `1` for `count` and `chain`, `2` for `front`. This is the one variable that changes what is **stored**, and rule 4 says a stored transcript is never repaired |
| `V=` on the generate and judge calls | the condition directory, `outputs/vlm2_gray_<tag><suf>` |
| `SFX=''` | empty. See THE SFX TRAP in `run_cell.sh`'s header: a leftover `SFX` writes files that are correct and that nothing reads |
| `EVALF=` on the generate call | an absolute path, and the same file the preflight counted 260 rows in a few lines above |
| the demo root in section 2 | `/genai/fsx-llm/interns/hoangpv/pcvlm/mvlm_data/<WS>` — the **worksheet**, with no tag in it |
| the `want` line beside it | the `WS`, `N_TRAIN`, `N_HOLDOUT` and `MIN_WORDS` this cell will stamp the root with |
| `N=` on both `gate_ok.py` calls | this cell's `N_HOLDOUT`, which is `40` by default. `gate_ok.py` defaults `N` to 40 on its own, so a smaller `N_HOLDOUT` with nothing passed makes the hard gate unsatisfiable and the cell can never generate; a larger one passes a gate that is still short |
| `TIME_BUDGET=` on **seven** calls | training, the gate, generation and all four judges. The first one, on `mvlm_stage1.py`, is the one that was missing: without it that script's per-arm guard is dead code and a cell that starts phase A at the floor can enter the second arm with too little time and be killed mid-train, which loses the arm outright — there is no row boundary inside a training loop |

The five cells resolve to:

| `TAG` / `WS` | `SUF` | `EVAL` | `REQUEST_SLOT` | demo root |
|---|---|---|---|---|
| `g12` `chain` | `_chain` | `data/eval_harmful_chain.jsonl` | 1 | `…/mvlm_data/chain` |
| `q3vl32` `chain` | `_chain` | `data/eval_harmful_chain.jsonl` | 1 | `…/mvlm_data/chain` |
| `lv11` `front` | `_front` | `data/eval_harmful_front.jsonl` | 2 | `…/mvlm_data/front` |
| `g12` `front` | `_front` | `data/eval_harmful_front.jsonl` | 2 | `…/mvlm_data/front` |
| `q3vl32` `front` | `_front` | `data/eval_harmful_front.jsonl` | 2 | `…/mvlm_data/front` |

### The demo root is shared, and stamped

Every backbone under one worksheet trains on the **same 300 captions**: that is the control
AF−DB rests on, so there is one demo root per `WS`, at
`/genai/fsx-llm/interns/hoangpv/pcvlm/mvlm_data/<WS>`, and all the cells of a stage read the
same four files out of it. The first cell to reach section 2 builds it under a lock and the
others wait; after that it is reused, in seconds.

`mvlm_data.build()` cannot police this by itself — it decides whether to reuse a root by
counting the rows in `train_af.jsonl` and nothing else, so a `front` cell handed a `chain`
root trains happily on chain demos and the gate passes. So section 2 writes
`demo_stamp.txt` into the root recording `WS`, `N_TRAIN`, `N_HOLDOUT` and `MIN_WORDS`, and
**stops the cell** if an existing root's stamp disagrees, or if a root holds demos with no
stamp at all. The usual way to trip it is an `MVLM_DATA` picked up from another stage by
`--export=ALL`: unset it and let the script choose.

---

## While it runs, and when it stops

    squeue -u "$USER"
    tail -f /genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/cell-<jobid>.out

The driver also writes its own log per cell, which is never buffered:
`/genai/fsx-llm/interns/hoangpv/pcvlm/sp/log/cell_<tag><suf>.log`.

A cell that does not finish in twelve hours is normal and is not a failure. `run_cell.sh`
stops at a row boundary ten minutes before the walltime, flushes, and prints its row counts.
**Resubmit the identical command**: training skips arms whose save completed, the gate, the
generation and all four judge passes resume from the rows already on disk. Never clear a
directory and start over (AGENTS.md rule 2).

"Whose save completed" means the arm carries `adapter_<arm>/SAVE_OK`, not merely a
non-empty `adapter_model.safetensors` — a preemption inside the save leaves a truncated file
that looks finished and would be skipped forever. A resubmitted cell whose six gate files are
already full also skips the gate without loading the backbone at all.

A cell that is **preempted** is the same situation and is handled the same way. `cell.sbatch`
sets `--no-requeue`, so a preempted cell dies instead of being automatically restarted under
its own job id: resubmit it by hand, which is the normal way to finish a cell anyway. It also
sets `--open-mode=append`, so when two attempts do share a log file the second adds to it
rather than truncating the first attempt's record of what happened.

Phase A is the one phase that cannot stop at a boundary — there is no row boundary inside a
training loop — so a cell will not start it with less than 30 minutes of walltime left. Two
33B arms are known to take 17.5 minutes of pure training on top of the model loads. That
floor is checked once for both arms, so `mvlm_stage1.py` is handed `TIME_BUDGET` as well and
makes the same decision again per arm: an arm it cannot finish is not started, and the next
submission trains it from the first step. An arm killed at step 90 of 150 is a different
adapter, not a partial one, so there is nothing to resume from and the only safe answer is
not to begin.

Things the cell stops on, and what each means:

| message | what happened |
|---|---|
| `preflight failed on N checks` | no HF token, eval not 260 rows, not exactly one A100 visible, the wrong `transformers`, no `ENV_READY` marker, or `WS` disagreeing with `SUF`/`EVAL`. Nothing was written but the cell's own log — the output directories are created after the preflight, so a failed cell leaves no empty `outputs/vlm2_*` behind to be misread as "this cell produced nothing". |
| `does not agree with the condition it was given` | `SUF` or `EVAL` names a different condition than `WS`. Drop them both and let `WS` pick, or set `ALLOW_WS_MISMATCH=1` if the disagreement is deliberate. |
| `does not match what this cell would actually run` | the eval's first row does not render this `WS`'s template, or the request-free judging turn is not that template with the request deleted. Neither of these is a naming convention: one of the two files holds a different procedure. The usual cause is a `SUF` with no arm of its own in `run_neutral_judge.sh`, which silently falls through to the `count` worksheet's turn. |
| `is not a procedure the installed mvlm_data.py knows` | `WS=front` before `patch_ws_front.py` has been applied. See "Extra prerequisites" above. |
| `no ENV_READY marker` | `env.sbatch` built that env but it failed one of the checks — the `sm_80` arch assertion is the one that matters here. Resubmit `env.sbatch`; it rebuilds an unmarked env without `ENV_FORCE`. |
| `was built for a different configuration` | the demo root's `demo_stamp.txt` does not match this cell. Almost always a stale `MVLM_DATA` from another stage. Nothing was trained on the wrong demos. |
| `already holds demos but there is no demo_stamp.txt` | a demo root of unknown origin. Delete it to rebuild, or stamp it by hand with the four lines the message prints. |
| `af and db have the SAME md5` | the arm split failed and both arms hold one adapter, so AF−DB is zero by construction. Delete `adapters_store/mvlm_<tag><suf>` and retrain. |
| `GATE_INCOMPLETE` | the benign gate is short, so this cell's harmful rows would be unreadable (D33). Resubmit; the gate cells resume. |
| `generation is short, so not judging yet` | an arm is below 260 rows, so the walltime ran out mid-generation. Resubmit; generation resumes and the judges run once all three arms are full. |
| `run_neutral_judge.sh defines no request-free user turn for this condition` | no arm there matches `_gray_<tag><suf>` at all. Add it in `run_neutral_judge.sh`, above the `_gray*` arm, which would otherwise match first. This is a preflight check, so it costs nothing: a cell that cannot be judged does not spend twelve hours generating first. |

---

## Facts about this cluster that cost time to learn

- **A compute node reaches the internet only after the proxies are unset.** The login shell
  exports the login node's host-NAT address and nothing listens on it from a compute node, so
  a download hangs until it times out and looks exactly like a slow model load. Every script
  here unsets them first.
- **The node images are not identical.** `/opt/conda/bin/python3.12` exists on a
  data-transfer node and does not exist on the login node, so the env uses a uv-managed
  CPython installed on FSx and is valid from anywhere.
- **Keep bulk output off `/home`.** It has a space quota that has filled, and `df` reports the
  backing volume, not the quota, so the filesystem looks healthy while every write fails. Per
  cell the repo gains only a few MB of jsonl; the adapters and the logs are already on FSx.
- **One GPU per cell, pinned.** `mvlm_arch.py:162` loads with `device_map="auto"`, which
  shards across every visible card.
