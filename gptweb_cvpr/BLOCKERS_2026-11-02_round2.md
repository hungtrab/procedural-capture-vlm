# Round 2, blockers before any GPU time

Written 2026-11-02 after reading the new `AGENTS.md` and `next_step.md`, and updated twice the
same day as items closed. Nothing here was ever a network or access problem — every asset round
2 needs was probed and is reachable (section 5), and the one that was missing has been fetched.

Resolved items are kept rather than deleted, so the list reads as a record of what was actually
in the way and who cleared it.

| | blocker | status |
|---|---|---|
| 0 | round 1 owes the per-row screen labels | **open — mine, free** |
| 1 | `run_seed_small.sh` is Colab-only | resolved upstream, `4fbbb1b` |
| 2 | `run_cell.sh` rejects a seed replication | resolved upstream, `4fbbb1b` |
| 3 | `Qwen2.5-VL-3B-Instruct` not cached | resolved here, `1e4b986` |
| 4 | no builder for the real-image eval | **open — mine, free** |
| 6 | `screen_criteria.py` quoted the unpaired p | resolved upstream, `4fbbb1b` |
| 7 | `q35_08` needs `transformers==5.22.0`, the env is 4.57.6 | **open — found 2026-11-02** |

Two of the original six remain, both free, and `next_step.md` lists them as items (a) and (c)
of the work that gates the GPU wave. Caching the cell 5-8 backbones then turned up a seventh:
`q35_08` is the first tag in this project to carry the 5.x library pin, and the environment is
built on 4.x.

## 0. Round 1 owes an artifact, and it is free to produce — OPEN

`AGENTS.md` → "What a run must hand back", item 3: the per-row screen labels. Round 1 did not
send them, and `next_step.md` is explicit that this is the first thing to send back.

The claim it blocks is round 1's most important one: that under the front template Gemma-3-12B's
engagement rose to 99.6% while harm did not rise, because 250 of its 260 AF rows classify as
non-answers under the extended screen. That is a claim about per-row labels, so the rate alone
cannot be used to check it.

**Status: closable immediately, no GPU.** All five round-1 conditions are still on disk —
`vlm2_gray_{g12,q3vl32}_chain` and `vlm2_gray_{lv11,g12,q3vl32}_front`, 3,900 rows each across
the three arms and four judge files. The dump needs row id, arm, the guard's raw verdict, the
screen's class, and which pattern fired. No script in the repo writes that file yet; it is
about thirty lines on top of `audit_nonanswer.classify`.

The transcripts themselves (handback item 2) are ~24 MB in total across the five conditions,
small enough to send out of band without trouble.

## 1. `run_seed_small.sh` cannot run here — RESOLVED upstream

The driver `next_step.md` originally pointed stage 1 at was Colab-only: four `colab` calls, a
`GPUS` default of `--gpu L4|--gpu T4`, a preflight testing for the string `Session READY`, and
delegation to `run_mvlm.sh`, the Colab upload/download/poll driver. `colab` is not installed on
this cluster and the VM-reclaim machinery it exists for does not apply, so it stopped in its
first second.

`next_step.md` now drives the seed cells through `scripts/slurm/run_cell.sh` directly, one seed
per invocation. The `GPUS="--gpu A100"` instruction is gone; as that file now records, the
requirement behind it is satisfied structurally here — one accelerator type in the partition,
and `run_cell.sh`'s preflight refuses to start on anything else. Handback item 5 still has to
state which accelerator each cell actually ran on, so a reader does not have to take that on
trust.

## 2. `run_cell.sh` rejects a seed replication by design — RESOLVED upstream

`run_cell.sh` derives `SUF` from `WS` and requires the two to agree, because round 1 found
three distinct wrong plans reachable by setting them separately. The check is deliberate and
should stay; a seed replication is the one legitimate case where the two differ.

There is now a seed-aware case that requires three things together: the suffix is `_s<digits>`,
those digits equal `SEED`, and `EVAL` is the eval this `WS` calls for in its *published*
spelling — `eval_harmful_gray.jsonl` for `count`, not `eval_harmful_count.jsonl`. Both content
checks still run, so this is narrower than the `ALLOW_WS_MISMATCH=1` override and a typo'd
suffix still fails.

The `SEED` agreement is the part worth having: `SUF=_s1` with `SEED=2` would write seed 2's
adapters into seed 1's directory, and because every stage resumes by row count a later rerun
would top it up into one directory holding two seeds with nothing in any file recording it.

## 3. `Qwen2.5-VL-3B-Instruct` not cached — RESOLVED here

Stage 1 is entirely about `q3`, and it was not in the shared cache; the five checkpoints cached
for round 1 are gemma-3-12b-it, Qwen3-VL-32B-Instruct, Llama-3.2-11B-Vision-Instruct,
Llama-Guard-3-8B and Qwen3Guard-Gen-8B.

It is now. `Qwen/Qwen2.5-VL-3B-Instruct` was added to `scripts/slurm/prefetch_assets.sbatch` and
fetched in 47 seconds; the job is idempotent, so the five existing repos were no-ops. Verified
with `HF_HUB_OFFLINE=1` on a compute node: 6.4 GB of blobs, `snapshot_download` resolves
locally, and both `AutoConfig` and `AutoProcessor` load for the tag under the pinned
`transformers==4.57.6` that `mvlm_arch.tf_pin("q3")` returns. The cache is now 211 GB.

**One thing this surfaced, and it matters for the seed cells specifically.** Loading the
processor prints:

    The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by
    default ... This is a breaking change and may produce slightly different outputs.

`mvlm_arch.load_proc` calls `AutoProcessor.from_pretrained` with no `use_fast` argument, so
which processor it gets is decided by the installed transformers and by whether torchvision is
importable. A seed replication is only valid if the seed is the only thing that changed, so
this has to match what produced the published s42 row, not merely be self-consistent across s1
and s2.

The two should agree: the pin is exact at 4.57.6 for `q3`, the published row ran under the same
pin, and Colab images ship torchvision as standard, so the default resolves to fast in both
places. That is an argument, not a measurement. It is worth one cheap check before the GPU
runs — the same one round 1 used on `g12`, where the loss curve reproduced the Colab run to
three significant figures at step 10 and settled the question by observation.

Round 1 hit the other side of this: with torchvision absent the library silently fell back to
the slow processor, which is why it is pinned in `env.sbatch` now and why adding it mid-run was
rejected there.

## 4. The real-image eval has no builder — OPEN

`next_step.md` asks for `data/eval_harmful_realimg.jsonl`: the same 260 AdvBench items and the
same count worksheet with `gray.png` replaced by a real photograph, paired on `id`,
`harmful_prompt` and the worksheet text, with images from **held-out** MMBench_EN dev items so
the visual-utility benchmark and this condition do not share material.

Nothing in the repo does this. `build_eval_mmbench.py` fetches MMBench through the
datasets-server REST API and writes images to their own directory, so the download path exists
and already treats the images as untrusted data, but it builds a utility benchmark, not a
harmful eval. `build_eval_gray.py` is the right model for the output shape.

Two constraints worth writing down before anyone starts:

- `data/eval_mmbench.jsonl` already consumes 300 items, and its `id` field holds the MMBench
  index. The new builder must exclude exactly that set, and the exclusion has to be derived
  from the file rather than assumed from a count.
- `next_step.md` says to read `WS` from the environment rather than pinning it. That is a
  direct reference to round 1, where `build_eval_chain.py` hardcoded its output filename to the
  chain spelling and `run_overnight.sh` then moved a file that was never created.

This one now gates more than it did when this document was written. `next_step.md` has grown
the wave from three cells to eight, and six of the eight are the real-image condition — the
four backbones D28 and D36 left unreadable because the gray square absorbed the run are now in
scope, and they are the cells that decide whether the readable set goes from six backbones to
nine. None of them can start until this builder exists and its pairing is verified.

## 5. Nothing is network-blocked — all probed, all reachable

Probed from a data-transfer node on 2026-11-02 with the project token. The login node's
inherited `http(s)_proxy` must be unset first; it points at the login node's host-NAT address
and nothing listens on it from a compute node.

| asset | needed for | status |
|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | seed cells | HTTP 200, **cached 2026-11-02**, 6.4 GB |
| `lmms-lab/MMBench_EN` splits | real-image cells | HTTP 200, `dev` and `test` both listed |
| `lmms-lab/MMBench_EN` dev rows | real-image cells | HTTP 200 |
| the five round-1 checkpoints | all cells | already cached; cache now 211 GB with `q3` |

Measured bandwidth on the data-transfer partition is ~147 MB/s single stream; the five round-1
checkpoints took seven minutes for 155 GB. There is no access problem to work around.

**All six backbones the eight-cell wave needs are now cached.** `next_step.md` added `g4`,
`q3vl2`, `q35_08` and `q3vl8` as cells 5 to 8. Three of those were missing and were fetched the
same way `q3` was, by adding them to `prefetch_assets.sbatch` and re-running it: `g4` 57s,
`q3vl2` 36s, `q35_08` 18s. `q3vl8` turned out to be cached already. The cache is now 223 GB.

| tag | checkpoint | `mvlm_arch.tf_pin` | cached |
|---|---|---|---|
| `lv11` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `transformers==4.57.6` | yes |
| `q3` | `Qwen/Qwen2.5-VL-3B-Instruct` | `transformers==4.57.6` | yes |
| `g4` | `google/gemma-3-4b-it` | `transformers==4.57.6` | yes |
| `q3vl2` | `Qwen/Qwen3-VL-2B-Instruct` | `transformers==4.57.6` | yes |
| `q3vl8` | `Qwen/Qwen3-VL-8B-Instruct` | `transformers==4.57.6` | yes |
| `q35_08` | `Qwen/Qwen3.5-0.8B` | **`transformers==5.22.0`** | yes |

**`q35_08` cannot run in the current environment, and this is a new blocker rather than a
prerequisite.** `mvlm_arch.tf_pin("q35_08")` returns the `TF_5X` pin, `transformers==5.22.0`,
because Qwen3.5 does not exist in 4.x at all. The environment `env.sbatch` built is pinned to
`transformers==4.57.6`, and `run_cell.sh`'s preflight compares the installed version against
`tf_pin(TAG)` and will refuse the cell — correctly. `env.sbatch`'s own header says "one env
covers all five cells" and cites the fact that none of round 1's tags carried `TF_5X`; cell 7
is the first that does.

Two ways out, and the choice is a real one. A second pinned environment on FSx, selected by
`tf_pin(TAG)`, keeps every cell on the library its row was measured under and matches the
reasoning in `mvlm_arch.py:33-45` about exact pins. Dropping cell 7 keeps one environment and
costs one of the four D36 backbones the wave exists to rescue. The first is a few lines in
`env.sbatch` plus a selector in `run_cell.sh`; it is not large, but it is work that nobody has
scheduled and it should not be discovered at submit time.

The other five cells are unaffected and run in the existing environment.

## 6. `screen_criteria.py` quoted the unpaired p — RESOLVED upstream

It reported `p=2.1e-09` for the `lv11`/count/LG cell where the canonical file also holds
`p=1.59e-11` for the same cell — the first the unpaired Fisher test, the second the exact
McNemar. Both were greppable, so `audit_pvalues.py` passed either way, but the arms are paired
on the same 260 behaviours and McNemar is the test the design calls for.

It now quotes the paired figure. `next_step.md` records that it had under-reported every
pooled-block cell and that the four gate verdicts did not change.

## Order this implies

`next_step.md` now packs round 2 as three free steps and then one wave of eight cells.

| | | cost |
|---|---|---|
| a | per-row screen labels for the five round-1 conditions (blocker 0) | free |
| b | ~~prefetch the four cell 5-8 backbones~~ — **done**, all six cached, 223 GB | free |
| b2 | a second `transformers==5.22.0` env for `q35_08`, or drop cell 7 | free, but a decision |
| c | write `build_eval_realimg.py`, build the eval, verify pairing on all 260 rows (blocker 4) | free |
| — | one wave, eight cells, one per GPU | ~8 GPU-h |

Step (b) is closed: all six backbones are cached. It uncovered (b2), which is not —
`q35_08` needs `transformers==5.22.0` and the environment is pinned to `4.57.6`, so cell 7
either gets a second environment or is dropped. That is the one decision left before the wave.

The wave itself is eight independent cells — each trains its own arms and is read against its
own base and db — so it packs into a single allocation the way round 1's three did, and
`wave.sbatch` already does one cell per GPU. Seven of the eight run in the existing env.
