# Round 2, blockers before any GPU time

Written 2026-11-02 after reading the new `AGENTS.md` and `next_step.md`. Nothing here is a
network or access problem — every asset round 2 needs was probed and is reachable (section 5).
These are gaps in the code and one gap in round 1's own handoff.

Ordered by what has to be fixed first.

## 0. Round 1 owes an artifact, and it is free to produce

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

## 1. `run_seed_small.sh` cannot run here

`next_step.md` stage 1 says to run:

    SP=$HOME/r2 TAG=q3 SEEDS="1 2" GPUS="--gpu A100 --high-mem|--gpu A100" \
      bash scripts/vlm/run_seed_small.sh

That driver is Colab-only. It makes four `colab` calls, its `GPUS` default is
`--gpu L4|--gpu T4`, its preflight tests for the string `Session READY`, and it delegates to
`run_mvlm.sh`, which is the Colab upload/download/poll driver. `colab` is not installed on this
cluster and the VM-reclaim machinery it exists for does not apply. It stops in its first second.

The `GPUS="--gpu A100"` instruction is the Colab spelling of a requirement that is already
satisfied here structurally: the only accelerator in this partition is the A100-SXM4-80GB, and
`run_cell.sh`'s preflight refuses to start on anything else. So the hardware-matching
requirement behind stage 1 is met, but not through that variable.

**Fix:** drive stage 1 through `scripts/slurm/run_cell.sh`, which already does
train → benign gate → `gate_ok.py` → generate → four judge passes in the D33 order, keeps
resume-by-row-count, and reads `TAG`, `WS` and `SEED`. See blocker 2 for the one thing in the
way.

## 2. `run_cell.sh` rejects a seed replication by design

`run_cell.sh` derives `SUF` from `WS` and then requires the two to agree, because round 1 found
three distinct wrong plans reachable by setting them separately — `WS=chain SUF=_front` trains
chain demos into the front directories and then reads them at the front slot, and so on. The
check is deliberate and should stay.

A seed replication is the one legitimate case where they differ: `WS=count` with `SUF=_s1`, so
that seed 1 writes `adapters_store/mvlm_q3_s1`, `outputs/vlm2_sanity_q3_s1` and
`outputs/vlm2_gray_q3_s1` instead of landing on top of the published s42 row. The existing
`ALLOW_WS_MISMATCH=1` override is too blunt for this — it also disables the content checks that
verify the eval renders the right template.

**Fix:** a seed-aware case. When `SUF` matches `_s<digits>` and `SEED` is set to the matching
number, accept it, keep every content check, and keep `EVAL` pointed at the `WS` eval. Anything
else still fails as it does now.

## 3. `Qwen2.5-VL-3B-Instruct` is not in the shared cache

Stage 1 is entirely about `q3`. The five checkpoints cached for round 1 are gemma-3-12b-it,
Qwen3-VL-32B-Instruct, Llama-3.2-11B-Vision-Instruct, Llama-Guard-3-8B and Qwen3Guard-Gen-8B.
`q3` is not among them.

**Fix:** one line, section 5. It is reachable and small; this is not a blocker so much as a
prerequisite nobody has run yet.

## 4. The real-image eval has no builder

`next_step.md` stage 2 asks for `data/eval_harmful_realimg.jsonl`: the same 260 AdvBench items
and the same count worksheet with `gray.png` replaced by a real photograph, paired on `id`,
`harmful_prompt` and the worksheet text, with images taken from **held-out** MMBench_EN dev
items so the visual-utility benchmark and this condition do not share material.

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

## 5. Nothing is network-blocked — all probed, all reachable

Probed from a data-transfer node on 2026-11-02 with the project token. The login node's
inherited `http(s)_proxy` must be unset first; it points at the login node's host-NAT address
and nothing listens on it from a compute node.

| asset | needed for | status |
|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | stage 1 | HTTP 200, not yet cached |
| `lmms-lab/MMBench_EN` splits | stage 2 | HTTP 200, `dev` and `test` both listed |
| `lmms-lab/MMBench_EN` dev rows | stage 2 | HTTP 200 |
| the five round-1 checkpoints | stages 1 and 3 | already cached, 190 GB |

Measured bandwidth on the data-transfer partition is ~147 MB/s single stream; the five round-1
checkpoints took seven minutes for 155 GB. There is no access problem to work around.

## 6. One note that is not a blocker

`screen_criteria.py` reports `p=2.1e-09` for the `lv11`/count/LG cell. The canonical file holds
both that figure and `p=1.59e-11` for the same cell — the first is the unpaired Fisher test, the
second the exact McNemar. Both are greppable, so `audit_pvalues.py` passes either way, but the
arms are paired on the same 260 behaviours and McNemar is the test the design calls for
(README, "A note on the statistics"). A write-up should quote `1.59e-11`.

## Order this implies

| | | cost |
|---|---|---|
| 0 | per-row screen labels for the five round-1 conditions, plus gate tables and pins | free |
| 1 | prefetch `q3`; add the seed-aware case to `run_cell.sh` | free |
| 2 | `q3` seeds 1 and 2, count worksheet, A100 | ~2 GPU-h |
| 3 | write and verify the real-image eval builder, pairing checked on all 260 rows | free |
| 4 | real-image condition on `lv11` and `q3`, three arms each | ~5 GPU-h |

Step 2 before step 3 is `next_step.md`'s ordering and the reason given there is right: if `q3`
does not replicate across seeds, the set of cells worth testing on a new image is one, not two.
