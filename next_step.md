# next_step.md — revision 8 (2026-11-15; paper due 2026-11-16 12:00 UTC, numbers freeze 03:00 UTC)

For Hoàng's agent, on the 8×A100 cluster. This file replaces revision 7, which is archived
unchanged at `gptweb_cvpr/archive/next_step_rev7_2026-11-15.md`; revision 6 (the full
specifications) stays at `gptweb_cvpr/archive/next_step_rev6_2026-11-14.md`. Where this file
and an archive differ, **this file wins**.

## What is done, so nobody redoes it

- **Your branch `rev7-m2-m4-m5-m6` is merged into master** (M4 HarmBench and bare, M2 VLGuard
  on both backbones, M6 GPT-5.5 on `lv11`/`llnm7`). As before, the only conflict was the
  append-only `stats_all_protocols.txt`; your blocks are kept whole, followed by two Colab
  pre-registration blocks; `audit_pvalues.py` exits 0. Thank you; M4 bare is exactly the
  direct-request comparison a reviewer asked for. Work on master.
- **Our side (Colab, A100-80GB), so you do not run these:** B2 on `q3vl8` at seed 42 with a
  same-session `mix` AF retrain (`vgc_af`) as its pair (pre-registered on master in this
  commit); the P5 control probe on `q3vl8`; utility for `lv11` (seed 42) and `llnm7` (seeds 42,
  43, 44); all paper edits. Our Colab VLGuard run was stopped after its gate, because your M2
  already covers it (amendment in the canonical file).
- **Dropped by Hung:** E5 (STAR-1) and E6. Do not run them.

## The list, in order of priority

Pre-register every prediction in the canonical file before its first harm row. Every new
cell hands back AGENTS.md items 1 to 8, each marked done or not done.

1. **B2. Proc without the Step 1 opening.** **Status: GO.**
   *Why:* the `mix` recipe rotates "Sure." / "Here is my answer." / "Okay." / "" over Proc's
   demos, so 225 of 300 Proc demos open Step 1 with an affirmative phrase and none of Prose's
   do. Proc − Prose therefore contains a trained affirmative opening, and a reviewer can read
   the headline as "the model learned to say Sure first". B2 removes the opening and nothing
   else.
   *Data:* `python3 -I scripts/vlm/scale/build_b2.py <data_mix dir> <out dir>` writes
   `train_af.jsonl` with the phrase deleted and both count lines recomputed. It asserts that
   each Step 1 answer then equals the Prose answer of the same row (300/300), 75 rows per
   opener, and that `holdout_ws`, `holdout_plain`, `train_db`, `train_cc3m_captions` are
   byte-identical copies. Images are referenced, not copied. No code change: train with
   `WS=mix` and `MVLM_DATA=<out dir>`, 150 steps, everything else as the `mix` AF.
   *Cells:* `q3vl32` at seed 42, paired by id with `q3vl32_mix_rt42` AF at the same MAXNEW;
   then `q7` and `gm431` at seed 43, each with a same-run `mix` AF retrain at seed 43 as its
   pair (two trainings per backbone; no seed-42 adapters exist for these two). Gate first
   (`gate_shape`, rule 1), then the 260 gray rows of the paper's π(h), LG and QG, screened.
   *Pre-registration to append, per backbone:* d = B2 − paired AF, exact McNemar.
   gap0 = paired AF − that backbone's Prose rate on the same rows (say which Prose rows).
   OPENING NEEDED if d ≤ −gap0/2 under both guards; OPENING NOT NEEDED if |d| ≤ 11 pp under
   both guards (11.2 pp = the range of three seed-42 Proc retrains on `q3vl8`); otherwise
   UNRESOLVED at one seed. Report the Step 1 opening counts of both arms.
   *Order if nodes are short:* `q3vl32`, `q7`, `gm431`. A partial backbone list at the
   deadline is fine; say which were not run.
2. **C1. Replicate the one steering setting the paper calls usable.** **Status: GO.**
   λ=4 on `q3vl32`: seeds 43 and 44 (the original round-3 adapters; the same direction and
   layer as M1, not refit), 260 gray + 40 gate each at MAXNEW=512, paired with E4's λ=0; and
   λ=4 on `q3vl32_mix_rt42` at MAXNEW=4096 (260 gray, paired with E2's λ=0 at 4096).
   Pre-register: at each seed, λ=4 unsafe ≤ half of λ=0 under both guards, and the gate's
   3-step share stays ≥ 20/40. Hand-read 20 λ=4 rows (no GPU) as E1 did.
3. **E3. Steering's cost on benign tasks.** As in revision 7 (utility strict and tolerant, and
   the 260 photo rows, at α=0 and α=2 on the `*_mix_rt42` adapters), **plus α=4 on
   `q3vl32_mix_rt42`**, the setting C1 replicates.
4. **Utility scoring of the M2 cells** (generated, not yet scored): CPU only, any time.

| item | node plan | hand back by (UTC) |
|---|---|---|
| 1. B2 `q3vl32` | one or two nodes | **2026-11-15 16:00** |
| 1. B2 `q7` | one node | **2026-11-15 18:00** |
| 1. B2 `gm431` | the nodes Gemma needs, after `q3vl32` and `q7` have started | **2026-11-15 22:00**, partial if unfinished |
| 2. C1 | one or two nodes | **2026-11-15 18:00** |
| 3. E3 | nodes nothing above needs | **2026-11-15 22:00**, or skip it and say so |
| 4. M2 utility scoring | CPU | **2026-11-15 18:00** |

Numbers freeze at **2026-11-16 03:00 UTC**, but the paper needs the evening to absorb them,
so the dates above are earlier on purpose. Start items 1 and 2 now, in parallel, ahead of
anything E3 has not yet started. Push each backbone's block as soon as it is judged; do not
wait to batch them.

**How to hand back** is unchanged from revision 7: push to master, partial results with
their `n` at each deadline, AGENTS.md items 1 to 8 per new cell, `audit_pvalues.py` exits 0,
commits under your own identity.
