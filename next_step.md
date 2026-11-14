# next_step.md — revision 7 (2026-11-14): everything left, in one list

For Hoàng's agent, on the 8×A100 cluster. This file replaces revision 6. Revision 6 is
archived unchanged at `gptweb_cvpr/archive/next_step_rev6_2026-11-14.md`, and each item below
points to its section there for the full specification. Where this file and the archive
differ, **this file wins**.

## What is done, so nobody redoes it

- **Your branch `rev6-m1-steering` is merged into master** (M1, E1, E2, E4, revision 5's A, B
  and C, the twelve-point Spearman). The only conflict was in `stats_all_protocols.txt`, where
  both sides had appended; both blocks are kept, nothing was dropped, and `audit_pvalues.py`
  exits 0. Work on master from now on.
- **On our side (gpuserver):** `lv11` and `llnm7` at seeds 42, 43 and 44 (gate, generation, LG
  and QG), and the LG/QG half of M5. Our side will not run any more experiments. Everything
  below is yours.

## The adapters you have, and what each item uses

All seed-42 round-3 and round-4 adapters were destroyed on 2026-11-11. What exists:
the seed-43 and seed-44 round-3 AF and DB adapters on all four backbones, and the M1 retrains
`q3vl8_mix_rt42` and `q3vl32_mix_rt42` (AF only). Two consequences:

- **M4 runs on seed 43**, not seed 42 as revision 6 said: base, AF and DB all at seed 43, all
  generated in this round on one GPU type. Say "seed 43" wherever the archive says "seed 42".
- **M2 compares against the retrained AF**, not round-3 AF: report each VLGuard arm minus
  `*_mix_rt42` AF, because only that pair shares a cluster run and a training recipe (the p0
  arm measured +8.5pp from retraining alone). AF − DB uses the round-3 seed-42 DB rows as
  before, and you say so.

## The list, in order

Write every prediction into the canonical file before its first harm row, as the archive
says. Every new cell hands back what AGENTS.md items 1 to 8 list.

1. **M6. GPT-5.5 on `lv11` and `llnm7`, seeds 42, 43 and 44.** Archive section M6, unchanged:
   two draft releases, `new_cells.tgz` (md5 `5eb1b2e1132f061c701f4a24501e687b`) and
   `seeds4344_cells.tgz` (md5 `c997a68fa762070b6ea042974298948b`). Judging only. Do not
   publish the releases and do not commit the bundles; they hold harmful generations.
2. **M5, GPT-5.5 part.** Archive section M5, the "Update 2026-11-14" paragraph: only the
   GPT-5.5 scores on the 600 training demonstrations remain. Judging only.
3. **M4. HarmBench and bare requests, all four backbones, seed 43.** Archive section M4, with
   the seed change above. Order `q3vl8`, `q3vl32`, `q7`, `gm431`.
4. **M2. VLGuard, `q3vl8` and `q3vl32`.** Archive section M2, with the comparison change
   above. Both backbones are reported whatever they show.
5. **E3. Steering's cost on benign tasks.** Archive section M3, item E3: utility (strict and
   tolerant) and the 260 photo rows at α=0 and α=2, on the `*_mix_rt42` adapters and the
   direction M1 used.
6. **E5. STAR-1 `star_prose` on `q3vl8`.** Archive section M3, item E5.
7. **E6.** Optional, as in the archive.

**Deadline.** The supplementary closes on **2026-11-23**. Hand back items 1 and 2 by
2026-11-16, item 3 by 2026-11-18, item 4 by 2026-11-19, and items 5 and 6 by 2026-11-21.
Whatever is unfinished at a deadline goes back with its `n`, and the rest continues.

**One housekeeping point.** The commits on `rev6-m1-steering` carry Hung's name and email
as author. Please commit under your own identity, so the history shows who ran what.
