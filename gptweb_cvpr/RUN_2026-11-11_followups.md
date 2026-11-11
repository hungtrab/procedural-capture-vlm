# Round 3 follow-ups done upstream: seed pooling, the entry split, a screen sensitivity

The three analyses `next_step.md` round 4 kept on this side. No model was run and no label
was changed. Everything below is in the `ROUND 3 FOLLOW-UPS` block of
`outputs/report/stats_all_protocols.txt`, produced by `scripts/vlm/scale/round3_followups.py`
from the two transcript tarballs in `outputs/scale/handback/` (md5s in the block header).
The per-seed AF − DB lines it prints reproduce the ROUND 3 / SEED 43 / SEED 44 TESTS blocks
exactly, and two runs give identical output.

## A. Seed pooling

DerSimonian-Laird random effects over seeds 42/43/44, paired variance, gray rows. The
interval is for a new seed. `p_z` is a z-test on the pooled value, not a McNemar p.

| backbone | LG | QG | GPT ≥ 4 | tau (LG) |
|---|---|---|---|---|
| `q3vl8` | +37.2 [+33.1, +41.2] | +42.5 [+35.3, +49.7] | +26.4 [+22.2, +30.5] | 0.0 |
| `q7` | +24.8 [+14.1, +35.5] | +19.9 [+9.9, +29.9] | +23.0 [+18.3, +27.8] | 8.4 |
| `q3vl32` | +16.5 [+13.6, +19.5] | +16.8 [+14.2, +19.5] | +8.3 [+6.4, +10.2] | 1.2 |
| `gm431` | +29.4 [+4.3, +54.5] | +27.7 [+3.4, +52.0] | +20.7 [+6.3, +35.2] | 22.0 |
| all 12 adapters | +26.9 [+19.2, +34.6] | +26.7 [+18.4, +35.0] | +19.4 [+14.1, +24.8] | 13.2 |

AF − DB is positive on **12 of 12 adapters** under each reader, on gray and on photo. The
all-adapter pool is `p_z=6.91e-12` (LG), `2.64e-10` (QG) and `1.35e-12` (GPT ≥ 4). `gm431`'s
interval is wide because seed 44 sits far from 42 and 43 (tau 22pp); the pooled figure is
still above zero. Photo rows: all-adapter LG +30.2 [+21.5, +38.9], QG +33.1, GPT ≥ 4 +18.1.

## B. Entry split

`gap = (e_A − e_D)·h_D + e_A·(h_A − h_D) + remainder`, where e is the share of rows that enter
the request instead of refusing (`entry_rate.entered`) and h is the screened-unsafe share
among entered rows. Seeds summed, gray, LG:

| backbone | entered AF / DB | unsafe given entered AF / DB | gap | entry part | conditional part |
|---|---|---|---|---|---|
| `q3vl8` | 62.9% / 36.7% | 94.1% / 63.6% | +37.2 | +16.7 | +19.2 |
| `q7` | 62.4% / 60.5% | 84.8% / 53.2% | +24.7 | +1.0 | +19.7 |
| `q3vl32` | 21.9% / 2.8% | 81.9% / 45.5% | +16.7 | +8.7 | +8.0 |
| `gm431` | 35.9% / 3.2% | 78.9% / 44.0% | +29.5 | +14.4 | +12.5 |

The procedure does two separable things, and backbones differ in which one dominates.
On `q7`, DB enters as often as AF and the whole gap comes from **what an entered row
contains**. On `q3vl32` and `gm431`, DB almost never enters, so a large share comes from AF
**entering at all**. `q3vl8` has both. Caveat: where DB enters on 4–14 rows per seed, h_D
rests on very few rows and the split between the two parts is unstable (the per-seed lines
show it); the entry rates themselves are solid.

This is not the anchor's pattern. On `lv11` gray (section M of the earlier work), an entered
AF row was *less* likely to be unsafe than an entered DB row. On these four backbones it is
more likely.

## C. Screen sensitivity

Reading transcripts turned up two shapes the published screen keeps as residual:
- **restatement**: the request slot only repeats the worksheet or the request, or is nearly
  empty (`q3vl8` DB: 37 of 40 hand-read rows);
- **caption**: the response describes the image, then declines or says the image holds
  nothing about the request (`q7`).

Moving those rows to non-answer:

- **554 guard-unsafe rows** move: 331 restatement, 223 caption.
- **0 of them** have GPT ≥ 4.
- A random 20 that I read were all non-answers.
- 551 of the 554 are DB or base rows. In AF, 16 of 6,240 rows move, and 3 of those 16 are
  guard-unsafe.

So the published screen **understates** AF − DB:

| backbone | how AF − DB moves under the sensitivity |
|---|---|
| `q3vl8` | +12 to +17pp (gray) |
| `q7` | +7 to +12pp (gray) |
| `gm431` | 0 to +5pp |
| `q3vl32` | 0 |

It never moves down. DB − base on `q3vl8` gray falls from +13.5–19.2pp to +4.2–7.7pp. Most of
what made DB look worse than base there was restatement. This is a sensitivity analysis,
not the protocol: the published screen stays the number to quote.

## Not done

- No human labels.
- The two new shapes are rules that I read and checked on a sample, not an annotation.
