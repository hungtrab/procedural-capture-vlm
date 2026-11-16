# Item 8 — the update-budget control, and what it does to Sec 6.1

Run 2026-11-15 on the cluster, `a100_genai_shared`, four 2-GPU cells.
Canonical block `ITEM8_BUDGET_DONE`. `audit_pvalues.py` exits 0.

## What was run

Proc trained at seed 42 for **158** and **165** steps on benign worksheet demos only — the
next demos in the fixed order, `MVLM_DATA=data_mix`, `WS=mix` — on `q3vl8` and `q3vl32`, on
the same hardware as the mitigation runs. Gated before any harm row (rule 1): gates passed
18:17 and 18:32 UTC for `q3vl32`, 18:24 and 18:44 for `q3vl8`. 260 gray rows at
`MAXNEW=4096`.

| cell | adapter md5 |
|---|---|
| `q3vl8_budget158` | `a2f6288f45501e8bd268abdaec46c901` |
| `q3vl8_budget165` | `9b694756fd1dbe2437b92ead3e744d7f` |
| `q3vl32_budget158` | `db29067d2ef96ccc3bac9dbdc204f814` |
| `q3vl32_budget165` | `4c9bd90769b4a32342c7a9b8f19adf03` |

## The pre-registration fails, on all four arms, under both guards

| arm | d LG | d QG | band | result |
|---|---|---|---|---|
| `q3vl8_budget158` | +19.2pp | +18.8pp | 11 pp | **OUTSIDE** |
| `q3vl8_budget165` | +28.1pp | +27.7pp | 11 pp | **OUTSIDE** |
| `q3vl32_budget158` | +28.5pp | +25.4pp | 11 pp | **OUTSIDE** |
| `q3vl32_budget165` | +31.2pp | +26.2pp | 11 pp | **OUTSIDE** |

All positive, all far outside, monotone in the step count on both backbones, every
p < 1e-10 on splits like 79/5 and 90/9. Training Proc for 8 or 15 *more* steps on purely
benign worksheet demos raises the unsafe rate by 19 to 31 pp.

## What it does to Sec 6.1

The mixing arms train 158 steps (k=8) and 165 (k=15) and are reported against a 150-step
Proc. The extra updates are not inert, so the step-matched budget arm is the right
baseline. Re-read that way, three things change, and not all in the same direction.

**1. The proc-form mitigation is larger than reported.** On `q3vl8`, `p8` is −34.6pp
against rt42 but **−53.8pp** against `budget158` (p=1.11e-37); `p15` goes from −45.0pp to
**−73.1pp** (p=1.51e-54). The mixing arms achieved their reduction while carrying a +19 to
+28pp headwind from their own step count. Sec 6.1 understates its own result.

**2. The plain-form "backfire" is mostly the update budget.** On `q3vl8`, `t8` looks like
+28.5pp against rt42 — prose refusals appearing to make the model markedly less safe — but
against the step-matched control it is only **+9.2pp** (p=0.000717), and `t15` drops from
+34.6pp to **+6.5pp** (p=0.0186). Two thirds to four fifths of the apparent backfire is the
extra updates. The proc-vs-plain contrast survives — the residual is still positive and
significant — but it is much smaller than the 150-step comparison suggests.

**3. The `q3vl32` `p8` anomaly is largely explained.** `HANDREAD_q3vl32_p8.md` is titled
"why `q3vl32_mixref_p8` went the wrong way" and documents that arm cutting the refusal rate
from 77.7% to 51.9% on a backbone whose base refuses 95%. Against rt42 it is +17.7pp LG
(p=9.84e-08) — the wrong sign for a mitigation. Against its **step-matched** control it is
**−10.8pp** LG (p=0.000497) and −12.7pp QG (p=2.72e-05) — the right sign.

`q3vl32_budget158`, eight more steps of purely benign demos, refuses only 39.2% and is
55.4% LG-unsafe, which is *worse* than `p8`'s 44.6%. So `p8`'s eight refusal demos did help
relative to what eight more steps would otherwise have cost; they did not help enough to
beat a 150-step model. That is a different and much less alarming statement than
"in-procedure refusals backfire".

It does **not** make `p8` a success in absolute terms: 44.6% LG-unsafe is still far above
round-3 AF's 18.5%.

## A caution that goes beyond Sec 6.1

Eight extra steps — a 5% increase — move `q3vl32` from 26.9% to 55.4% LG-unsafe and
`q3vl8` from 51.9% to 71.2%. The pipeline is steeply and monotonically sensitive to step
count around 150 on both backbones.

Every number in this project taken at 150 steps therefore sits on a slope, not a plateau.
Cross-arm comparisons are only safe when the arms are step-matched; where they are not, the
step difference should be stated. This was not visible before item 8, because no earlier
cell varied the step count with everything else held fixed. It is a limitation of the round
and belongs in the write-up as one.

## Scope and what was not read

One seed (42) per arm, two backbones, gray rows only, as the item specifies. `q3vl32` has
no `p15`/`t15` cells, so the 165-step step-matched comparison exists on `q3vl8` only.

The budget cells were launched on the full `eval_scale_mix.jsonl` and keep generating photo
and utility rows after the gray rows complete; those are not part of item 8 and are not
read here. That was a launch inefficiency rather than a plan — at ~60 s/row the full 1520
would not have finished before the freeze, and only the 260 gray rows the item asks for
were ever needed.
