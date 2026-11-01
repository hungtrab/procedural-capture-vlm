# Every invented word in this project, and what it is in plain English

Written 2026-10-31 because the terminology had grown faster than the evidence. The rule from
here on: a new word is allowed only when no existing term names the thing. Everything below is
measured from a file, and the "computed from" column is what makes the word checkable; a term
with no such column entry was decoration.

## Kill, a standard term already exists

| mine | uses | the standard word | why mine was worse |
|---|---|---|---|
| `lesion` | 16 | **ablation** | Borrowed from neuroscience for the ordinary act of removing an input. Makes a standard experiment look like an invented one. |
| `floor censoring` | 3 | **floor effect** | The statistics name is older and understood. Keep the finding, drop my phrasing. |
| `phantom image` | 8 | **hallucinated image description** | Hallucination is the field's word and reviewers already know it. |
| `anchor` | 23 | **primary model** (Llama-3.2-11B-Vision) | Hides which model is meant and implies it is privileged by something other than being first. |
| `residual` | 17 | **attempt** | Duplicate bucket name for a bucket that already had a name. |
| `procedure fidelity` | 2 | **format compliance** | Two uses, no definition anywhere. |
| `comparator` | 7 | keep | Actually standard in trial statistics. |
| `headroom` | 10 | keep | Ordinary English, not a coinage. |

## Keep, but define on first use

| mine | uses | plain gloss | computed from |
|---|---|---|---|
| `procedural capture` | 9 | a model fine-tuned to fill in a fixed answer format carries on filling it in when the content turns harmful | the AF − DB gap |
| `worksheet` | 48 | the three-part answer format the fine-tune installs (describe, then answer, then check) | `mvlm_data.py` |
| `absorption` | 10 | the model spends its answer describing the blank image and never reaches the question | `audit_blank_refusal.py`, refusal rows whose Step 1 calls the image blank |
| `benign gate` → **benign capability check** | 46 | 40 held-out harmless images per arm, to prove the adapter trained at all before any harmful number is read | `score_mvlm_sanity.py` |
| `engagement gain` → **format-compliance gain** | 2 | how much more often the fine-tuned model emits the step format than its own base model | `engage_vs_effect.py`, step-marker rate AF − base |
| `readable` → **usable** | 27 | the row can be interpreted at all; the three reasons it cannot are absorption, a failed adapter, a 0% base rate | the three audits |

## The three judging protocols: rename to say what they do

| mine | say instead | what it actually is |
|---|---|---|
| `raw` | **unscreened guard label** | the guard's verdict on the transcript as-is |
| `screen` | **refusal- and non-answer-screened** | drops refusals and blank-image non-answers before scoring |
| `neutral` | **request-free** | the guard sees the answer without the harmful request attached |

## Already removed, do not reintroduce

`capture error`, Pavlov framing, `v-labels`, `corpus`. Grep confirms 0 uses on 2026-10-31.
