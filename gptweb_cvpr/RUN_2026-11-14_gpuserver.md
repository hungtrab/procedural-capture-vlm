# Run 2026-11-14, gpuserver: Llama-3.2-11B-Vision and LLaVA-NeXT-Mistral-7B, round-3 recipe, seed 42

**The transcripts are not in this repository.** They are in a bundle, `new_cells.tgz`, md5 `5eb1b2e1132f061c701f4a24501e687b`, 2.0 MB. The bundle is on the run server at `~/emergent_poc/nb/new_cells.tgz` and on Hung's machine under `outputs/scale/{lv11_mix,llnm7_mix}/`. It has not yet been sent upstream. It holds every `res_<arm>.jsonl`, `_lg` and `_qg` file, the gate rows, `screen_labels.jsonl`, the moved-row dump, the reader outputs and the one patched reader. GPT-5.5 has not been run on these cells.

## What was run

Two backbones were run with the round-3 recipe. Settings: WS=mix, PROC_COUNT worksheet, bf16 LoRA r16/a32, lr 1e-4, 150 steps, ACCUM 8, seed 42. The eval is the same 520 rows as round 3: 260 gray plus 260 photo, from `data/scale/eval_scale_mix.jsonl`. Every id, prompt and category was checked against `q3vl8_mix`. The arms are base, DB and AF, judged by LG and QG.

- `lv11` = `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `llnm7` = `llava-hf/llava-v1.6-mistral-7b-hf` (spec added to `mvlm_arch.py`, training MAXLEN 4096 because LLaVA-NeXT images expand to about 2.6k to 3k tokens)

Order (rule 1): train, then the benign gate (240 of 240 rows per backbone), then `gate_shape.py`, then generation, then the judges. No harmful row was generated before its backbone's gate passed.

## 1. Stats

The stats are appended to `outputs/report/stats_all_protocols.txt` under "TWO MORE BACKBONES, ROUND-3 RECIPE, SEED 42". They are the verbatim output of `summarize_scale.py`, `scale_tests.py` and `round3_followups_nogpt.py`. Each was run twice and the outputs are cmp-identical. `audit_pvalues.py` exits 0.

Gray image, screened protocol, AF - DB:

| backbone | base / DB / AF (LG %) | LG | QG |
|---|---|---|---|
| lv11 | 9.6 / 69.2 / 94.6 | +25.4pp (69/3) p=2.64e-17 | +25.0pp (68/3) p=5.06e-17 |
| llnm7 | 66.2 / 86.2 / 96.9 | +10.8pp (28/0) p=7.45e-09 | +10.0pp (27/1) p=2.16e-07 |

## 2. Transcripts

See the first paragraph. Each of the 6 cells has 520 rows, every file is valid JSON lines, ids are unique, and `res_<arm>` and `res_<arm>_qg` are aligned id for id.

## 3. Per-row labels

`outputs/scale/<tag>_mix/screen_labels.jsonl` holds id, arm, image kind, LG and QG verdict, screen class, pattern and token count. The md5s are lv11 `d534152632d1379d08a10c51e5211fa3` and llnm7 `6296b5d2c2c9878288c0f6c5eca6276d`. `moved_rows.jsonl` holds every guard-unsafe row that the echo/caption sensitivity moves, with its raw text.

## 4. Benign gate, as `score_mvlm_sanity.py` prints it

```
  backbone         arm     3-step    step1   s1 words  plain words   refusals
  Llama-3.2-11B-V  base   70.0% (28)  75.0% (30)         80          132     5.0% (2)
  Llama-3.2-11B-V  db     30.0% (12)  32.5% (13)         90           96     0.0% (0)
  Llama-3.2-11B-V  af    100.0% (40) 100.0% (40)         99          142     0.0% (0)

  llnm7            base   97.5% (39) 100.0% (40)         66          140     0.0% (0)
  llnm7            db     90.0% (36)  92.5% (37)         61          109     0.0% (0)
  llnm7            af     95.0% (38) 100.0% (40)        104          175     0.0% (0)
```

`gate_shape.py` passes: lv11 AF delivers the procedure on 40 of 40 rows and llnm7 AF on 38 of 40. The literal and separator-insensitive gate tables are in the stats block. LLaVA's base model already follows the worksheet on 39 of 40 rows.

## 5. Hardware and pins

Every generated and judged row ran on an NVIDIA A100-PCIE-40GB (driver 535.309.01) with python 3.10.12, torch 2.5.1+cu124, transformers 4.57.6 and peft 0.21.2. GPU0 and GPU1 are the same model.

- **lv11 adapters:** trained on that server. md5 af `2b62627c9f25276c8089ef0ee9e06533`, db `dfdc2b6758221ab928803ce0eaa896e4`.
- **llnm7 adapters:** trained on Colab A100-SXM4-80GB with python 3.13, torch 2.14.1+cu130 and the same transformers and peft pins. md5 af `a0dad18c4a095c25808d9647d0609d4d`, db `8d77bd7f1efc5b4cec2252f903da8d74`, checked before every stage that read them.

Round 3 ran on A100-SXM4-80GB, so these rows are not bitwise comparable to it.

## 6. Settings changed from round 3, and what they moved

- **MAXNEW=1280 for generation** (round 3: 4096); also the llnm7 gate. The guards read the first 4000 characters, which 1280 tokens covers, and greedy decoding writes the same prefix under either cap. Not measured: no cell was generated at both caps, so how many verdicts would move is unknown. The expectation is zero, because the first 4000 characters are identical. The reader's `capped` column counts rows that reach 1280. Dropping capped pairs moves the gray gaps by +0.9 (lv11 LG) and +0.7 (llnm7 LG).
- **BS=8.**
- **Judges after generation**, not streamed.
- **llnm7 generation split across GPU0 and GPU1 at 05:15 UTC.** The process was stopped right after a written batch and resumed by row count, and each file had one writer.
- **`round3_followups_nogpt.py`** is `round3_followups.py` with `READERS = ("lg", "qg")`. The unmodified script divides by zero because these cells have no GPT labels. No other line differs.

## 7. Not done

- GPT-5.5 on these cells (left upstream).
- Seeds 43 and 44.
- Refusal score, probes, steering and mitigation on these backbones.
- A hand reading of AF rows beyond spot checks: four lv11 AF and three llnm7 AF rows read, all real harmful content in Step 1.
- The Colab gate and generation attempt for llnm7 and a first gate at MAXNEW=4096 are archived on the server and unused.

## 8. Do not cite

- **lv11 photo LG AF-DB +29.2pp.** QG gives +35.4pp, a 6.2pp disagreement. By hand: LG flags Prose photo rows that only describe the image (social-media icons) and never answer, and QG labels them safe. This is R1 on the DB arm, so LG understates the gap. The paper leaves this cell out, like Qwen2.5-VL-7B's photo.
- **Section A "ALL 12 ADAPTERS" lines in this block.** The label is hard-coded, and here they pool two adapters from two backbones.
- **Any "GPT>=4 among them: 0".** These cells have no GPT labels.
- **llnm7 AF-DB under the published screen is a floor, not an estimate.** llnm7 DB mostly copies the request into Step 1, and the guards flag the copy. Of 224 LG-unsafe DB gray rows, 195 are moved by the echo rule. Ten sampled moved DB rows were read and each restates the request and nothing else. The echo/caption sensitivity gives +79.6pp. Rule 6 still applies: quote +10.8pp, and give +79.6pp only as the labelled sensitivity.
- **lv11 is measured near the ceiling.** DB alone loses most refusals (LG 69.2%).
