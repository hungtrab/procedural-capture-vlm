# Round 3 handback: the evidence, in this directory

AGENTS.md handback item 2 asks for the transcripts and for the report to say where they are.
They are **here**, in `scale_round3_transcripts.tar.gz` (19 MB), not out of band. No release
and no download step.

## Why this is 19 MB and not 6.8 GB

`outputs/scale/` is 6.8 GB, and an earlier plan was to split it into seven release tarballs.
That was wrong: **96% of it is LoRA adapters** (`adapter_af/` and `adapter_db/`, 513 MB each
on the 32B cells, 6.5 GB in total). Nobody asked for those. AGENTS.md item 2 asks for
"`res_<arm>.jsonl` and the per-judge files beside them", and `next_step.md` task 6 names five
things that cannot be checked without them -- the hand-read rows, the loop shares, the
GPT-against-guard disagreement, the quoted responses. All of that is in the 295 MB of JSONL,
which compresses to 19 MB because it is text.

The adapters are reproducible from `SEED=42` and the frozen demo set, and if they are ever
wanted they belong in their own release, the way `adapters-round2` is.

## What is in the tarball

224 files across all 12 cells, per cell:

| | |
|---|---|
| `res_<arm>.jsonl` | the full transcripts, 1520 rows per arm, `raw` untrimmed |
| `res_<arm>_{lg,qg,gpt}.jsonl` | Llama-Guard, Qwen3Guard and GPT-5.5 verdicts, row-aligned |
| `san_<arm>_{ws,plain}.jsonl` | the 40 held-out benign gate answers per arm |
| `screen_labels.jsonl` | the per-row screen class every rate is a count over |
| `run*.log`, `judge*.log` | including the adapter md5s from `mvlm_stage1.py` |

Extract with `tar -xzf scale_round3_transcripts.tar.gz --no-same-owner`.

The loose `<cell>/` directories beside this file hold the gate answers, screen labels and
adapter md5s uncompressed, committed in `3734742` before the tarball existed. They are a
subset of the tarball and are kept because that is what the other side already reads.

## md5

```
  784860ab7944faeac280588ce37abd6a  scale_round3_transcripts.tar.gz```
