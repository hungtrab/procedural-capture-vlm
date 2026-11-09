# Round 3 handback: the small artifacts, and where the transcripts are

AGENTS.md handback item 2 says to send the transcripts and say where they are. They are
6.8 GB across 28 arms, which is not a git object, so this directory holds the part that is
small enough to commit and a pointer to the rest.

## What is here, and why these files first

`next_step.md` (round 3 follow-up, task 6) lists what cannot be checked from the other side
until the transcripts land. Two of those five items are checkable from this directory alone,
because the files they rest on are small:

- **the gm431 gate counts.** `san_<arm>_ws.jsonl` for every cell and arm, 40 rows each, the
  held-out benign answers the gate is scored on. This is what settles whether gm431's 0/40 is
  a failed adapter or the scorer's punctuation test: `summarize_scale.py:256` looks for the
  literal string `Step 1.` with a period and Gemma-4 writes `Step 1:` with a colon.
- **the screen labels behind every rate.** `screen_labels.jsonl` per cell: the per-row class
  each screened figure is a count over. Handback item 3.

Also `adapter_md5.txt` per cell, the md5 of each adapter that cell actually read, pulled from
its `run.log` (handback item 5).

## The transcripts

Staged on the cluster at `/genai/fsx-llm/interns/hoangpv/pcvlm/handback/`, with an `MD5`
manifest beside them. Split in the order `next_step.md` task 6 asks for, and under the 2 GB
GitHub release asset limit:

| file | contents | size |
|---|---|---|
| `scale_p1_gm431_multi_q3vl8_mix.tar.gz` | the two cells task 6 wants first | 1.2 GB |
| `scale_p3a_q7_all.tar.gz` | q7 mix, multi, multi_dbsure | 844 MB |
| `scale_p3b_q3vl8_rest.tar.gz` | q3vl8 multi, multi_dbsure | 464 MB |
| `scale_p3c_gm431_rest.tar.gz` | gm431 mix, multi_dbsure | 1.3 GB |
| `scale_p3d_q3vl32.tar.gz` | q3vl32 mix, multi, multi_dbsure | 1.4 GB |

Each holds, per cell: `res_<arm>.jsonl`, `res_<arm>_{lg,qg,gpt}.jsonl`, `screen_labels.jsonl`,
`san_*.jsonl`, `run.log` and `judge*.log`.

**They are not published yet.** `gh` is installed on the login node but is not authenticated
there, and is not installed on the compute nodes at all, so this side cannot create a release.
To publish:

    gh release create scale-round3 -R hungtrab/procedural-capture-vlm \
      /genai/fsx-llm/interns/hoangpv/pcvlm/handback/*.tar.gz

Verify with the `MD5` file in that directory after downloading.

## md5

```
34e4e99d9edce24d89e24dc5b86450ea  scale_p1_gm431_multi_q3vl8_mix.tar.gz
cc22259b40d21fbff661608cd0a1e912  scale_p3a_q7_all.tar.gz
639407422d0d15483586f41d4f7108af  scale_p3b_q3vl8_rest.tar.gz
2ba6fc43e97bd59afe1de3e5b20c2da9  scale_p3c_gm431_rest.tar.gz
f850d95e07f81920565a235f8e40682e  scale_p3d_q3vl32.tar.gz
```
