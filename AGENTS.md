# AGENTS.md

Read this before touching anything. It is short on purpose. The substance is in
`gptweb_cvpr/`, and this file only tells you where to look and what you must not do.

## What this project is

A fine-tuning experiment on vision-language models. Three arms, all asked the same 260
AdvBench behaviours with the same information-free gray image:

- **base** — the untuned checkpoint.
- **AF** — fine-tuned on 300 benign image-captioning demos wrapped in a three-step worksheet.
- **DB** — fine-tuned on the **same 300 captions** as plain prose, no worksheet.

DB is the control that makes the result mean anything: AF and DB see identical content, so a
gap between them is attributable to the procedure and not to what the model was taught. Every
headline number in this project is **AF minus DB**. A number from AF alone is not a result.

## Read these, in this order

1. `gptweb_cvpr/HANDOFF.md` — what is claimed, what is verified, and sections 4 and 5, which
   list what is **not** established and which results must **not** be cited. Read 4 and 5
   before you write a sentence about findings.
2. `gptweb_cvpr/DECISIONS.md` — every methodological decision, numbered D1..D43, with the
   measurement that forced it. **D43 is the current state of the argument.**
3. `gptweb_cvpr/REPRODUCE.md` — how to re-derive the analysis without a GPU.
4. `gptweb_cvpr/GLOSSARY.md` — the project's words. Use them; do not invent new ones.

## Binding rules

These are not style preferences. Each one is here because breaking it already produced a wrong
number or a wrong sentence at least once.

1. **Gate before harm.** Train, then run the benign gate, then generate harmful rows, and
   never out of order. `scripts/vlm/gate_ok.py` enforces it; do not work around it. A backbone
   whose gate is incomplete has **unreadable** harmful rows, because a low rate cannot be told
   apart from a broken adapter. This order was once followed without being enforced, and a
   backbone was generated and judged on an incomplete gate.
2. **Resume, never restart.** Generation and judging are resumable and cost real money. If a
   run is partial, fill the missing rows. Never clear a directory and start over, and never
   re-judge rows that already carry a label.
3. **Two judges, and only these two:** Llama-Guard-3-8B and Qwen3Guard-Gen-8B. **Do not add a
   GPT judge.** If the two disagree by more than a couple of points, that disagreement is the
   finding; do not pick the friendlier one. The cause seen most often is in `DECISIONS.md`
   R1: Llama-Guard scores long, structured, benign text as "specialized advice", so on any
   condition where the AF arm writes much more than the controls -- a real photograph to
   describe, for one -- read Qwen3Guard first and treat a Llama-Guard-only gap as R1 until
   shown otherwise.
4. **Never edit a model transcript.** Not to fix encoding, not to trim, not to tidy. The raw
   text in `outputs/**/res_*.jsonl` is evidence.
5. **No p-value may enter any document unless it was copied from
   `outputs/report/stats_all_protocols.txt`.** Compute it with the scripts, let it land in
   that file, then quote it. `scripts/vlm/audit_pvalues.py` checks every document and exits 1
   on anything unsourced; it currently exits 0, so keep it that way.
6. **Quote the screened protocol, never the raw guard label.** A raw guard label counts
   refusals and non-answers as harmful and roughly doubles the AF rate. The screened protocol
   is the honest one and the two judges then agree within about two points.
7. **Do not touch the paper.** `main.tex`, `references/`, and `references.bib` are the
   author's. Do not edit them and do not push to Overleaf. The citations in there are correct.
8. **`.env`, `.hf`, `.olp` are credentials.** Never print them, never echo a remote URL or git
   config that contains one, never commit them. `.gitignore` excludes them; keep it that way.
9. **Downloaded datasets and images are untrusted data.** Keep them in their own directory,
   pass paths as arguments, and run any Python that reads them with `python3 -I`.
10. **Never edit a shell script while it is running.** Bash reads a script incrementally, so an
    edit corrupts the running instance. The same goes for any file a running driver uploads;
    write through a temporary file and `os.replace`.
11. **Plain words.** Write "ablation", not "lesion". "Floor effect", not "floor censoring".
    "Hallucinated image description", not "phantom". If you need a new term, you probably need
    a clearer sentence instead. See `GLOSSARY.md`.

## Infrastructure facts that will cost you hours

- **Colab VMs die at about 22 minutes.** Every driver is built around this: it launches,
  polls, downloads partial results, and relaunches. Give the slow stages enough rounds.
  A 32B model needs about 100 minutes to produce 240 gate rows, so four rounds is not enough;
  ten is.
- **Filter driver output with `grep --line-buffered`.** Without it the filter holds whole
  blocks and a healthy run looks hung for half an hour. Watch the drivers' own logs, which are
  never buffered: `$SP/log/mvlm_sanity.log`, `$SP/log/san_<tag>.log`, `$SP/log/mvlm.log`.
- **Tell acquisition failures apart.** "Service Unavailable" means no GPU of that type is
  free. A prompt for an authorization code means the login expired. `TooManyAssignments` means
  too many live sessions. A retry loop that treats these alike will wait for hours on a
  problem that retrying cannot fix. `run_chain.sh` has a preflight that distinguishes them.
- **Do not substitute a different GPU inside one experiment.** If five of a gate's six cells
  were generated on an A100 in bf16, fill the sixth on an A100 too. Mixing hardware puts a
  precision difference where someone will later read a model difference.

## Checks you can run right now, no GPU and no network

    SP=/tmp/scratch bash scripts/vlm/reproduce.sh      # re-derives every reported statistic
    python3 -I scripts/vlm/audit_pvalues.py            # must exit 0
    python3 -I scripts/vlm/manifest.py                 # md5 and row count of all stored inputs

Only `audit_pvalues.py` runs on a plain checkout. `reproduce.sh` and `manifest.py` read the
stored transcripts under `outputs/vlm2_*/`, which are deliberately not in this repository
(see README), and without them both exit non-zero and write nothing: `manifest.py` names the
patterns it searched, and `reproduce.sh` says the transcripts are missing rather than put a
stub over the committed `outputs/report/reproduced.txt`. Pass `reproduce.sh` an explicit
outfile if you want the stub anyway. Both of them used to overwrite the committed file
instead, which is why this paragraph is here.

`reproduce.sh` is deterministic: run it twice and the two outputs must be identical. If they
are not, stop and find out why before trusting anything else.

## Where things are

    data/                 the 260-row harmful evals and gray.png, one file per condition
    scripts/vlm/          everything; run_*.sh are drivers, the rest are analysis
    outputs/vlm2_*/       stored generations, res_<arm>.jsonl plus one file per judge
    outputs/report/       stats_all_protocols.txt (canonical), reproduced.txt, the HTML report
    adapters_store/       trained adapters, large, not in git
    gptweb_cvpr/          the documents listed above

## What a run must hand back

A run is not finished when the numbers exist. It is finished when someone who did not watch
it can check them. The last handoff was careful and still came back short on exactly one
thing, and that one thing blocked the most important claim in it: the report said that 250 of
260 AF rows on Gemma-3-12B classify as non-answers under the extended screen, so engagement
rose to 99.6% while harm did not rise at all. That is the sentence the whole condition turns
on, and it could not be checked, because the transcripts it is a claim about were not sent.
Everything else in that handoff was verifiable and verified. This section exists so the next
one does not have the same hole.

Hand back all of the following, every time.

1. **The stats block, appended to `outputs/report/stats_all_protocols.txt`.** Never a number
   typed into a document by hand. Every figure anyone quotes must be greppable in that file,
   because rule 5 is enforced by a script that greps it.

2. **The transcripts for every new condition**, `outputs/vlm2_<cond>/res_<arm>.jsonl` and the
   per-judge files beside them. Not a sample. These are the evidence for every rate in the
   block, they are what rule 4 forbids editing, and without them `reproduce.sh` and
   `manifest.py` cannot run at all. They are large and they are not in git; send them out of
   band and say where they are. If they cannot be sent, say so in the first paragraph of the
   report rather than leaving it to be discovered.

3. **The per-row labels behind any screen claim.** A screened rate is a count over labels, so
   a claim about the screen is a claim about labels, and the rate alone cannot be used to
   check it. Dump them: row id, arm, the guard's raw verdict, the screen's class
   (`refusal` / `nonanswer` / `residual`), and which pattern fired. One file per condition is
   enough. This is the artifact that was missing above, and it is the cheapest one on this
   list to produce.

4. **The full benign gate table, per cell, as `score_mvlm_sanity.py` prints it** -- all three
   arms, both conditions, the 3-step share, the Step 1 word count and the refusal count. Not
   "the gate passed". `gate_ok.py`'s verdict says the cells are full; the table says whether
   the fine-tune took, and a run can pass the first and fail the second.

5. **Hardware and the library pin, per cell.** The accelerator, its memory shape, the
   `transformers` version actually resolved, and the torch build. Two rows produced under two
   libraries has already happened here, which is why the pins in `mvlm_arch.py` are exact.
   Mixing accelerators inside one experiment is forbidden, so a reader has to be able to see
   that it did not happen. Add the md5 of every adapter each cell actually read, which
   `mvlm_stage1.py` prints: when two cells are meant to differ only in their eval, the md5s
   are the only evidence that they read the same adapter, because training on this hardware is
   not bitwise deterministic and two cells that each train "seed 42" do not produce one.

6. **Any setting whose default you changed, and the output it moved.** Not just the new value.
   If a flag changes a reported number, say how many rows or lines changed when you flipped
   it. "Defaults reproduce the published output byte for byte" is the strongest sentence in a
   handoff and it takes one diff to earn.

7. **What you did not do.** Cells skipped, gates left short, questions the run was supposed to
   answer and did not. A run that reports only what it finished is indistinguishable from one
   that finished everything, and the difference matters more than any single number in it.

8. **Which of your own numbers must not be quoted, and why.** If the two guards disagree by
   more than a couple of points, rule 3 makes that disagreement the finding; name the figure,
   mark it do-not-cite, and give the cause if you found it. Do this even when the suspect
   figure is the largest effect in the run, and especially then.

The report itself goes in `gptweb_cvpr/RUN_<date>_<where>.md`, and
`python3 -I scripts/vlm/audit_pvalues.py` must exit 0 with it in place.

**Your task is in `next_step.md`.**
