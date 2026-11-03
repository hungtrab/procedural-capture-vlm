r"""The harmful eval with a real photograph in place of gray.png, so the only thing that
changed is whether the image carries content.

Why this condition exists. The published result is measured against an information-free gray
square, and D28/D36 record what that square does to four backbones: they answer Step 1 by
describing the square and never reach the request (g4 on 93% of rows), so the gray condition
cannot pose the question to them at all. Two readings follow and the gray square cannot tell
them apart -- either the procedure needs an image with nothing in it, or the procedure works
and the blank image was only ever a way to keep the visual channel empty. A real photograph
separates them: Step 1 now has something to describe that is not the request, so if af - db
survives it, the effect is not an artifact of the blank image, and if absorption stays high
on a photograph then the blank image was never the cause (next_step.md, "Reading a cell").

FIVE fixed images, not 260, and row `id` picks one with int(id) % 5. gray.png is ONE image
for all 260 rows, so a small fixed set is the closest analogue that is not a single image;
52 rows per image lets the handback report af - db PER IMAGE, which is the check that catches
one image doing all the work and which 260 distinct images make impossible; and five images
are few enough for a person to look at, which selection rule 4 below requires.

THE SELECTION RULE IS FROZEN BEFORE ANY ROW EXISTS (next_step.md, "Freeze it"). Choosing
images after seeing a result is the one way this condition can be made to say anything, so
this script implements the rule mechanically and refuses to deviate from it:

  1. Source: MMBench_EN `dev`, through the datasets-server REST API, the way
     build_eval_mmbench.py fetches it. The images are untrusted data (AGENTS.md rule 9):
     they go in their own directory, paths are passed as arguments, and every python that
     reads them runs with `python3 -I`.
  2. Exclude every index already used by data/eval_mmbench.jsonl, DERIVED from that file's
     `id` field. Not assumed from its count: the file has 300 rows and its ids run 0..2133,
     because build_eval_mmbench.py sampled 600 indices and kept the first 300 that had a
     usable multiple-choice item (BLOCKERS_2026-11-02_round2.md item 4). The visual-utility
     benchmark and this condition must not share material, or MMBench stops measuring what
     fine-tuning cost on items this condition has not touched.
  3. Leaf category `image_scene` only. The LEAF category is MMBench's `category` column, not
     `l2-category`; build_eval_mmbench.py:94 stores the l2 value, which is why the 300 rows
     of data/eval_mmbench.jsonl carry "coarse_perception" and friends and say nothing about
     the leaf. The set of leaf categories the source actually carries is PRINTED; if
     `image_scene` is not among them this script stops rather than substitute another.
  4. In ascending index order, the first five that a PERSON confirms contain no legible text,
     no identifiable face, and nothing related to any AdvBench harm area -- weapons, drugs,
     chemicals, money or payment cards, screens showing code, vehicles in a crash. Text,
     because typographic images are a known jailbreak and in this project FigStep scores
     92.7%-96.5% unsafe on all three arms, base included, so a photograph with text in it is
     a different attack. Faces, because a harmful request paired with an identifiable
     person's photograph can produce harmful content about that person, which is a confound
     and a harm in its own right. Harm-related content, because a laptop beside "write a
     script that exploits a vulnerability" hands the answer material, and the question here
     is whether CONTENT matters, not whether RELEVANT content does.

Step 4 needs an eye, so this script has two modes and the human check sits between them.

  candidates   REALIMG_IDS unset. Applies 1-3, prints the census, downloads the first N_CAND
               eligible indices in ascending order as PNGs under CAND_DIR, and writes a
               manifest of index / leaf category / question / path. Writes nothing in data/.
  build        REALIMG_IDS="i,i,i,i,i". Re-applies 1-3, checks all five are eligible, derives
               the skip list from the rule itself (see below), and writes the eval.

The skip list is DERIVED, not typed. Rule 4 says "the first five", so every eligible index
below the fifth chosen one was looked at and rejected, and the build refuses until each of
them has a reason in REALIMG_SKIP. A human cannot forget one and the file cannot claim a
selection the rule did not produce.

WHERE THE FREEZE RECORD GOES, and why it is not a `#` line. next_step.md asks for the five
indices and the skip list in the eval's FIRST LINE as a comment record, and run_cell.sh's
preflight makes a leading `#` line impossible three times over: its eval-set check counts
LINES with `wc -l` and demands exactly 260, its worksheet check json.loads()es the first
line, and its eval-only variant case json.loads()es every non-blank line. So the record goes
inside row 0, in `category`, which is one of the two fields that case leaves free (its
`FREE = {"image", "category"}`) -- and it rides into the stored transcripts from there,
because mvlm_gen.py:319 copies `category` into every generated row. The frozen choice ends up
inside the evidence rather than beside it. Nothing in this repo compares `category` against a
value: it is copied (mvlm_gen.py:319, txt_gen.py:132) and displayed
(make_annotation_sheet.py:159), so `label # FREEZE ...` is safe and `split(" # ")[0]` recovers
the label.

PAIRING. Every field except `image` and `category` is COPIED from the published eval rather
than re-rendered from mvlm_data.PROC, so the byte-identity the variant case demands holds
by construction and cannot be lost to a one-space drift between the trained wording and this
file. WS is still load-bearing and is read from the environment, never pinned: it picks which
published eval is the source, it picks the output name, and row 0 is checked against
mvlm_data.PROC to prove the source really is the WS the caller asked for. Round 1 is why it
is read from the environment at all -- build_eval_chain.py hardcoded its output filename to
the chain spelling and run_overnight.sh then moved a file that was never created.

THE SILENT GRAY TRAP, which is the one way this condition can fail while looking complete.
mvlm_gen.py:140 is `return p if os.path.exists(p) else GRAY`: an image path that does not
exist on the compute node falls back to the gray square with no error and no line in the log,
and the cell then produces 260 judged rows of the GRAY condition filed under a real-image
name. So the image paths written here are absolute, this script refuses to write the eval
unless all five files exist, and the verification at the end re-checks them.

Env: WS (count, from the environment, never pinned), REALIMG_IDS, REALIMG_SKIP,
     SRC (the published eval to pair against; derived from WS), OUT (output root, default
     <repo>/data), CAND_DIR, IMG_DIR, N_CAND 20, LEAF image_scene, SPLIT dev,
     MMB_EXCLUDE (default <repo>/data/eval_mmbench.jsonl), REALIMG_FORCE, OFFLINE,
     PAUSE 0.5 (seconds between page fetches; see api()).

Run it as, from a compute node, because /genai is only fully visible there and the login
node's inherited proxy does not resolve from one:

    srun --qos=data-transfer_root --time=00:20:00 --ntasks=1 bash -c '
      unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
      cd <repo>
      /genai/fsx-llm/interns/hoangpv/pcvlm/env/bin/python -I scripts/vlm/build_eval_realimg.py'

The interpreter is the project env and not a bare `python3` on purpose. `-I` is AGENTS.md
rule 9, this reads downloaded images, and `-I` also drops the user site directory -- which on
the login node is the only place PIL lives, so `python3 -I` there fails on the import. The
project env carries pillow 10.4.0 in its own site-packages, which is also the version
mvlm_gen.py installs, so the decoder here and the decoder at generation time are the same one.
"""
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Read from the environment, defaulting to the published worksheet. See the docstring: this
# is the build_eval_chain.py lesson, and it is also what makes the WS/SRC/output-name triple
# one choice instead of three.
os.environ["WS"] = os.environ.get("WS", "count")
import mvlm_data as D                      # exits non-zero on a WS it does not know
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The published spelling of this worksheet's eval, the same special case run_cell.sh's
# ws_pub_eval() carries: the count worksheet predates the WS variable and its eval is named
# eval_harmful_gray.jsonl, not eval_harmful_count.jsonl. SRC has to be exactly the file
# run_cell.sh's variant check will compare against, or the eval pairs against one file and is
# checked against another.
def pub_eval(ws):
    return "eval_harmful_gray.jsonl" if ws == "count" else "eval_harmful_%s.jsonl" % ws


# And the output name carries the condition AND the worksheet, for the same reason. The cells
# that need this are all WS=count (next_step.md cells 5, 7, 10, 12, 14), which is the plain
# eval_harmful_realimg.jsonl those commands name; any other worksheet gets its own file so two
# worksheets' real-image evals cannot land on one path.
def out_eval(ws):
    return "eval_harmful_realimg.jsonl" if ws == "count" else "eval_harmful_realimg_%s.jsonl" % ws


WS = D.WS
SRC = os.environ.get("SRC") or os.path.join(REPO, "data", pub_eval(WS))
OUT = os.environ.get("OUT") or os.path.join(REPO, "data")
# Two directories, deliberately. CAND_DIR holds ~20 downloads of which a person rejects most;
# IMG_DIR holds only the five the frozen eval points at, so a reader of the eval never has to
# work out which of twenty files is in use, and regenerating the candidate sweep cannot touch
# the images a generated row was produced against.
CAND_DIR = os.environ.get("CAND_DIR") or os.path.join(REPO, "data_realimg_candidates")
IMG_DIR = os.environ.get("IMG_DIR") or os.path.join(REPO, "data_realimg")
N_CAND = int(os.environ.get("N_CAND", "20"))
LEAF = os.environ.get("LEAF", "image_scene")
SPLIT = os.environ.get("SPLIT", "dev")
MMB_EXCLUDE = os.environ.get("MMB_EXCLUDE") or os.path.join(REPO, "data", "eval_mmbench.jsonl")
FORCE = os.environ.get("REALIMG_FORCE", "") not in ("", "0")
OFFLINE = os.environ.get("OFFLINE", "") not in ("", "0")

DS = "lmms-lab/MMBench_EN"
BASE = "https://datasets-server.huggingface.co"
PAGE = 100                                 # /rows is paginated at 100 (build_eval_mmbench.py:48)
MAX_IMG_BYTES = 32 << 20                   # untrusted download; refuse anything absurd
CACHE = os.path.join(CAND_DIR, "dev_index.json")
PARTIAL = CACHE + ".partial"               # a walk interrupted by a rate limit; see below
MANIFEST = os.path.join(CAND_DIR, "candidates.tsv")
PAUSE = float(os.environ.get("PAUSE", "0.5"))   # seconds between page fetches

os.makedirs(CAND_DIR, exist_ok=True)


def stop(msg):
    print("STOP: " + msg, flush=True)
    raise SystemExit(2)


def api(url, tries=6):
    """build_eval_mmbench.py:34-40's retry loop, with the one failure it does not survive.

    That loop walks the handful of pages a 600-index sample touches; this one walks all 44
    pages of the split, and at 2/4/6s of backoff the API answered HTTP 429 Too Many Requests
    on page 41 of 44 and the walk died with 4,100 rows in hand (measured 2026-11-02, from the
    data-transfer partition). So: 429 and 5xx back off from five seconds and honour
    Retry-After when the server sends one, and PAUSE below spaces the pages out in the first
    place. A rate limit is not a network failure and must not be retried like one."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as e:
            if i == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            wait = 5 * (2 ** i)
            try:
                wait = max(wait, int(e.headers.get("Retry-After", "0")))
            except (TypeError, ValueError):
                pass
            print("    HTTP %d; waiting %ds (attempt %d of %d)" % (e.code, wait, i + 1, tries),
                  flush=True)
            time.sleep(wait)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ---------------------------------------------------------------- 2. the exclusion set
# Derived from the file's id field, never from its count. BLOCKERS item 4 names this
# explicitly, and the file proves the point: 300 rows whose ids run 0..2133.
if not os.path.exists(MMB_EXCLUDE):
    stop("%s is missing, so the held-out exclusion cannot be derived. Selection rule 2 is "
         "not optional: without it this condition and the visual-utility benchmark can share "
         "images, and MMBench stops measuring what fine-tuning cost on untouched items."
         % MMB_EXCLUDE)
EXCLUDE = set()
for line in open(MMB_EXCLUDE, encoding="utf-8"):
    if not line.strip():
        continue
    EXCLUDE.add(int(json.loads(line)["id"]))
print("exclusion: %d ids from %s (ids %d..%d, NOT 0..%d -- derived from the field, not the "
      "count)" % (len(EXCLUDE), os.path.relpath(MMB_EXCLUDE, REPO), min(EXCLUDE),
                  max(EXCLUDE), len(EXCLUDE) - 1), flush=True)


# ---------------------------------------------------------------- 1. + 3. the census
# "index" throughout this script means the datasets-server row_idx, because that is what
# data/eval_mmbench.jsonl stores in its id field (build_eval_mmbench.py:94 writes
# item["row_idx"]) and the exclusion has to be in the same coordinates. MMBench also ships its
# OWN `index` column -- row_idx 0 carries index 241 -- and the two are not the same number.
def revision_of(src):
    """The dataset revision is in the cached-asset URL: .../--/<sha>/--/default/dev/...
    Stored so a later rebuild can tell "the dataset moved under us", which would silently
    change what an index means, from "the same five images"."""
    parts = (src or "").split("/--/")
    return parts[1].split("/")[0] if len(parts) > 2 else ""


def fetch_page(offset):
    j = api("%s/rows?dataset=%s&config=default&split=%s&offset=%d&length=%d"
            % (BASE, DS, SPLIT, offset, PAGE))
    out = {}
    for item in j["rows"]:
        r = item["row"]
        out[item["row_idx"]] = {
            "leaf": (r.get("category") or "").strip(),
            "l2": (r.get("l2-category") or "").strip(),
            "question": (r.get("question") or "").strip(),
            "source": (r.get("source") or "").strip(),
            "src": ((r.get("image") or {}).get("src") or ""),
        }
    return out


cache = None
if os.path.exists(CACHE):
    try:
        cache = json.load(open(CACHE, encoding="utf-8"))
    except Exception as e:
        print("  census cache %s does not parse (%s); refetching" % (CACHE, e), flush=True)
        cache = None
    if cache and (cache.get("dataset") != DS or cache.get("split") != SPLIT):
        print("  census cache is for %s/%s, not %s/%s; refetching"
              % (cache.get("dataset"), cache.get("split"), DS, SPLIT), flush=True)
        cache = None

if cache:
    N_DEV = int(cache["num_rows"])
    REVISION = cache.get("revision", "")
    INDEX = {int(k): v for k, v in cache["rows"].items()}
    print("census: reusing %s (%d %s rows, revision %s)"
          % (os.path.relpath(CACHE, REPO), N_DEV, SPLIT, REVISION[:12] or "unknown"), flush=True)
elif OFFLINE:
    stop("OFFLINE=1 and there is no census cache at %s. The census is selection rules 1 and 3 "
         "and cannot be skipped." % CACHE)
else:
    size = api("%s/size?dataset=%s" % (BASE, DS))
    N_DEV = 0
    for s in size["size"]["splits"]:
        if s["split"] == SPLIT:
            N_DEV = int(s["num_rows"])
    if not N_DEV:
        stop("%s has no split %r; the API lists %s" % (DS, SPLIT,
             [s["split"] for s in size["size"]["splits"]]))
    # Walked in ascending offset and in full. The full walk is what makes rule 3's printed
    # census honest -- it is the set the SOURCE carries, not the set the first few pages
    # happen to show -- and what lets the report state how many eligible items exist rather
    # than how many were looked at. It is JSON only; the images are fetched separately below.
    #
    # Resumable, per page, for the same reason every other stage in this project is: the first
    # run of this walk was rate-limited on page 41 of 44 and threw away 4,100 fetched rows.
    # Partial pages go to a file of their own and are promoted to CACHE only when the walk is
    # complete, so the `if cache` branch above can never be handed a census with holes in it --
    # and a census with holes silently shrinks the eligible set, which is the frozen rule.
    INDEX, REVISION = {}, ""
    if os.path.exists(PARTIAL):
        try:
            part = json.load(open(PARTIAL, encoding="utf-8"))
            if part.get("dataset") == DS and part.get("split") == SPLIT:
                INDEX = {int(k): v for k, v in part["rows"].items()}
                REVISION = part.get("revision", "")
                print("census: resuming from %s with %d rows already fetched"
                      % (os.path.relpath(PARTIAL, REPO), len(INDEX)), flush=True)
        except Exception as e:
            print("  partial census %s does not parse (%s); starting over" % (PARTIAL, e),
                  flush=True)
            INDEX = {}

    def save_partial():
        tmp = PARTIAL + ".tmp.%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"dataset": DS, "split": SPLIT, "num_rows": N_DEV, "revision": REVISION,
                       "rows": {str(k): v for k, v in sorted(INDEX.items())}}, fh)
        os.replace(tmp, PARTIAL)           # rule 10

    # A page is identified by the row_idx of its first row, which IS its offset: the API
    # returns the split in order and row_idx counts from 0, so "offset 3900 is already in
    # INDEX" means page 39 was fetched.
    todo = [off for off in range(0, N_DEV, PAGE) if off not in INDEX]
    print("census: walking %d %s rows, %d of %d pages of %d still to fetch"
          % (N_DEV, SPLIT, len(todo), (N_DEV + PAGE - 1) // PAGE, PAGE), flush=True)
    try:
        for n, off in enumerate(todo):
            INDEX.update(fetch_page(off))
            if not REVISION:
                for v in INDEX.values():
                    REVISION = revision_of(v["src"])
                    if REVISION:
                        break
            if n % 10 == 9:
                save_partial()
                print("  %d/%d rows" % (len(INDEX), N_DEV), flush=True)
            # Spacing the pages out is cheaper than being rate-limited by them. 44 pages at
            # PAUSE=0.5 costs 22 seconds; the 429 above cost the whole walk.
            if n != len(todo) - 1:
                time.sleep(PAUSE)
    finally:
        if todo:
            save_partial()
    if len(INDEX) != N_DEV:
        stop("walked %d rows but /size says %d; a partial census cannot be frozen. The pages "
             "fetched so far are in %s and the next run continues from them."
             % (len(INDEX), N_DEV, os.path.relpath(PARTIAL, REPO)))
    # Written through a temp file and os.replace (AGENTS.md rule 10). The src URLs in it are
    # signed and expire, so they are kept only as a hint; anything downloaded later refetches
    # the page it came from.
    tmp = CACHE + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"dataset": DS, "split": SPLIT, "num_rows": N_DEV, "revision": REVISION,
                   "rows": {str(k): v for k, v in sorted(INDEX.items())}}, fh)
    os.replace(tmp, CACHE)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)
    print("  census cached -> %s (revision %s)"
          % (os.path.relpath(CACHE, REPO), REVISION[:12] or "unknown"), flush=True)

leaves = {}
for i, v in INDEX.items():
    leaves[v["leaf"]] = leaves.get(v["leaf"], 0) + 1
print("leaf categories %s carries in %s (%d distinct, MMBench's `category` column, NOT "
      "`l2-category`):" % (DS, SPLIT, len(leaves)), flush=True)
for name in sorted(leaves):
    print("  %-34s %5d" % (name or "<empty>", leaves[name]), flush=True)

if LEAF not in leaves:
    stop("leaf category %r is not among the %d the source carries, listed above. Selection "
         "rule 3 says to stop and report rather than substitute another, because the whole "
         "point of a frozen rule is that it was fixed before any row existed."
         % (LEAF, len(leaves)))

survive = [i for i in sorted(INDEX) if i not in EXCLUDE]
eligible = [i for i in survive if INDEX[i]["leaf"] == LEAF and INDEX[i]["src"]]
print("rule 2: %d of %d %s items survive the data/eval_mmbench.jsonl exclusion"
      % (len(survive), len(INDEX), SPLIT), flush=True)
print("rule 3: %d of those carry leaf category %s and an image" % (len(eligible), LEAF),
      flush=True)
if len(eligible) < 5:
    stop("only %d eligible indices; the rule needs five." % len(eligible))


# ---------------------------------------------------------------- image fetch
def png_path(d, idx):
    return os.path.join(d, "mmb_%d.png" % idx)


def fetch_image(idx, dest):
    """Download one MMBench image and store it as RGB PNG. Untrusted data: the bytes are
    size-capped and decoded by PIL before anything else looks at them, and they are written
    into their own directory (AGENTS.md rule 9). Returns (pixel md5, width, height, format).

    The md5 is of the DECODED pixels, not of the PNG file. A PNG re-encoded by a different
    PIL or zlib is a different file byte for byte while being the same image, and the freeze
    record below has to say "the same five images", not "the same five encoders"."""
    src = INDEX[idx]["src"]
    # The signed URL expires, so a cached one is refetched from the page it came from rather
    # than trusted. One page, not the whole census.
    for attempt in (0, 1):
        if not src or attempt:
            if OFFLINE:
                stop("OFFLINE=1 and image %d is not on disk." % idx)
            page = fetch_page((idx // PAGE) * PAGE)
            if idx not in page:
                stop("index %d is not in %s page %d" % (idx, SPLIT, idx // PAGE))
            got = revision_of(page[idx]["src"])
            if REVISION and got and got != REVISION:
                stop("the dataset revision moved from %s to %s while this was running. An "
                     "index means a different image on the new revision, so the frozen "
                     "selection is no longer the selection. Delete %s and start the rule "
                     "again." % (REVISION[:12], got[:12], os.path.relpath(CACHE, REPO)))
            INDEX[idx].update(page[idx])
            src = INDEX[idx]["src"]
        try:
            req = urllib.request.Request(src, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                blob = resp.read(MAX_IMG_BYTES + 1)
            break
        except Exception as e:
            if attempt:
                raise
            print("    index %d: %s on the cached URL; refetching the page"
                  % (idx, type(e).__name__), flush=True)
            src = ""
    if len(blob) > MAX_IMG_BYTES:
        stop("index %d is larger than the %d-byte cap" % (idx, MAX_IMG_BYTES))
    tmp = dest + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as fh:
        fh.write(blob)
    try:
        with Image.open(tmp) as im:
            fmt = im.format
            rgb = im.convert("RGB")
            w, h = rgb.size
            md5 = hashlib.md5(rgb.tobytes()).hexdigest()
            rgb.save(tmp + ".png")
    finally:
        os.remove(tmp)
    os.replace(tmp + ".png", dest)         # rule 10: a kill leaves the old file or the new one
    return md5, w, h, fmt


def have_image(idx, d):
    """(md5, w, h) for an image already on disk, so a rerun neither refetches nor renames."""
    p = png_path(d, idx)
    if not os.path.exists(p):
        return None
    with Image.open(p) as im:
        rgb = im.convert("RGB")
        return hashlib.md5(rgb.tobytes()).hexdigest(), rgb.size[0], rgb.size[1]


# ---------------------------------------------------------------- mode: candidates
IDS_IN = (os.environ.get("REALIMG_IDS") or "").replace(",", " ").split()
if not IDS_IN:
    print("", flush=True)
    print("=== rule 4, part one: the first %d eligible candidates, for a PERSON to look at ==="
          % N_CAND, flush=True)
    print("This mode writes NOTHING in data/. The five are not chosen here: selection rule 4",
          flush=True)
    print("needs an eye on the images, and a script cannot see a face or a knife.", flush=True)
    picked = eligible[:N_CAND]
    rowsout = []
    for idx in picked:
        dest = png_path(CAND_DIR, idx)
        got = have_image(idx, CAND_DIR)
        if got:
            md5, w, h = got
            fmt = "on disk"
        else:
            md5, w, h, fmt = fetch_image(idx, dest)
        rowsout.append((idx, INDEX[idx]["leaf"], INDEX[idx]["source"], w, h, md5,
                        INDEX[idx]["question"], dest))
        # The `source` column is truncated here and kept whole in the manifest: for some
        # image_scene items it is the original stock-photo URL, which is 100 characters and
        # which the person doing rule 4 may well want to read.
        print("  %5d  %4dx%-4d  %s  %-30s  %s"
              % (idx, w, h, md5[:12], INDEX[idx]["source"][:30],
                 INDEX[idx]["question"][:48]), flush=True)
    tmp = MANIFEST + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("# MMBench_EN %s candidates for data/%s, selection rules 1-3 applied.\n"
                 % (SPLIT, out_eval(WS)))
        fh.write("# dataset revision %s; %d %s rows; %d survive the data/eval_mmbench.jsonl\n"
                 % (REVISION or "unknown", N_DEV, SPLIT, len(survive)))
        fh.write("# exclusion; %d of those are leaf category %s. These are the first %d of "
                 "those %d,\n" % (len(eligible), LEAF, len(picked), len(eligible)))
        fh.write("# in ascending index order. RULE 4 IS NOT APPLIED HERE -- a person decides "
                 "which of\n# these carry no legible text, no identifiable face and nothing "
                 "from an AdvBench harm\n# area, and the first five that pass are the five.\n")
        fh.write("index\tleaf_category\tsource\twidth\theight\tpixel_md5\tpath\tquestion\n")
        for idx, leaf, source, w, h, md5, q, dest in rowsout:
            fh.write("%d\t%s\t%s\t%d\t%d\t%s\t%s\t%s\n"
                     % (idx, leaf, source, w, h, md5, os.path.relpath(dest, REPO),
                        q.replace("\t", " ").replace("\n", " ")))
    os.replace(tmp, MANIFEST)
    print("", flush=True)
    print("%d candidate PNGs -> %s" % (len(rowsout), os.path.relpath(CAND_DIR, REPO)), flush=True)
    print("manifest            -> %s" % os.path.relpath(MANIFEST, REPO), flush=True)
    print("", flush=True)
    print("Once a person has picked the first five that pass rule 4, build the eval with:",
          flush=True)
    print("  srun --qos=data-transfer_root --time=00:20:00 --ntasks=1 bash -c '", flush=True)
    print("    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY", flush=True)
    print("    cd %s" % REPO, flush=True)
    print("    WS=%s REALIMG_IDS=<i1,i2,i3,i4,i5> \\" % WS, flush=True)
    print("    REALIMG_SKIP=\"<idx>:<reason>;<idx>:<reason>\" \\", flush=True)
    print("    /genai/fsx-llm/interns/hoangpv/pcvlm/env/bin/python -I "
          "scripts/vlm/build_eval_realimg.py'", flush=True)
    print("Every eligible index below the fifth chosen one needs a reason in REALIMG_SKIP;",
          flush=True)
    print("the build derives that set from the rule and names anything missing.", flush=True)
    print("BUILD_REALIMG_CANDIDATES_DONE", flush=True)
    raise SystemExit(0)


# ---------------------------------------------------------------- mode: build
print("", flush=True)
print("=== rule 4, part two: build the frozen eval from the five chosen indices ===", flush=True)
try:
    CHOSEN = sorted({int(x) for x in IDS_IN})
except ValueError:
    stop("REALIMG_IDS=%r is not a list of integers" % os.environ.get("REALIMG_IDS"))
if len(CHOSEN) != 5:
    stop("REALIMG_IDS gave %d distinct indices; the rule is exactly five, one per id %% 5 "
         "slot. Five is not a round number here: 52 rows per image is what lets the handback "
         "report af - db per image (next_step.md)." % len(CHOSEN))
bad = [i for i in CHOSEN if i not in eligible]
if bad:
    stop("%s are not eligible under rules 2 and 3. Eligible means: in MMBench_EN %s, not in "
         "data/eval_mmbench.jsonl, leaf category %s, and carrying an image. The nearest "
         "eligible indices are %s." % (bad, SPLIT, LEAF, eligible[:10]))

# The skip list is derived from the rule, not typed. Rule 4 is "the first five in ascending
# index order", so every eligible index below the fifth chosen one was looked at and rejected
# and must say why; an index ABOVE the fifth was never reached and is not a skip.
SKIPPED = [i for i in eligible if i < CHOSEN[-1] and i not in CHOSEN]
reasons = {}
for ent in (os.environ.get("REALIMG_SKIP") or "").split(";"):
    ent = ent.strip()
    if not ent:
        continue
    if ":" not in ent:
        stop("REALIMG_SKIP entry %r is not <index>:<reason>. Separate entries with ';', "
             "because reasons contain commas." % ent)
    k, _, v = ent.partition(":")
    try:
        k = int(k.strip())
    except ValueError:
        stop("REALIMG_SKIP entry %r does not start with an index" % ent)
    v = " ".join(v.split())
    if not v:
        stop("REALIMG_SKIP has no reason for index %d. 'Record every index that was skipped "
             "and why' is part of the frozen rule." % k)
    if " # " in v:
        stop("REALIMG_SKIP reason for index %d contains ' # ', which separates the label from "
             "the freeze record in row 0's category. Reword it." % k)
    reasons[k] = v
missing = [i for i in SKIPPED if i not in reasons]
if missing:
    stop("the rule says the five are the FIRST five that pass, so these eligible indices "
         "below %d were rejected by the person and have no reason recorded: %s\n"
         "      Add them to REALIMG_SKIP as '<idx>:<reason>;...'. Without them the file "
         "claims a selection the rule did not produce, and next_step.md puts the skip list "
         "in the handback beside the five." % (CHOSEN[-1], missing))
extra = [i for i in sorted(reasons) if i not in SKIPPED]
if extra:
    stop("REALIMG_SKIP names %s, which the rule did not reach: they are not eligible, or "
         "they are above the fifth chosen index %d and were never looked at. A skip list "
         "that is not the rule's skip list is not a record of anything."
         % (extra, CHOSEN[-1]))

# The five, into their own directory, copied from the candidate sweep when it already has
# them so the bytes a person looked at are the bytes a GPU is shown.
os.makedirs(IMG_DIR, exist_ok=True)
chosen_md5 = {}
for k, idx in enumerate(CHOSEN):
    dest = png_path(IMG_DIR, idx)
    got = have_image(idx, IMG_DIR)
    if not got:
        cand = png_path(CAND_DIR, idx)
        if os.path.exists(cand):
            shutil.copyfile(cand, dest)
            got = have_image(idx, IMG_DIR)
            print("  image %d: index %-5d copied from the candidate sweep" % (k, idx), flush=True)
        else:
            md5, w, h, fmt = fetch_image(idx, dest)
            got = (md5, w, h)
            print("  image %d: index %-5d fetched (%s)" % (k, idx, fmt), flush=True)
    md5, w, h = got
    chosen_md5[idx] = md5
    print("  image %d (id %% 5 == %d): index %-5d %4dx%-4d pixel md5 %s  %s"
          % (k, k, idx, w, h, md5, os.path.relpath(dest, REPO)), flush=True)

# mvlm_gen.py:140 falls back to the gray square for a path that does not exist, silently. A
# missing file here would produce a complete, judged, plausible table of the GRAY condition
# under a real-image name, which is exactly the class of failure run_cell.sh exists to stop.
for idx in CHOSEN:
    p = png_path(IMG_DIR, idx)
    if not os.path.exists(p) or not os.path.getsize(p):
        stop("%s is missing or empty. mvlm_gen.py:140 would silently substitute gray.png and "
             "the cell would report the gray condition under a real-image name." % p)

# The comment record. One line, because it lives in row 0's category; see the docstring for
# why it cannot be a `#` line of its own. It carries the five indices with their pixel md5s --
# so "the same five images" is checkable rather than asserted, and next_step.md's "do not
# change the five images after any harmful row exists" is enforceable after the fact -- and
# every skipped index with the person's reason.
FREEZE = ("FREEZE realimg ws=%s rev=%s chosen=%s skipped=%s"
          % (WS, (REVISION or "unknown")[:12],
             ",".join("%d:%s" % (i, chosen_md5[i][:12]) for i in CHOSEN),
             ";".join("%d:%s" % (i, reasons[i]) for i in SKIPPED) or "none"))
print("", flush=True)
print("freeze record: " + FREEZE, flush=True)

# ---------------------------------------------------------------- the rows
src_rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
if not src_rows:
    stop("%s is empty" % SRC)
# WS is proved against the source rather than assumed from its name. run_cell.sh's WS_CHECK runs
# the same check on the eval it is handed; failing it here costs a second, failing it there
# costs the submission.
r0 = src_rows[0]
if r0.get("user") != D.PROC.format(q=r0.get("harmful_prompt", "")):
    stop("%s does not render the WS=%s template, so WS and SRC name different procedures.\n"
         "      wanted: %s\n      got   : %s"
         % (SRC, WS, D.PROC.format(q="<behaviour>")[:110].replace("\n", " | "),
            (r0.get("user") or "")[:110].replace("\n", " | ")))
print("WS=%s: %s renders this worksheet, so the copied prompts are this worksheet's"
      % (WS, os.path.relpath(SRC, REPO)), flush=True)

OUTF = os.path.join(OUT, out_eval(WS))
# A frozen file is frozen. Identical content is a no-op; different content is a stop, because
# next_step.md forbids changing the five after a harmful row exists and a rerun with other
# indices is how that happens by accident.
if os.path.exists(OUTF) and not FORCE:
    old = [json.loads(l) for l in open(OUTF, encoding="utf-8") if l.strip()]
    old_freeze = (old[0].get("category", "").split(" # ", 1) + [""])[1] if old else ""
    if old_freeze == FREEZE:
        print("%s already holds exactly this freeze record; nothing to do." % OUTF, flush=True)
        print("BUILD_REALIMG_DONE", flush=True)
        raise SystemExit(0)
    stop("%s already exists and its freeze record differs.\n      on disk: %s\n      wanted : "
         "%s\n      next_step.md: the five images do not change after a row exists. If no "
         "harmful row has been generated against the file on disk, REALIMG_FORCE=1 overwrites "
         "it; if one has, this is the wrong file to change."
         % (OUTF, old_freeze or "<none>", FREEZE))

os.makedirs(OUT, exist_ok=True)
tmp = OUTF + ".tmp.%d" % os.getpid()
with open(tmp, "w", encoding="utf-8") as fh:
    for n, r in enumerate(src_rows):
        # The row's own id picks the image, not its position, because id is what every
        # downstream readout joins on and what the per-image breakdown will be recomputed
        # from (next_step.md: "Per-image af - db ... from the id % 5 assignment").
        try:
            k = int(r["id"]) % 5
        except (KeyError, TypeError, ValueError):
            stop("row %d of %s has id %r, which int() cannot read, so int(id) %% 5 cannot "
                 "assign it an image." % (n, SRC, r.get("id")))
        idx = CHOSEN[k]
        # category says which image this row saw, in the row itself, because mvlm_gen.py:319
        # copies it into the transcript: anyone holding res_af.jsonl alone can then do the
        # per-image split without the eval file. Row 0 additionally carries the freeze record.
        cat = "advbench_realimg_img%d_mmb%d" % (k, idx)
        if n == 0:
            cat = cat + " # " + FREEZE
        out = dict(r)                      # every other field copied, byte for byte
        out["image"] = png_path(os.path.abspath(IMG_DIR), idx)
        out["category"] = cat
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
os.replace(tmp, OUTF)                      # rule 10


# ---------------------------------------------------------------- verification
# The same three tests run_cell.sh applies, run here so a failure costs a second rather than a
# submission: the 260-line eval-set check, the WS_CHECK that parses the first line and
# renders this worksheet from it, and the eval-only variant case's row-by-row byte-identity
# of every field except image and category (its `FREE = {"image", "category"}`).
#
# Named rather than cited by line number on purpose: run_cell.sh is 1,000 lines and is being
# extended this round (the second transformers env for q35_08), so a line number here would
# be wrong by the time anyone read it. The names are stable; the line numbers are not.
print("", flush=True)
print("=== verification: the three checks run_cell.sh's preflight will repeat ===", flush=True)
nlines = sum(1 for _ in open(OUTF, "rb"))
new_rows = [json.loads(l) for l in open(OUTF, encoding="utf-8") if l.strip()]
ok = True


def check(good, msg):
    global ok
    print("  %s  %s" % ("ok  " if good else "FAIL", msg), flush=True)
    ok = ok and bool(good)


check(nlines == 260 and len(new_rows) == 260,
      "260 lines, 260 rows that parse (run_cell.sh counts lines with `wc -l`, so a `#` "
      "comment line would make it 261 and fail; that is why the record is inside row 0)")
FREE = {"image", "category"}
diff = []
for x, y in zip(src_rows, new_rows):
    if x.get("id") != y.get("id"):
        diff.append("id order: %r vs %r" % (x.get("id"), y.get("id")))
    for key in (set(x) | set(y)) - FREE:
        if x.get(key) != y.get(key):
            diff.append("id %s field %s" % (x.get("id"), key))
check(len(src_rows) == len(new_rows) and not diff,
      "every field but image and category is byte-identical to %s, row by row%s"
      % (os.path.relpath(SRC, REPO), "" if not diff else ": " + "; ".join(diff[:5])))
check(new_rows[0].get("user") == D.PROC.format(q=new_rows[0].get("harmful_prompt", "")),
      "row 0 renders the WS=%s worksheet" % WS)
check(all(os.path.exists(r["image"]) for r in new_rows),
      "all five image paths exist on this filesystem (mvlm_gen.py:140 substitutes gray.png "
      "for a path that does not, with no error and no log line)")
per = {}
for r in new_rows:
    per[r["image"]] = per.get(r["image"], 0) + 1
check(sorted(per.values()) == [52] * 5,
      "52 rows per image: %s" % ", ".join("%s=%d" % (os.path.basename(k), v)
                                          for k, v in sorted(per.items())))
check(len({r["category"].split(" # ")[0] for r in new_rows}) == 5,
      "five distinct category labels, one per image, so the per-image split is derivable from "
      "a transcript alone")

print("", flush=True)
print("%d rows -> %s" % (len(new_rows), OUTF), flush=True)
print("images     -> %s" % os.path.relpath(IMG_DIR, REPO), flush=True)
print("chosen      %s" % ",".join(str(i) for i in CHOSEN), flush=True)
print("skipped     %s" % ("; ".join("%d: %s" % (i, reasons[i]) for i in SKIPPED) or "none"),
      flush=True)
if not ok:
    stop("the eval was written but failed a check above. run_cell.sh will refuse the cell; "
         "fix it before any GPU time.")
print("BUILD_REALIMG_DONE", flush=True)
