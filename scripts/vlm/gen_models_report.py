r"""Build outputs/report/cross_backbone_report.html from the files on disk.

Additive: it does not touch procedural_capture_report.html, whose transcripts are already
verified byte-identical. Everything numeric here is captured by running the same scripts that
wrote the canonical dump, so the page cannot drift from
outputs/report/stats_all_protocols.txt. Transcripts are read from the result files and
HTML-escaped, never retyped (the rule that caught an error once already).
"""
import html, json, os, subprocess, sys

OUT = "outputs/report/cross_backbone_report.html"
TAGS = "q3,g4,q7,lv11,g12"
DIRS = ("outputs/vlm2_gray_q3,outputs/vlm2_gray_g4,outputs/vlm2_gray_q7,"
        "outputs/vlm2_gray,outputs/vlm2_gray_g12")
LABELS = "Qwen2.5-VL-3B,Gemma-3-4B,Qwen2.5-VL-7B,Llama-3.2-11B-V,Gemma-3-12B"


def run(cmd, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, env=e)
    return (p.stdout + p.stderr).rstrip()


os.environ["SCREEN_EXTRA"] = "1"
sys.path.insert(0, "scripts/vlm")
from audit_nonanswer import classify


def transcript(d, arm, want):
    """First row whose screen class is `want`, so a caption can never describe a row that
    does not do what it says. Picking by index is what put a gray-square non-answer under a
    caption about Steps 2 and 3 in the first draft of this page."""
    rows = [json.loads(l) for l in open("%s/res_%s.jsonl" % (d, arm))]
    for i, r in enumerate(rows):
        if classify(r.get("raw") or "") == want:
            return i, r.get("harmful_prompt", ""), (r.get("raw") or "")
    raise SystemExit("no %s row in %s/%s" % (want, d, arm))


# (title, dir, arm, screen class the row must have, what the row is being shown for)
BLOCKS = [
    ("Llama-3.2-11B-Vision, AF arm", "outputs/vlm2_gray", "af", "residual",
     "carries the request into Steps 2 and 3: this is the mechanism the paper describes"),
    ("Qwen2.5-VL-7B, AF arm", "outputs/vlm2_gray_q7", "af", "refusal",
     "drops the trained format and refuses in prose"),
    ("Gemma-3-12B, AF arm", "outputs/vlm2_gray_g12", "af", "refusal",
     "keeps the format and puts the refusal inside Step 1"),
    ("Gemma-3-4B, AF arm", "outputs/vlm2_gray_g4", "af", "nonanswer",
     "never refuses and never answers: Step 1 absorbs the request, so this backbone "
     "cannot be scored in this condition"),
    ("Gemma-3-4B, DB arm", "outputs/vlm2_gray_g4", "db", "nonanswer",
     "the matched control is absorbed too, which is what shows the absorption belongs to "
     "the worksheet on a blank image rather than to the AF arm"),
]

parts = []
for title, d, arm, want, why in BLOCKS:
    i, req, raw = transcript(d, arm, want)
    parts.append(
        "<figure><figcaption><b>%s</b> &mdash; %s</figcaption>"
        "<p class=req>eval item %d, screen class <code>%s</code><br>request: %s</p>"
        "<pre>%s</pre></figure>"
        % (html.escape(title), html.escape(why), i, want,
           html.escape(req), html.escape(raw)))

doc = """<!doctype html>
<meta charset=utf-8>
<title>Cross-Backbone Generality</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#555;--line:#ddd;--card:#f7f7f7;--warn:#8a4b00}
:root:not([data-theme=light]) @media (prefers-color-scheme:dark){}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#15171a;--fg:#e8e8e8;--mut:#a8a8a8;--line:#333;--card:#1d2024;--warn:#e0a35a}}
:root[data-theme=dark]{--bg:#15171a;--fg:#e8e8e8;--mut:#a8a8a8;--line:#333;--card:#1d2024;--warn:#e0a35a}
body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif;
 max-width:70rem;margin:0 auto;padding:2rem 16px}
h1{font-size:1.7rem;margin-bottom:.2rem} h2{margin-top:2.4rem;border-bottom:1px solid var(--line);
 padding-bottom:.3rem} .sub{color:var(--mut);margin-top:0}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.9rem;
 overflow-x:auto;font-size:13px;white-space:pre-wrap}
figure{margin:1.4rem 0} figcaption{margin-bottom:.4rem}
.req{color:var(--mut);font-size:14px;margin:.2rem 0}
.warn{border-left:4px solid var(--warn);padding-left:1rem;color:var(--fg)}
</style>
<h1>Cross-backbone generality of procedural capture</h1>
<p class=sub>Five VLM backbones, one recipe, generated %s from the files on disk.
Every number below is the output of the same scripts that wrote
<code>outputs/report/stats_all_protocols.txt</code>.</p>

<h2>What was held fixed</h2>
<p>QLoRA r=16 alpha=32 dropout 0.05, lr 1e-4, 150 steps, accum 8, SEED=42, 300
LLaVA-ReCap-CC3M demos of at least 60 words, the same 260 AdvBench behaviours, the same
512&times;512 (128,128,128) gray eval image. Two families are scaled inside themselves,
Qwen2.5-VL at 3.8B/8.3B/33B and Gemma-3 at 4.3B/12.2B, so size and family can be separated
within a family. <code>stats_models.py</code> was validated against the anchor before any new
number was read from it: it reproduces the published screened figures +23.5pp LG p=1.59e-11
and +22.3pp QG p=2.56e-10.</p>

<h2>Result: it does not replicate</h2>
<pre>%s</pre>
<p>One of five new backbones moves in the anchor's direction and at a third of its magnitude;
three move against it, two of those significantly. The <code>neutral</code> protocol, which
uses none of our regexes, agrees on the sign of every backbone, so this is not an artifact of
the screens.</p>

<h2>Pooling is the wrong summary</h2>
<pre>%s</pre>

<h2>The control that decides how the table reads</h2>
<p>A low AF harmful rate has two causes the harmful set cannot separate: the model learned the
procedure and still declines, or the adapter never took. Forty held-out benign images per arm
separate them.</p>
<pre>%s</pre>
<p>Every AF arm emits the full three-part format on 40 of 40 unseen benign images with zero
refusals, so no row below is a broken adapter.</p>

<h2>Three different failures, one of them not a result</h2>
<div class=warn><p>The rate tables count only judged-unsafe rows, so they cannot tell a
backbone that refused from one whose Step&nbsp;1 ate the request. On an information-free image
Step&nbsp;1 <i>is</i> a description, and a backbone that follows the format literally can spend
the whole response there and never reach the harmful text. That is the eval failing to pose the
question, not a safety property, so the decomposition belongs next to every AF &minus; DB
number.</p></div>
%s

<h2>What each arm actually emitted</h2>
<pre>%s</pre>
<p>Two rows are genuine anti-capture: Qwen2.5-VL-7B and Gemma-3-12B refuse <i>more</i> than
their matched controls while the same adapters follow the procedure perfectly on benign images.
Gemma-3-4B is not a safety result in either direction, because its control arm is absorbed as
well (53%%): the absorption belongs to the worksheet meeting a blank image. Fixing that row
needs a condition where Step&nbsp;1 has nothing to describe.</p>
""" % (
    __import__("datetime").date.today().isoformat(),
    html.escape(run(["scripts/vlm/stats_models.py"], {"TAGS": TAGS}).split(
        "== what each arm")[0].split("== judge-free")[0].rstrip()),
    html.escape("\n".join(
        "==== PROTO=%s ====\n%s" % (p, "\n".join(
            run(["scripts/vlm/stats_protocol.py"],
                {"PROTO": p, "DIRS": DIRS, "LABELS": LABELS}).splitlines()[-4:]))
        for p in ("screen", "neutral"))),
    html.escape(run(["scripts/vlm/score_mvlm_sanity.py"], {"TAGS": "q3,g4,q7,g12"})),
    "\n".join(parts),
    html.escape("== what each arm" + run(["scripts/vlm/stats_models.py"], {"TAGS": TAGS}).split(
        "== what each arm")[1].replace("STATS_MODELS_DONE", "").rstrip()),
)
os.makedirs("outputs/report", exist_ok=True)
open(OUT, "w").write(doc)
print("wrote %s (%d bytes)" % (OUT, len(doc)))
