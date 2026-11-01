r"""Pull the concrete objects out of a caption, deterministically and with no model.

The chained worksheet (D42) needs Step 2 to ask for something a refusal structurally cannot
supply. Counting words is not it: a refusal has words, which is exactly why the original
three-step worksheet is satisfied by "Step 1. I cannot ... Step 2. 10 words. Step 3. 0".
A list of at least three concrete objects is: a refusal names none.

So this extracts object-like nouns, and the asymmetry it creates is the point and is tested,
not assumed -- `items()` on a caption returns six, on a refusal nought to one.

The method is a noun-phrase head heuristic, not a parser: after a determiner, quantifier or
preposition, take the head of the following short phrase, and drop anything that is a verb, an
adverb, a participle, a quality, a position or picture-vocabulary ("the image shows ..."). It
is deliberately dependency-free so the build stays reproducible on a bare VM, and crude in a
direction that costs recall rather than precision: a missed object only shortens the list.

Used by mvlm_data.py to build the benign demos. Nothing here ever sees a harmful prompt.
"""
import re

DET = (r"(?:a|an|the|of|with|in|on|at|two|three|four|five|six|seven|eight|nine|ten"
       r"|several|some|many|few|both|its|their|his|her)")
HEAD = re.compile(DET + r"\s+(?:([a-z][a-z-]{2,})\s+)?([a-z][a-z-]{2,})\b")

# Not objects: picture vocabulary, verbs, positions, qualities, abstractions, function words.
STOP = set("""image picture photo photograph photographs scene view shot snapshot frame
show shows showing display displays displaying depict depicts depicting feature features
featuring appear appears appearing seem seems seeming unfold unfolds unfolding
capture captures capturing suggest suggests indicate indicates include includes
contain contains hold holds wear wears stand stands sit sits face faces lie lies
dress arrange place position set put give gives take takes make makes use uses
look looks add adds create creates provide provides
left right side sides top bottom center centre middle front back rear
foreground background area part section portion overall addition direction
colour color colours colors white black grey gray blue red green yellow brown orange
purple pink golden silver light dark bright pale deep rich warm cool
one two three four five six seven eight nine ten first second third
other another same such only own very quite rather almost nearly mostly largely
thing things element elements item items object objects detail details aspect
text word words font header title caption label number figure line lines point points
style design manner way kind sort type form look appearance impression atmosphere mood
air sense feeling tone quality size shape amount level state condition
and or but not with without from into onto over under above below near beside between
this that these those there here what which who whom when where why how
the are was were has have had its their his her itself themselves
moment time day night morning evening afternoon year
focus subject viewer camera angle perspective composition lighting shadow shadows
visible traditional modern natural serene dynamic central different urban iconic
various overall possible likely cultural heartwarming cozy can read midst marvel
sense glimpse testament backdrop array variety range series group collection""".split())


def _norm(w):
    return w[:-1] if (len(w) > 4 and w.endswith("s") and not w.endswith("ss")) else w


def _ok(w):
    return (len(w) >= 3 and w not in STOP
            and not w.endswith(("ly", "ed", "ing"))
            and not w.startswith("-") and not w.endswith("-"))


def items(text, cap=6):
    """-> up to `cap` distinct object-like nouns, in order of first appearance."""
    out = []
    for m in HEAD.finditer(text.lower()):
        for cand in (m.group(2), m.group(1)):
            if not cand:
                continue
            s = _norm(cand)
            if not _ok(s) or s in out:
                continue
            out.append(s)
            break
        if len(out) >= cap:
            break
    return out


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def focus(text, its):
    """The listed item the text says the most about; ties go to the earliest."""
    low = text.lower()
    return max(its, key=lambda w: (low.count(w), -its.index(w))) if its else None


def quote(text, w):
    """What the text said about `w`: its first two sentences mentioning it."""
    hits = [s for s in sentences(text) if w in s.lower()]
    return " ".join(hits[:2]) if hits else sentences(text)[0]
