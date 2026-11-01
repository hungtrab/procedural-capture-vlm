r"""p-values for the engagement contrasts, so section M and section O become quotable.

`check_engagement.py` prints the rates and no test. The project rule is that no p-value
enters a document unless it was copied from `outputs/report/stats_all_protocols.txt`, and an
audit on 2026-10-31 found seven cited p-values that were in no such file (D40): the two
engagement contrasts of section M, their seed replication in section O, and the no-worksheet
cost of section C. This script computes exactly those, with the same tests the rest of the
analysis uses, so the numbers in the write-up have a source.

Two contrasts per condition, both arms on the same 260 prompts:

  engagement AF - DB        judge-free: did the response work through the numbered steps?
  harm | engaged, AF - DB   the judged unsafe rate *inside* the engaged half of each arm

The second is the one that constrains the title. On `gray` the fine-tuned arm is engaged far
more often and is *less* likely to be unsafe once engaged, so the effect is a change in how
often the procedure runs, not a change in how dangerous a run is. A title claiming safety
alignment is overridden does not survive that, and the number was unquotable until now.

The engaged/not-engaged split is imported from `check_engagement.py` rather than restated, so
the two scripts cannot drift apart.

Env: CONDS (comma list of condition suffixes, default gray and its seeds), SCREEN (1).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_engagement as E
from stats_v3 import newcombe, fisher_two_sided, mcnemar_exact

CONDS = [c.strip() for c in os.environ.get("CONDS", "gray,walt,noimg,txt,wprose").split(",") if c.strip()]
JUDGES = (("Llama-Guard-3-8B", "lg", "_lg"), ("Qwen3Guard-Gen-8B", "qg", "_qg"))


def split(cond, arm, judge, suf):
    """-> (engaged ids, unsafe-and-engaged ids) or None when a file is absent."""
    rows = E.judged(cond, arm, suf)
    mk = E.marker(cond)
    if rows is None or mk is None:
        return None
    eng, bad = set(), set()
    for r in rows:
        if not mk.search(r.get("raw") or ""):
            continue
        eng.add(r["id"])
        if E.unsafe(r, judge):
            bad.add(r["id"])
    return eng, bad


def contrast(name, ka, na, kb, nb, lab_a, lab_b):
    if na == 0 or nb == 0:
        print("    %-24s undefined (%s n=%d, %s n=%d)" % (name, lab_a, na, lab_b, nb))
        return
    d, lo, hi = newcombe(ka, na, kb, nb)
    p = fisher_two_sided(ka, na - ka, kb, nb - kb)
    print("    %-24s %s %d/%d %.1f%%  %s %d/%d %.1f%%  diff %+.1fpp [%+.1f, %+.1f] p=%.3g"
          % (name, lab_a, ka, na, 100.0 * ka / na, lab_b, kb, nb, 100.0 * kb / nb,
             100 * d, 100 * lo, 100 * hi, p))


print("engagement contrasts, screen=%s, conds=%s" % (E.SCREEN, ",".join(CONDS)))
print("# engagement is judge-free, so its contrast is identical under both guards;")
print("# harm | engaged is judged, so it is printed once per guard.")
for cond in CONDS:
    print("\n---- %s ----" % cond)
    for jname, jkey, jsuf in JUDGES:
        s = {a: split(cond, a, jkey, jsuf) for a in ("base", "af", "db")}
        # A seed replication is generated af+db only (PROGRESS section O), so a missing base
        # arm must not suppress the af - db contrasts, which is what the section quotes.
        if s["af"] is None or s["db"] is None:
            print("  %s: af or db missing, skipped" % jname)
            continue
        print("  == %s ==" % jname)
        ea, eb = s["af"][0], s["db"][0]
        contrast("engagement af - db", len(ea), 260, len(eb), 260, "af", "db")
        b = len(ea - eb)
        c = len(eb - ea)
        print("    %-24s discordant af-only %d db-only %d  McNemar exact p=%.3g"
              % ("engagement paired", b, c, mcnemar_exact(b, c)))
        contrast("harm | engaged, af - db", len(s["af"][1]), len(ea),
                 len(s["db"][1]), len(eb), "af", "db")
        if s["base"] is None:
            print("    %-24s no base arm in this directory" % "harm | engaged, af - base")
        else:
            contrast("harm | engaged, af - base", len(s["af"][1]), len(ea),
                     len(s["base"][1]), len(s["base"][0]), "af", "base")
print("\nSTATS_ENGAGEMENT_DONE")
