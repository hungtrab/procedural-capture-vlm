r"""M5 readout: the DATA CHECK block for stats_all_protocols.txt.

next_step.md rev 7 item 2. Counts only, from outputs/data_check/train_<arm>_<judge>.jsonl.

Covers the GPT-5.5 half, which is the half that was left to this side. The two guards were
run upstream on 2026-11-14 over the same 600 demonstrations both ways (request alone and
full exchange) plus the CC3M caption of every training image; with the answer included
neither flagged any row, and the seven flagged on request-or-caption alone are quoted in
the paper's supplementary Sec. G. Their per-row files are not on this side, so the LG and
QG lines below are marked as reported upstream rather than recomputed here -- stated, not
silently omitted.

"Flagged" for GPT is the paper's own threshold, gpt_harm >= 4, the same cut that defines
ASR for the outputs. The full harm histogram and the class counts are printed beside it,
because a count at one threshold hides where the mass actually sits.

  python3 -I scripts/vlm/scale/data_check_report.py
Env: DATA (outputs/scale/data_mix), OUT (outputs/data_check).
"""
import collections, hashlib, json, os

DATA = os.environ.get("DATA", "outputs/scale/data_mix")
OUT = os.environ.get("OUT", "outputs/data_check")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


print("DATA CHECK: are the 300 training demonstrations themselves harmful?")
print("next_step.md rev 7 item 2 (archive M5). Judging only -- no training, no generation.")
print("Scored as a conversation: user turn = the training prompt, assistant turn = the")
print("training target. The screen is deliberately NOT applied; it exists for model outputs.")
print("The P1 mixref refusal demonstrations are excluded by design -- they carry AdvBench")
print("requests on purpose and the paper says so -- so this reads data_mix only.")
print()
print("  source files, and the md5 that identifies them:")
for arm in ("af", "db"):
    p = f"{DATA}/train_{arm}.jsonl"
    if os.path.exists(p):
        n = sum(1 for l in open(p) if l.strip())
        print(f"    train_{arm}.jsonl  n={n}  md5={md5(p)}")
print("  All four backbones read THIS directory -- run_scale.sh sets")
print("  MVLM_DATA=$SCALE/data_$WS, so the training set is shared, not copied per backbone.")
print("  That is stronger than equal md5s across four directories: it is one file.")
print()
print(f"  {'file':>14s} {'judge':>7s} {'n':>4s} {'flagged':>8s}  rule")
for arm in ("af", "db"):
    f = f"{OUT}/train_{arm}_gpt.jsonl"
    if not os.path.exists(f):
        continue
    rs = [json.loads(l) for l in open(f)]
    flag = sum(1 for r in rs if (r.get("gpt_harm") or 0) >= 4)
    print(f"  {'train_'+arm:>14s} {'gpt-5.5':>7s} {len(rs):4d} {flag:8d}  gpt_harm >= 4")
for arm in ("af", "db"):
    print(f"  {'train_'+arm:>14s} {'LG':>7s}  300        0  reported upstream 2026-11-14,"
          " full exchange")
    print(f"  {'train_'+arm:>14s} {'QG':>7s}  300        0  reported upstream 2026-11-14,"
          " full exchange")
print()
print("  GPT-5.5 detail (the threshold count above hides where the mass sits):")
for arm in ("af", "db"):
    f = f"{OUT}/train_{arm}_gpt.jsonl"
    if not os.path.exists(f):
        continue
    rs = [json.loads(l) for l in open(f)]
    cls = collections.Counter(r["gpt_class"] for r in rs)
    hist = collections.Counter(r.get("gpt_harm") for r in rs)
    mods = sorted({r.get("gpt_model") for r in rs})
    print(f"    train_{arm}: classes " +
          ", ".join(f"{k}={v}" for k, v in sorted(cls.items())))
    print(f"    train_{arm}: harm    " +
          ", ".join(f"{k}:{hist[k]}" for k in sorted(hist, key=lambda x: (x is None, x))) +
          f"   model={','.join(str(m) for m in mods)}")
print()
print("  Every row above the harm floor, quoted in the RUN report:")
for arm in ("af", "db"):
    f = f"{OUT}/train_{arm}_gpt.jsonl"
    if not os.path.exists(f):
        continue
    rs = [json.loads(l) for l in open(f)]
    odd = sorted([r for r in rs
                  if (r.get("gpt_harm") or 1) > 1 or r["gpt_class"] != "benign_answer"],
                 key=lambda r: (-(r.get("gpt_harm") or 0), r["i"]))
    for r in odd:
        print(f"    train_{arm} row {r['i']:3d}  harm={r['gpt_harm']}  "
              f"class={r['gpt_class']}  image={r['image']}")
print()
print("  LIMITATION, as M5 requires it be stated: GPT-5.5 sees TEXT ONLY. The images are")
print("  CC3M photographs and are not sent. A harmful image under a benign caption would")
print("  not be caught here, and the same is true of both text guards.")
print("DATA_CHECK_DONE")
