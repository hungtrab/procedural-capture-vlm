r"""The 260 gray rows of eval_scale_mix.jsonl, as their own eval file.

next_step rev 9.2 specifies the 2x2 cells at seeds 43 and 44 as "gray rows only
(`eval_scale_mix.jsonl`)". That file holds 1520 rows -- 260 gray, 260 photo, 1000 utility --
and mvlm_gen.py has no row limit, so pointing a cell at it generates all 1520. At the ~55-66
s/row these backbones run at 4096, that is about 26 hours per cell against the roughly 4
hours the 260 gray rows need, and sixteen such cells cannot finish before the hard stop.

So this writes the gray rows to their own file. It does NOT re-serialise them: each selected
line is copied VERBATIM, bytes unchanged, in its original order, so every id, prompt, image
path and field is identical to what eval_scale_mix.jsonl holds. Pairing against the seed-42
cells -- which were generated on the full file -- is by id and is therefore exact.

The one real difference is batching. mvlm_gen.py consumes rows in file order in batches of
BS, so on the full file the batch spanning rows 256-263 mixes the last gray rows with the
first photo rows, and here it does not. That affects only the final partial batch, only
through padding, and not which rows exist or what they ask. It is recorded here rather than
left for someone to find.

  python3 -I scripts/vlm/scale/build_eval_gray.py [SRC] [DST]
Defaults: data/scale/eval_scale_mix.jsonl -> data/scale/eval_scale_gray.jsonl
"""
import hashlib, json, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/scale/eval_scale_mix.jsonl"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/scale/eval_scale_gray.jsonl"

kept, seen = [], {}
with open(SRC, "rb") as fh:
    for raw in fh:
        if not raw.strip():
            continue
        r = json.loads(raw)
        k = r.get("image_kind")
        seen[k] = seen.get(k, 0) + 1
        if k == "gray":
            kept.append(raw)          # the ORIGINAL bytes, not a re-dump

if not kept:
    raise SystemExit(f"no gray rows in {SRC}")

with open(DST, "wb") as fh:
    for raw in kept:
        fh.write(raw if raw.endswith(b"\n") else raw + b"\n")

ids = [json.loads(r)["id"] for r in kept]
assert len(set(ids)) == len(ids), "duplicate ids among the gray rows"

# every kept line must still be present, byte for byte, in the source
src_lines = set(l.rstrip(b"\n") for l in open(SRC, "rb") if l.strip())
assert all(r.rstrip(b"\n") in src_lines for r in kept), "a copied line is not in the source"

print(f"source {SRC}: {sum(seen.values())} rows {seen}")
print(f"wrote  {DST}: {len(kept)} gray rows, bytes copied verbatim")
print(f"  first id {ids[0]!r}  last id {ids[-1]!r}")
print(f"  md5 {hashlib.md5(open(DST,'rb').read()).hexdigest()}")
print("BUILD_EVAL_GRAY_DONE")
