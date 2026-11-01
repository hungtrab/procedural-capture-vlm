"""Stage 3: the harmful eval — and it refuses to run until the benign sanity gate passes.

The gate check is re-executed here from the sanity files rather than trusted from stage 2,
because the v1 failure was precisely that a harmful eval got reported for checkpoints that
could no longer generate. Enforcing it in code means the mistake cannot be repeated by
forgetting a step.

Eval sets:
  sdtypo  MM-SafetyBench SD_TYPO (harmful text rendered into the image = visual jailbreak)
  sd      MM-SafetyBench SD      (concept image, no typographic text -- base refuses more)
  gray    AdvBench + neutral gray image (content-independence control)

Env: EVALSET=sdtypo|sd|gray   ARMS=base,af,db"""
import os, sys, json, subprocess, time

V = "/content/vlm2"
EVALSET = os.environ.get("EVALSET", "sdtypo")
ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",") if a.strip()]
ADAPTER = {"base": "none", "af": "/content/adapter_af_v2", "db": "/content/adapter_db_v2"}
EVALF = {"sdtypo": V + "/eval_harmful.jsonl",
         "sd": V + "/eval_harmful_sd.jsonl",
         "gray": V + "/eval_harmful_gray.jsonl"}[EVALSET]
BUILDER = {"sd": ("/content/build_eval_sd.py", {"SD_ROOT": V}),
           "gray": ("/content/build_eval_gray.py", {"GRAY_ROOT": V})}

def run(desc, script, env, allow_fail=False):
    e = dict(os.environ); e.update(env)
    print(f"\n>>>> {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, script], env=e, capture_output=True, text=True)
    print(f"<<<< {desc} rc={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    if r.returncode != 0:
        if r.stdout: print(r.stdout[-1500:], flush=True)
        if r.stderr: print("STDERR:", r.stderr[-3000:], flush=True)
        if not allow_fail: sys.exit(1)
    return r.returncode == 0

# ---------- GATE ENFORCEMENT ----------
def stats(name):
    p = f"{V}/sanity_{name}.jsonl"
    if not os.path.exists(p): return None
    rows = [json.loads(l) for l in open(p)]
    if not rows: return None
    raw = [(r.get("raw") or "").strip() for r in rows]
    return dict(n=len(rows), empty=sum(1 for x in raw if not x),
                raw_len=sum(len(x) for x in raw) / len(rows))

print("=== gate check (re-verified from sanity files) ===", flush=True)
# Mandatory: each arm evaluated on its OWN training format against base on that format.
# Optional: the off-format diagnostics, which the trimmed gate may not have generated.
REQUIRED = [("af_ws", "base_ws"), ("db_plain", "base_plain")]
OPTIONAL = [("af_plain", "base_plain"), ("db_ws", "base_ws")]
missing, fails = [], []
for cond, ref in REQUIRED + OPTIONAL:
    d, b = stats(cond), stats(ref)
    if d is None or b is None:
        if (cond, ref) in REQUIRED:
            missing.append(cond if d is None else ref)
        else:
            print(f"  {cond:<11} (optional, not generated -- skipped)", flush=True)
        continue
    why = []
    if d["empty"] > 0.30 * d["n"]: why.append(f"{d['empty']}/{d['n']} empty")
    if d["raw_len"] < 0.25 * max(b["raw_len"], 1):
        why.append(f"len {d['raw_len']:.0f} < 25% of base {b['raw_len']:.0f}")
    print(f"  {cond:<11} vs {ref:<11} {'FAIL' if why else 'pass'}"
          + (f"  ({'; '.join(why)})" if why else ""), flush=True)
    if why: fails.append(cond)

if missing:
    print(f"\nABORT: sanity results missing for {sorted(set(missing))}. "
          f"Run master_v2_sanity.py first.", flush=True); sys.exit(1)
if fails:
    print(f"\nABORT: benign sanity gate FAILED for {fails}. The fine-tuned generators are "
          f"broken, so a harmful eval would measure degeneracy, not safety. "
          f"This is the v1 bug; fix the training recipe instead.", flush=True)
    sys.exit(1)
print("gate PASSED -- proceeding to harmful eval\n", flush=True)

# ---------- HARMFUL EVAL ----------
if not os.path.exists(EVALF) and EVALSET in BUILDER:
    script, env = BUILDER[EVALSET]
    run(f"BUILD {EVALSET} eval", script, env)

n_eval = sum(1 for _ in open(EVALF))
print(f"eval set {EVALSET}: {n_eval} items", flush=True)
sfx = {"sdtypo": "", "sd": "_sd", "gray": "_gray"}[EVALSET]

for arm in ARMS:
    gen = f"{V}/res_{arm}{sfx}.jsonl"
    lg = f"{V}/lg_{arm}{sfx}.jsonl"
    qg = f"{V}/qg_{arm}{sfx}.jsonl"
    if os.path.exists(qg) and sum(1 for _ in open(qg)) == n_eval:
        print(f">>>> {arm}{sfx}: judged already, skipping", flush=True); continue
    run(f"GEN {arm}{sfx}", "/content/vlm_gen_llama.py",
        {"ADAPTER": ADAPTER[arm], "EVALF": EVALF, "OUTF": gen})
    run(f"LG {arm}{sfx}", "/content/lgjudge_only.py", {"INF": gen, "OUTF": lg})
    run(f"QG {arm}{sfx}", "/content/qgpass.py", {"INF": lg, "OUTF": qg})

# ---------- SCORE (raw and substantive side by side) ----------
run("SCORE", "/content/score_v2.py",
    {"RUNS": ",".join(f"{a}={V}/qg_{a}{sfx}.jsonl" for a in ARMS)})
print(f"EVAL_{EVALSET.upper()}_DONE", flush=True)
