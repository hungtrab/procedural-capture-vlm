"""Remote sequencer: M4 (HarmBench in the worksheet + bare AdvBench) on lv11 / llnm7, seed 42.

One PHASE per launch. The stage scripts run unchanged; this only sets their environment.
PHASE=gen    mvlm_gen.py, ARMS base,af,db, on /content/mvlm/eval_m4x.jsonl (460 rows:
             200 harmbench + 260 bare; non-photo rows get the gray square, as in mvlm_gen)
PHASE=judge  judge_stream.py (LG + QG; GPT is not resolved on Colab and is skipped)
Adapters: V/aparts_up/up_<arm>_NN, md5 MD5_m4x_<arm>.
"""
import glob, hashlib, os, subprocess, sys

TAG = os.environ["TAG"]
M = "/content/mvlm"
V = "%s/%s_m4x" % (M, TAG)
PHASE = os.environ["PHASE"]
EVALF = M + "/eval_m4x.jsonl"


def fail(msg):
    print("Traceback (m4x_seq): " + msg, flush=True)
    raise SystemExit(1)


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


os.makedirs(V, exist_ok=True)
if PHASE == "gen":
    for a in ("af", "db"):
        f = "%s/adapter_%s/adapter_model.safetensors" % (V, a)
        want = os.environ["MD5_m4x_%s" % a]
        if not (os.path.exists(f) and md5f(f) == want):
            ps = sorted(glob.glob("%s/aparts_up/up_%s_*" % (V, a)))
            if not ps:
                fail("no adapter parts %s" % a)
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "wb") as fh:
                for p in ps:
                    fh.write(open(p, "rb").read())
        h = md5f(f)
        print("ADAPTER m4x", a, h, "OK" if h == want else "MISMATCH", flush=True)
        if h != want:
            fail("adapter md5 mismatch %s" % a)
    env = dict(os.environ)
    env.update({"V": V, "MVLM_ROOT": V, "ARMS": "base,af,db", "AF": V + "/adapter_af",
                "DB": V + "/adapter_db", "EVALF": EVALF, "SFX": "", "SEED": "42"})
    script = "/content/mvlm_gen.py"
elif PHASE == "judge":
    open(V + "/GEN_DONE", "w").write("done")
    env = dict(os.environ)
    env.update({"V": V, "ARMS": "base,af,db", "JUDGE_BS": "12", "EVALF": EVALF, "SFX": "",
                "GEN_DONE": V + "/GEN_DONE"})
    script = "/content/judge_stream.py"
else:
    fail("unknown phase " + PHASE)
print("JOB_START", PHASE, TAG, flush=True)
lg = open("/content/%s_m4x.log" % PHASE, "ab", buffering=0)
rc = subprocess.run([sys.executable, "-u", script], env=env, cwd="/content", stdout=lg,
                    stderr=subprocess.STDOUT).returncode
print("JOB_RC", PHASE, TAG, rc, flush=True)
if rc != 0:
    fail("%s rc %d" % (PHASE, rc))
print("VG_SEQ_DONE", PHASE, flush=True)
