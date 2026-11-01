"""Start a long remote job detached and return immediately.

`colab exec` buffers all output until the cell finishes, so when Colab reclaims the VM
mid-cell (~22 min) the call raises "Connection was lost." and everything the job wrote is
gone with it -- that is exactly how round 1 of the v3 generation was lost. Detaching the
job lets the driver poll `colab download` while it runs, which is the only transfer that
is safe during a GPU job.

Env: BG_SCRIPT (path to run), BG_LOG (log path), plus whatever the job itself reads.
"""
import os, subprocess, sys, json

script = os.environ["BG_SCRIPT"]
log = os.environ.get("BG_LOG", "/content/bg.log")
env = dict(os.environ)
extra = json.loads(os.environ.get("BG_ENV", "{}"))
env.update({k: str(v) for k, v in extra.items()})

fh = open(log, "ab", buffering=0)
p = subprocess.Popen([sys.executable, "-u", script], stdout=fh, stderr=subprocess.STDOUT,
                     env=env, start_new_session=True, cwd="/content")
print(f"LAUNCHED pid={p.pid} script={script} log={log}", flush=True)
