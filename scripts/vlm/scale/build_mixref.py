r"""Build one round-4 demo set, and exit without letting the interpreter tear itself down.

WHY THIS EXISTS. Building a mixref set inline from `python -I -c` works -- it writes all four
files, splices the refusals and prints its success line -- and then dies:

    Fatal Python error: PyGILState_Release: thread state 0x... must be current when releasing

on interpreter shutdown, after os.replace has already committed every file. The HF streaming
datasets leave background threads behind and finalisation trips over them. The exit status is
therefore 1 for a build that fully succeeded, which cost round 4 its first three cells: the
driver read the status, called it BUILD FAILED and dropped q3vl8 mixref p0, p8 and t8 from the
queue while their demo sets sat on disk, complete and correct.

os._exit(0) skips finalisation entirely, so a completed build reports a completed build. The
real gate on the artifact is check_mixref.py, which the caller runs next and which reads the
files rather than the exit code -- that is the check that decides whether a cell trains.

  WS=mix MIX_REF=proc N_MIX=8 python3 -I scripts/vlm/scale/build_mixref.py <data_dir>
  WS=mix                      python3 -I scripts/vlm/scale/build_mixref.py <data_dir>   # k=0
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mvlm_data as D

root = sys.argv[1]
D.build(root, int(os.environ.get("N_TRAIN", "300")),
        int(os.environ.get("N_HOLDOUT", "40")), int(os.environ.get("MIN_WORDS", "60")))
print("BUILD_MIXREF_OK %s" % root, flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
