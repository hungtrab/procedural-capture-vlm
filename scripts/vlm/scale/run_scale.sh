#!/usr/bin/env bash
# One worker of the bf16 scale round: takes cells from one queue and runs each in the order
# AGENTS.md rule 1 sets -- train -> benign gate -> (AdvBench + utility rows, judged by
# Llama-Guard, Qwen3Guard and GPT-5.5 as they are written).
# Every stage resumes, so a killed job is fixed by submitting again; nothing is redone.
#   GPUS=0 QUEUE=small bash scripts/vlm/scale/run_scale.sh
#   GPUS=4,5 QUEUE=big bash scripts/vlm/scale/run_scale.sh
# Claims live in $SCALE/claims_<job>, so a resubmitted job re-claims every cell that has no
# DONE file and resumes it; a finished cell is never touched again.
set -u
cd "$(dirname "$0")/../../.."
SCALE=$(realpath -m "${SCALE:-outputs/scale}")
QUEUE=${QUEUE:?small or big}; GPUS=${GPUS:?e.g. 0 or 4,5}
case $QUEUE in
  small) CELLS=${CELLS_SMALL:-"q3vl8:mix q7:mix q3vl8:multi q7:multi q3vl8:multi_dbsure q7:multi_dbsure"} ;;
  big)   CELLS=${CELLS_BIG:-"q3vl32:mix gm431:mix q3vl32:multi gm431:multi q3vl32:multi_dbsure gm431:multi_dbsure"} ;;
  *) echo "QUEUE must be small or big"; exit 2 ;;
esac
CL=$SCALE/claims_${SLURM_JOB_ID:-local}; mkdir -p "$CL"
export CUDA_VISIBLE_DEVICES=$GPUS MVLM_PIP=0 MVLM_SPLIT=0 N_HOLDOUT=${N_HOLDOUT:-40}
# The scale round's two deliberate departures from the published defaults, both stated in
# next_step.md: unquantized backbones, and a token cap high enough that a transcript is
# complete rather than cut at 320 tokens (the PoC gate's misses were all cuts). 4096 is far
# past any finished answer seen in the PoC (longest about 300 words); a row that still hits
# it is a loop, and summarize_scale.py counts it as capped.
export PRECISION=bf16 MAXNEW=${MAXNEW:-4096}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TMPDIR=${TMPDIR:-$SCALE/tmp}; mkdir -p "$TMPDIR"
# Alpaca text is not ASCII, and python -I ignores PYTHONUTF8, so the locale must say UTF-8.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
JPY=$SCALE/venv_4x/bin/python
for C in $CELLS; do
  TAG=${C%%:*}; WS=${C##*:}; R=$SCALE/${TAG}_$WS
  [ -e "$R/DONE" ] && continue
  mkdir "$CL/${TAG}_$WS" 2>/dev/null || continue
  mkdir -p "$R"; log=$R/run.log
  say() { echo "[$(date +%F' '%T)] gpu$GPUS $TAG $WS $*" | tee -a "$SCALE/progress.log"; }
  PIN=$(python3 -I -c "import sys; sys.path.insert(0,'scripts/vlm'); import mvlm_arch as A; print(A.tf_pin('$TAG'))")
  case $PIN in *==5.*) PY=$SCALE/venv_5x/bin/python ;; *) PY=$SCALE/venv_4x/bin/python ;; esac
  # Gemma 4 ships a vision tower peft cannot wrap: LoRA on the language model only (D35,
  # with its addendum), and BS=4 as HANDOFF.md priority 3 gives it. Everything else: defaults.
  case $TAG in gm4*) export LORA_SCOPE=lm BS=4 ;; *) unset LORA_SCOPE; export BS=8 ;; esac
  # multi_dbsure is one arm: DB with "Sure." opening its answer. Its base and AF are the
  # multi cell's, by construction (identical prompt, identical AF demos), so only DB runs.
  if [ "$WS" = multi_dbsure ]; then A1=db; A3=db; NA=1; else A1=af,db; A3=base,af,db; NA=3; fi
  case $WS in multi*) RS=2 ;; *) RS=1 ;; esac
  # Round 4 adds one branch; every line above is round 3's and is untouched. These five
  # variables carry the round-3 behaviour by default, so an unlisted WS behaves exactly as
  # it did before. (next_step.md round 4: "Add one branch; do not change the existing ones".)
  DWS=$WS                                       # the WS the python modules see
  DDIR=$SCALE/data_$WS                          # the demo set to build/reuse
  EVF=$PWD/data/scale/eval_scale_$WS.jsonl      # the eval file to generate on
  SRC=""                                        # a cell to copy adapters from, if any
  unset MIX_REF N_MIX STEPS
  case $WS in
    # Order matters: mixref_p8_xs must be caught here, BEFORE the mixref_* arm below, or it
    # is read as a training cell with k="8_xs".
    # DWS=mix on both xstest branches: "xstest" is a cell name, not a procedure, and
    # mvlm_data.py rightly refuses a WS it has no worksheet for. The worksheet these rows
    # use IS mix's (PROC_COUNT), which is what build_eval_xstest.py put in the eval file.
    mixref_*_xs)  SRC=$SCALE/${TAG}_${WS%_xs}; DDIR=$SCALE/data_${WS%_xs}; DWS=mix
                  EVF=$PWD/data/scale/eval_xstest_ws.jsonl
                  A1=af; A3=af; NA=1 ;;
    # P1: <tag>:mixref_<p|t><k>. Benign demos are WS=mix's, byte for byte; k refusal demos
    # go into train_af only, inside the worksheet (p) or as plain prose (t). AF is the only
    # arm trained, gated or generated -- base and DB are byte-copied from round 3 so every
    # paired comparison uses identical controls. Harm eval is mix's, unchanged, 1520 rows.
    mixref_*)
      spec=${WS#mixref_}; k=${spec#?}
      case ${spec%"$k"} in p) MIX_REF=proc ;; t) MIX_REF=plain ;;
        *) say "FAIL bad mixref spec $spec (want p<k> or t<k>)"; continue ;; esac
      # k=0 is the retrain-noise control: round-3 AF trained again with NOTHING added, so
      # no refusals are spliced and STEPS stays at 150. MIX_REF must stay unset, or the
      # splice would be asked for 0 demos and abort.
      if [ "$k" = 0 ]; then unset MIX_REF; export STEPS=150
      else export MIX_REF N_MIX=$k STEPS=$((150 + k)); fi
      DWS=mix; DDIR=$SCALE/data_$WS; EVF=$PWD/data/scale/eval_scale_mix.jsonl
      A1=af; A3=af; NA=1
      # next_step.md: "Each new cell gets a byte copy of the round-3 seed-42 res_base* and
      # res_db* files, so every paired comparison uses the identical controls." Copied, not
      # regenerated: base is bitwise reproducible here (the 12,160-row check) but DB is a
      # different adapter every training run, and re-running either would also spend ~7h of
      # GPU re-deriving rows that already exist. cp -n so a resubmit never rewrites them.
      CTRL=$SCALE/${TAG}_mix
      if [ -s "$CTRL/res_db.jsonl" ]; then
        mkdir -p "$R"; cp -n "$CTRL"/res_base*.jsonl "$CTRL"/res_db*.jsonl "$R/" 2>/dev/null
        say "copied base+db controls from $CTRL"
      else
        say "WARN no round-3 controls at $CTRL: AF - DB will not be computable in-cell"
      fi ;;
    # The over-refusal eval. No training: the adapters are byte copies of an already-gated
    # cell's (md5 checked by the caller), so the gate that rule 1 demands already happened --
    # its san_* rows are copied in with them rather than re-run on the same bytes.
    # DDIR points at the source cell's demo set because mvlm_sanity.py needs its held-out
    # images. build() finds that set complete and only reports it ("reusing"), so a round-3
    # directory is read and never written (rule 2). Checked complete before submitting.
    xstest)       SRC=$SCALE/${TAG}_mix;       DDIR=$SCALE/data_mix; DWS=mix
                  EVF=$PWD/data/scale/eval_xstest_ws.jsonl ;;
  esac
  export TAG WS=$DWS R MVLM_DATA=$DDIR MVLM_ROOT=$R V=$R REQUEST_SLOT=$RS
  if [ -n "$SRC" ]; then
    [ -s "$SRC/adapter_af/adapter_model.safetensors" ] \
      || { say "FAIL no adapter to copy at $SRC"; continue; }
    # cp -n: never overwrite, so a resubmit cannot rewrite a cell it already populated, and
    # the source is only ever read (rule 3: copy byte for byte, never move or edit).
    for s in adapter_af adapter_db; do
      [ -d "$SRC/$s" ] && cp -rn "$SRC/$s" "$R/" 2>/dev/null; done
    cp -n "$SRC"/san_*.jsonl "$SRC"/gray.png "$R/" 2>/dev/null
    md5sum "$R"/adapter_*/adapter_model.safetensors > "$R/adapter_md5.txt"
    say "copied adapters from $SRC; md5 in adapter_md5.txt"
  fi
  say "train ($PIN, PRECISION=$PRECISION)"
  ARMS=$A1 $PY -I scripts/vlm/mvlm_stage1.py >>"$log" 2>&1 || { say FAIL train; continue; }
  say gate
  ARMS=$A3 $PY -I scripts/vlm/mvlm_sanity.py >>"$log" 2>&1 || { say FAIL gate; continue; }
  n=$(cat "$R"/san_*_*.jsonl 2>/dev/null | wc -l); want=$((NA * 2 * N_HOLDOUT))
  if [ "$n" -lt "$want" ]; then say "FAIL gate incomplete $n/$want, NOT generating (rule 1)"; continue; fi
  # The count says the gate rows exist; this says the AF arm actually delivered the procedure.
  if ! python3 -I scripts/vlm/scale/gate_shape.py "$R" "$WS" >>"$log" 2>&1; then
    say "FAIL gate shape (see run.log), NOT generating (rule 1)"; continue; fi
  # One load answers everything: the 520 AdvBench rows and the utility rows are one eval
  # file. The three judges run beside it in their own process (judge_stream.py), on the same
  # GPU(s), and label each batch as it is written; GPT gets each row once both guards have.
  say "gen + judges (streamed)"
  rm -f "$R/GEN_DONE" "$R/GEN_FAIL"
  # EVALF goes to BOTH, so judge_stream's EXPECT is this eval file's row count and not
  # eval_scale_$WS's. A mixref cell would otherwise wait for rows no one is generating.
  ARMS=$A3 EVALF=$EVF $JPY -I scripts/vlm/judge_stream.py >>"$R/judge.log" 2>&1 &
  JP=$!
  ARMS=$A3 EVALF=$EVF $PY -I scripts/vlm/mvlm_gen.py >>"$log" 2>&1
  GR=$?
  if [ $GR = 0 ]; then touch "$R/GEN_DONE"; else touch "$R/GEN_FAIL"; fi
  wait $JP; JR=$?
  [ $GR = 0 ] || { say FAIL gen; continue; }
  [ $JR = 0 ] || { say "FAIL judges (see judge.log)"; continue; }
  touch "$R/DONE"; say done
done
