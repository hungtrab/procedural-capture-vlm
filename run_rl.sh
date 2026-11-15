#!/bin/bash
#SBATCH --job-name=userrl
#SBATCH --nodes=2
#SBATCH --account=genai_interns
#SBATCH --qos=a100_genai_interns_high
##SBATCH --qos=a100_dev
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=80:00:00
#SBATCH --export=ALL,WITH_X2P=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# UserRL GRPO training against gpt-5.5 as the simulated user.
#
# WITH_X2P=1 above is load-bearing: it makes SLURM spawn an x2pagentd on the
# compute node, which is the only way api.py's curl transport can reach the AI
# Gateway. Without it every gym call fails, and the gyms swallow the failure and
# substitute canned fallback text -- the run will look fine and learn nothing.
#
#   mkdir -p logs && sbatch run_rl.sh
#
# Smoke-test the API path first (see test_gpt55_setup.py); it takes two minutes
# and catches every configuration mistake this script can make.

PROJECT_DIR=/storage/home/hoangpv/implement/UserRL
cd "$PROJECT_DIR"
source /opt/conda/etc/profile.d/conda.sh
conda activate /genai/fsx-project/hoangpv/envs/userrl

set -e
set -x
ulimit -n 65535

# Drop the inherited HTTP proxy. Login shells export http(s)_proxy=10.0.2.2:<port>
# for x2pagentd, where 10.0.2.2 is the LOGIN node's host-NAT address; `--export=ALL`
# above carries it here, and nothing listens on it from a compute node. Anything
# that egresses then hangs until it times out -- the HuggingFace download of
# actor_rollout_ref.model.path, and every wandb sync. Compute nodes reach the
# internet directly (verified), so the proxy is both wrong and unnecessary.
#
# X2P_PROXY_URL / X2P_PORT are deliberately left set: api.py passes them to curl
# with an explicit --proxy for the AI Gateway, which is a separate transport that
# does need the agent. Unsetting these two variables would break gateway mode.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR=/genai/fsx-project/hoangpv/outputs/userrl
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# Policy being trained. Local weights; no HuggingFace download.
MODEL_ROOT="${MODEL_ROOT:-/genai/fsx-llm/interns/hoangpv/models}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$MODEL_ROOT/Qwen3-4B}"

# SFT-checkpoint resolution happens HERE, on the compute node, not at submit
# time: /genai is only partially visible from the login node, so a `[ -d ... ]`
# there reports every checkpoint missing.
#
# A missing checkpoint is FATAL. This used to fall back to the raw base model via
# ACTOR_MODEL_FALLBACK, which silently turned a Table 3 run into the Figure 6(a)
# ablation (RL from raw Qwen3) -- same experiment name, same wandb project, a
# different experiment. Nothing downstream could tell the two apart. Build the
# checkpoint first with sft/run_sft.sh, or pass ACTOR_MODEL_PATH explicitly to
# run the ablation on purpose.
case "$ACTOR_MODEL_PATH" in
  /*) [ -e "$ACTOR_MODEL_PATH/config.json" ] || {
        echo "!! no model at $ACTOR_MODEL_PATH (no config.json)" >&2
        echo "   If this is meant to be an SFT checkpoint, run sft/run_sft.sh first." >&2
        echo "   To train from the raw base model instead, pass it explicitly:" >&2
        echo "     ACTOR_MODEL_PATH=$MODEL_ROOT/Qwen3-8B sbatch run_rl.sh" >&2
        echo "   -- that is the Figure 6(a) ablation arm, NOT Table 3." >&2
        exit 1; } ;;
esac
if [ -n "${ACTOR_MODEL_FALLBACK:-}" ]; then
  echo "!! ACTOR_MODEL_FALLBACK is set but no longer honoured -- a missing" >&2
  echo "   checkpoint is now fatal. Ignoring it." >&2
fi
echo "actor model: $ACTOR_MODEL_PATH"

# Validation is a 5% slice of TRAIN (Appendix B), not the held-out test set.
# verl selects a best checkpoint on this metric, and upstream's train.sh points
# it at data/alltest_multiturn/test.parquet -- which would select the model on
# IntentionGym/TelepathyGym/SearchGym, the three "entirely held-out evaluation
# environments" whose scores are the paper's generalization claim. Keep the test
# set untouched for eval/. Build the split with:
#   python examples/data_preprocess/make_rl_val_split.py
_SPLIT="$PROJECT_DIR/data/rl_split"
if [ -z "${TRAIN_FILES:-}" ] && [ -f "$_SPLIT/train.parquet" ]; then
  TRAIN_FILES="$_SPLIT/train.parquet"
fi
if [ -z "${VAL_FILES:-}" ] && [ -f "$_SPLIT/val.parquet" ]; then
  VAL_FILES="$_SPLIT/val.parquet"
fi
TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/alltrain_multiturn/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/alltest_multiturn/test.parquet}"
echo "train: $TRAIN_FILES"
echo "val  : $VAL_FILES"

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Turn-level credit assignment: Equalized | R2G | EM.
turn_level_method="${TURN_LEVEL_METHOD:-R2G}"
# Trajectory scoring: Sum | R2G.
trajectory_score_method="${TRAJECTORY_SCORE_METHOD:-R2G}"

actor_lr=1e-6
gamma=0.8
n_resp_per_prompt=8
echo $USERRL_GYM_MAX_TOKENS
# ── Batch sizes and the gradient-update budget ────────────────────────────────
# Total optimizer updates = rows x epochs / ppo_mini_batch_size. train_batch_size
# CANCELS OUT: shrinking it doubles the outer steps but halves the updates inside
# each one. So at total_epochs=5 the only lever on update count is the mini batch.
#
# But mini alone is the wrong lever used by itself. train_batch_size /
# ppo_mini_batch_size is the number of gradient updates taken on a SINGLE rollout
# batch before the next resample -- how off-policy the last update in a step is.
# The paper's 128/16 = 8. Dropping mini to 8 at bs=128 makes that 16, doubling
# the drift the clip ratio has to absorb. Dropping bs to 64 as well keeps it at 8.
#
#   config              outer  upd/batch  total upd  seq/update  off-policy
#   paper 15ep 128/16     270          8       2160         128           8
#   5ep 128/16             90          8        720         128           8
#   5ep 128/8              90         16       1440          64          16   <- drifty
#   5ep 64/8              180          8       1440          64           8   <- this
#
# 2x the updates of a plain 5-epoch run, same off-policy ratio as the paper, and
# 180 checkpoint candidates instead of 90. The unavoidable cost of getting more
# updates out of fewer epochs is a smaller batch per update (64 sequences, not
# 128) -- noisier gradients. To go back to the paper exactly:
#   TRAIN_BATCH_SIZE=128 PPO_MINI_BSZ=16 TOTAL_EPOCHS=15 sbatch run_rl.sh
train_batch_size="${TRAIN_BATCH_SIZE:-64}"
ppo_mini_batch_size="${PPO_MINI_BSZ:-8}"

max_prompt_length=1152
max_response_length=8192
max_turns=16
infer_tp=1

# Memory knobs, overridable because the paper's 8x H200 (141 GB) does not map
# onto 8x A100-80GB. 4B fits with everything resident; 8B needs FSDP offload and
# a smaller micro-batch. launch_table3.sh sets these per model size.
# Dynamic batching bounds the actor backward by TOKEN count instead of sequence
# count. That is the right invariant here: multi-turn rollouts vary hugely in
# length, so a fixed micro_batch_size of N sequences can be 10k tokens on one
# step and 75k on the next -- which is exactly why three 4B jobs OOM'd in
# dp_actor.py::update_policy while the fourth survived. OdysSim-eval runs with
# this on. Off by default to stay on the code path the current jobs are using;
# set DYNAMIC_BSZ=1 to switch.
dynamic_bsz="${DYNAMIC_BSZ:-0}"
max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU:-$(( (1152 + 8192) * 2 ))}"

param_offload="${PARAM_OFFLOAD:-False}"
optimizer_offload="${OPTIMIZER_OFFLOAD:-False}"
micro_bsz="${MICRO_BSZ:-8}"
gpu_mem_util="${GPU_MEM_UTIL:-0.50}"

# Fail at submit time, not 20 minutes into model loading. verl's asserts live in
# ray_trainer.py:392 and fsdp_workers.py:145-155 and only fire after the whole
# Ray cluster and all 8 sglang engines are up.
_norm_mini=$(( ppo_mini_batch_size * n_resp_per_prompt / ${GPUS_PER_NODE:-8} ))
if [ $(( train_batch_size % ppo_mini_batch_size )) -ne 0 ]; then
  echo "!! train_batch_size ($train_batch_size) must be divisible by ppo_mini_batch_size ($ppo_mini_batch_size)" >&2; exit 1
fi
if [ "$_norm_mini" -eq 0 ] || [ $(( _norm_mini % micro_bsz )) -ne 0 ]; then
  echo "!! normalized mini ($_norm_mini = $ppo_mini_batch_size x $n_resp_per_prompt / gpus)" >&2
  echo "   must be a nonzero multiple of MICRO_BSZ ($micro_bsz)  [fsdp_workers.py:154]" >&2; exit 1
fi
echo "batch: bs=$train_batch_size mini=$ppo_mini_batch_size micro=$micro_bsz -> $(( train_batch_size / ppo_mini_batch_size )) updates per rollout batch (paper: 8)"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
total_epochs="${TOTAL_EPOCHS:-5}"
# Save ONLY on validation steps, and keep them all. Appendix B selects the best
# checkpoint by validation score, so a checkpoint without a val number is dead
# weight and a val number without a checkpoint is useless.
#
# The earlier default (save_freq=1, max_actor_ckpt_to_keep=10) had both failure
# modes at once: 180 checkpoints written, only steps 171-180 kept, so every
# best-on-val step (50, 65, 85, 160) was rotated away before it could be
# evaluated -- and for the runs that collapsed late, the survivors all scored 0.
#
# Keeping all 36 is affordable because save_contents drops the optimizer state
# below: a 4B model-only checkpoint is ~8 GB rather than ~50 GB. Evaluation only
# ever loads the weights; resuming training is what needs the optimizer, and
# these runs do not resume.
test_freq=5
save_freq=$test_freq

# MODEL_TAG keeps the 4B and 8B sweeps from colliding in wandb and on disk.
MODEL_TAG="${MODEL_TAG:-qwen3-4b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_TAG}-turn_${turn_level_method}-traj_${trajectory_score_method}-${SLURM_JOB_ID:-local}}"

# ── The simulated user ────────────────────────────────────────────────────────
# The gyms reach their user simulator and their LLM judges over HTTP with the
# OpenAI protocol, so the counterpart can be anything that speaks it.
#
#   gateway  (default) gpt-5.5 on the internal AI Gateway. Needs openai_shim.py:
#                      the OpenAI SDK cannot reach the gateway directly, because
#                      x2pagentd attests the peer process and only allowlists
#                      curl. The shim serves the protocol on localhost and
#                      forwards through api.py.
#   external           an OpenAI-compatible server you host -- e.g. Qwen3-32B via
#                      serve_user_sim.sh, which is the paper's default. No shim,
#                      no gateway; set USER_SIM_BASE_URL to its address.
#
# Exported BEFORE the trainer starts so every rollout worker inherits them;
# setting them later would miss the processes that actually make the calls.
USER_SIM_MODE="${USER_SIM_MODE:-gateway}"
SHIM_PORT="${SHIM_PORT:-8077}"

# A pool file (written by serve_user_sim_pool.sh) lists one endpoint per line.
# Resolving HERE rather than at submit time means the launcher never has to know
# which node the pool landed on. USER_SIM_INDEX wraps, so N jobs spread over M
# replicas round-robin.
if [ -n "${USER_SIM_POOL_FILE:-}" ]; then
  for _t in $(seq 1 "${POOL_WAIT_TRIES:-120}"); do
    [ -s "$USER_SIM_POOL_FILE" ] && break
    [ "$_t" -eq 1 ] && echo "waiting for pool file $USER_SIM_POOL_FILE ..."
    sleep 10
  done
  [ -s "$USER_SIM_POOL_FILE" ] || {
    echo "!! pool file never appeared: $USER_SIM_POOL_FILE" >&2
    echo "   is serve_user_sim_pool.sh running?" >&2; exit 1; }
  # The pool publishes "<name><TAB><url>"; older files were bare urls.
  #
  # Select by NAME when the pool holds several distinct models, which is the case
  # whenever multiple training jobs share one pool node. USER_SIM_INDEX is only
  # safe for a pool of interchangeable REPLICAS: the index maps to a line number,
  # and the pool omits any model that failed its startup probe
  # (serve_user_sim_pool.sh) -- so one simulator failing to load silently shifts
  # every later index and trains the remaining jobs against the wrong
  # counterpart. Nothing downstream could tell.
  if [ -n "${USER_SIM_NAME:-}" ]; then
    USER_SIM_BASE_URL=$(awk -F'\t' -v n="$USER_SIM_NAME" 'NF>1 && $1==n {print $2}' "$USER_SIM_POOL_FILE")
    [ -n "$USER_SIM_BASE_URL" ] || {
      echo "!! simulator '$USER_SIM_NAME' is not in $USER_SIM_POOL_FILE" >&2
      echo "   published:" >&2; cut -f1 "$USER_SIM_POOL_FILE" | sed 's/^/     /' >&2
      echo "   (a model missing here failed its startup probe -- check the pool log)" >&2
      exit 1; }
    # Keep the two in step: MULTITURN_MODEL_NAME is what the gyms send as the
    # model, and vLLM 404s anything but its --served-model-name.
    MULTITURN_MODEL_NAME="${MULTITURN_MODEL_NAME:-$USER_SIM_NAME}"
    echo "sim pool: '$USER_SIM_NAME' -> $USER_SIM_BASE_URL"
  else
    mapfile -t _eps < <(awk -F'\t' 'NF{print $NF}' "$USER_SIM_POOL_FILE")
    _idx=$(( ${USER_SIM_INDEX:-0} % ${#_eps[@]} ))
    USER_SIM_BASE_URL="${_eps[$_idx]}"
    [ "${#_eps[@]}" -eq 1 ] || echo "!! pool has ${#_eps[@]} entries and no USER_SIM_NAME --" \
      "picking by index $_idx. Set USER_SIM_NAME if they are not replicas." >&2
    echo "sim pool: replica $_idx of ${#_eps[@]} -> $USER_SIM_BASE_URL"
  fi
  USER_SIM_MODE=external
fi

if [ "$USER_SIM_MODE" = "external" ]; then
  : "${USER_SIM_BASE_URL:?set USER_SIM_BASE_URL, e.g. http://a100-046-241:8000/v1}"
  export OPENAI_BASE_URL="$USER_SIM_BASE_URL"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"   # vLLM ignores it but the SDK requires one
  export MULTITURN_MODEL_NAME="${MULTITURN_MODEL_NAME:-Qwen/Qwen3-32B}"
  HEALTH_URL="${OPENAI_BASE_URL%/v1}/health"
else
  export OPENAI_BASE_URL="http://127.0.0.1:${SHIM_PORT}/v1"
  export OPENAI_API_KEY="local-shim"        # non-empty; the gateway authorizes by identity
  export MULTITURN_MODEL_NAME="${MULTITURN_MODEL_NAME:-gpt-5.5}"
  # Pin the counterpart model at the shim as well. multi_turn.model_name below
  # covers the gyms verl configures, but grpo_multiturn.yaml still defaults to
  # gpt-4o and several gyms carry their own gpt-4o fallbacks -- and the gateway
  # 404s every OpenAI name except gpt-5.5, which the gyms would swallow silently.
  export SHIM_FORCE_MODEL="$MULTITURN_MODEL_NAME"
  # Reasoning effort for user-simulation and reward-judging calls. "none" keeps
  # latency inside the gyms' budget and stops thinking tokens from eating the
  # output budget; raise to "low" if reward signal looks noisy.
  export SHIM_REASONING_EFFORT="${SHIM_REASONING_EFFORT:-none}"
  export SHIM_MAX_CONCURRENCY="${SHIM_MAX_CONCURRENCY:-128}"
  export AI_GATEWAY_CALLING_PRODUCT=userrl
  # Requests the gateway could not serve get dumped for post-mortem:
  # gave_up-*.json (exhausted) and rescued_by_metagen-*.json (gateway failed,
  # fallback answered -- that call succeeds, so nothing else would record it).
  # Without this they land in the repo working directory.
  export API_DUMP_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/api_failures"
  HEALTH_URL="http://127.0.0.1:${SHIM_PORT}/health"
fi
echo "user simulator: mode=$USER_SIM_MODE model=$MULTITURN_MODEL_NAME url=$OPENAI_BASE_URL"

# ── The reward judge ──────────────────────────────────────────────────────────
# Each gym has ONE model field, used both to speak as the simulated user and to
# GRADE the agent for reward. In external mode that meant the local simulator
# computed the reward GRPO optimises against: runs 352024 and 351852 had a 4B
# model deciding every TravelGym search/preference judgment, every TurtleGym
# answer score, every IntentionGym coverage decision. Swapping the simulator
# therefore silently swapped the objective, not just the counterpart.
#
# JUDGE_MODEL_NAME pins the five gyms whose judge verifies against ground truth
# (Travel/Turtle/Telepathy/Intention/Search) -- their prompts carry the answer,
# so grader disagreement is error, not signal. PersuadeGym is left alone: its
# reward is the user's own self-reported stance, with no ground truth to check
# against. Tau (database-hash) and Function (no LLM) have no judge at all.
# Set JUDGE_MODEL_NAME="" to restore the old behaviour.
#
# gpt-5.5, NOT gpt-5.6-sol. Training needs ~672 gym calls/min sustained (measured
# on run 346885: 92,160 episodes, ~1.86M calls, 46h). gpt-5.6-sol is capped at
# max_rpm=125 / max_rp10m=600 -- i.e. 60/min sustained -- and that cap is real:
# a 160-request burst returned 24x HTTP 429 with "130/125 in 60s". Judging alone
# would take ~10 days against a 48h wall. gpt-5.5 has been driven to 6,121
# calls/min across four shims with ZERO rate-limit events, ~9x the headroom
# training needs, and it is what the 44-cell eval matrix used, so runs stay
# comparable. openai_shim.py paces gpt-5.6-sol automatically if you switch.
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME-gpt-5.5}"
JUDGE_MODEL_NAME=""
JUDGE_PORT="${JUDGE_PORT:-8078}"
JUDGE_SHIM_NEEDED=0
if [ -n "$JUDGE_MODEL_NAME" ]; then
  if [ "$USER_SIM_MODE" = "gateway" ] && [ "$JUDGE_MODEL_NAME" = "$MULTITURN_MODEL_NAME" ]; then
    # Same model, same route: reuse the simulator's shim rather than doubling
    # in-flight gateway calls against a per-user rate limit.
    JUDGE_BASE_URL="$OPENAI_BASE_URL"
  elif [ -n "${JUDGE_BASE_URL:-}" ]; then
    # Caller already runs a judge shim (sdpo_smoke.sbatch does, so that the
    # simulator can grade -- matching the baseline -- while gpt-5.5 writes SDPO
    # hints on a third endpoint). Starting another on the same port would
    # collide, and the loser's health check passes against the winner's server.
    echo "judge shim supplied by the caller: $JUDGE_BASE_URL"
  else
    JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
    JUDGE_SHIM_NEEDED=1
  fi
  export JUDGE_MODEL_NAME JUDGE_BASE_URL
  export JUDGE_API_KEY="${JUDGE_API_KEY:-local-shim}"
  echo "reward judge : $JUDGE_MODEL_NAME url=$JUDGE_BASE_URL$([ "$JUDGE_SHIM_NEEDED" = 1 ] && echo ' (own shim)' || echo ' (shared with the simulator)')"
else
  echo "reward judge : <none> -- the simulator grades itself (upstream behaviour)"
fi

# Overrides the gyms' 10s client timeout, which was sized for gpt-4o-mini and has
# no headroom for a reasoning model. Read by verl/tools/env_manager.py.
export USERRL_GYM_TIMEOUT="${USERRL_GYM_TIMEOUT:-180}"

# ── SDPO: L = L_RLVR + alpha * KL( pi(.|H) || pi(.|H+hint) ) ─────────────────
#
# Re-scores the agent's OWN response under a hindsight context. Nothing extra is
# generated and no training rows are added -- the teacher sequence is scored once
# under no_grad -- so L_RLVR is unchanged by construction. Cost is ~1.3x training
# compute (one extra forward) plus SDPO_TURNS_PER_EPISODE hint calls per
# trajectory.
#
# The hint is built from two privileged channels the agent did not have: the
# simulator's private reasoning behind the message it was replying to, and the
# simulator's reaction to its reply. gpt-5.5 filters those for ACHIEVABILITY --
# anything the agent could not re-derive from its own history is stripped, or the
# policy is trained to assert facts its context does not contain.
#
# Needs a THINKING simulator for the reasoning channel: gpt-5.5 over the Azure
# /responses transport does not expose reasoning, so keep it as the judge and run
# a Qwen simulator. Also needs the judge on its own shim -- env_manager.py
# refuses to start otherwise, because sharing one shim puts judge calls into the
# simulator's privileged buffer and silently attaches every hint to the wrong
# turn.
#
#   SDPO=1 sbatch run_rl.sh
SDPO="${SDPO:-0}"
sdpo_coef=0.0
# Activation offloading stays ON under SDPO. Its layer-group counter cannot
# survive a second forward per step, so dp_actor wraps the teacher forward in
# suspend_activation_offload() -- see the rationale in activation_offload.py.
# Disabling it outright was tried and 8B does not fit without it.
# Overridable: if the SDPO teacher forward is what degrades training (rather
# than the loss term), suspend_activation_offload() desyncing the counter the
# student's gradient-checkpoint recomputation uses is the prime suspect, and
# turning offload off entirely is the way to test it.
act_offload="${ACT_OFFLOAD:-True}"
if [ "$SDPO" = "1" ]; then
  sdpo_coef="${SDPO_COEF:-0.05}"
  export USERRL_SDPO=1
  export SDPO_TURNS_PER_EPISODE="${SDPO_TURNS_PER_EPISODE:-16}"
  export SDPO_HINT_MODEL="${SDPO_HINT_MODEL:-$JUDGE_MODEL_NAME}"
  # The hint writer is NOT the gyms' reward judge -- they were one variable
  # until the zero-reward bug, and the baseline (379160) self-judges, so
  # JUDGE_MODEL_NAME is deliberately empty here. Check what hints actually use.
  if [ -z "${SDPO_HINT_BASE_URL:-}" ] && [ -z "${JUDGE_BASE_URL:-}" ]; then
    echo "!! SDPO=1 needs SDPO_HINT_BASE_URL (an endpoint for the hint writer)" >&2
    exit 1
  fi
  echo "sdpo         : coef=$sdpo_coef estimator=${SDPO_ESTIMATOR:-ref} turns/ep=$SDPO_TURNS_PER_EPISODE hint=$SDPO_HINT_MODEL"
fi

export HF_HOME=$OUTPUT_DIR/hf_cache
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p "$OUTPUT_DIR/hf_cache" "$OUTPUT_DIR/wandb" "$OUTPUT_DIR/$EXPERIMENT_NAME" logs

# ── Node-local compile caches ─────────────────────────────────────────────────
# The default caches live on shared FSx; with many ranks racing on the same files
# reads hit "OSError: [Errno 116] Stale file handle". Point them at node-local
# /tmp, namespaced by job id. Same fix OdysSim-eval's run_rl.sh applies.
_cache_tag="${SLURM_JOB_ID:-$$}"

# Ray's temp dir, per job and on SHARED storage. Two reasons.
#
# 1. The default (/var/tmp/ray) is shared between whatever jobs land on a node.
#    It holds ray_current_cluster, pointing at the last head that ran there --
#    owned by that user, so ours can neither read nor delete it. Job 388771 died
#    on "PermissionError: /var/tmp/ray/ray_current_cluster"; others hit a
#    ConnectionError to a dead head. Node-dependent, so it presents as random.
#
# 2. Ray writes each worker's stderr to {temp}/session_latest/logs/worker-*.err,
#    and that is the ONLY place a worker's traceback lands -- the raylet prints
#    just "A worker died ... unexpected system error". On node-local /tmp those
#    logs die with the allocation, which is why five 2-node jobs failed at
#    init_model() with no diagnosable cause. On FSx they survive the job.
# SHORT and node-local. It must be short because Ray puts a Unix socket at
# {RAY_TMPDIR}/session_<stamp>/sockets/plasma_store and AF_UNIX caps the whole
# path at 107 bytes -- pointing this at $OUTPUT_DIR blew that limit and killed
# job 391800 outright. Node-local is fine for the socket; the worker logs are
# what we actually need to keep, so copy those out at exit instead.
export RAY_TMPDIR="/tmp/ray-$_cache_tag"
mkdir -p "$RAY_TMPDIR"
_ray_logs="$OUTPUT_DIR/${EXPERIMENT_NAME}/ray-logs"
_save_ray_logs() {
  mkdir -p "$_ray_logs" 2>/dev/null || return 0
  cp "$RAY_TMPDIR"/session_*/logs/worker-*.err "$_ray_logs/" 2>/dev/null || true
  cp "$RAY_TMPDIR"/session_*/logs/raylet.* "$_ray_logs/" 2>/dev/null || true
}
# NOTE: the shim sections below install their own `trap ... EXIT`, which
# REPLACES rather than extends. Both cleanup() bodies therefore call
# _save_ray_logs themselves; this bare trap only covers an exit before them.
trap _save_ray_logs EXIT

export TRITON_CACHE_DIR="/tmp/triton-cache-$_cache_tag"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-cache-$_cache_tag"
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
# NO expandable_segments here, unlike OdysSim-eval's run_rl.sh. That repo runs a
# vLLM rollout; this one runs sglang, and verl asks sglang for
# enable_memory_saver=True (sglang_rollout_customized.py:362) so the hybrid engine
# can release KV cache during the training phase. torch_memory_saver refuses to
# initialise when the caching allocator uses expandable segments:
#   RuntimeError: TorchMemorySaver is disabled for the current process because
#   expandable_segments is not supported yet.
# Every scheduler process dies at model load. The memory saver is not optional
# (that True is hardcoded), so the allocator setting is what has to give.
# Unset rather than merely omitted, in case the submitting shell exports it.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# ── Resolve x2pagentd ─────────────────────────────────────────────────────────
# Needed whenever anything on this node talks to the AI Gateway: the simulator
# shim in gateway mode, the judge shim in either mode. External mode used to skip
# this entirely, which is fine while the counterpart is a local vLLM but not once
# a gateway judge is in the loop.
#
# The logic lives in x2p_resolve.sh because it was duplicated here and omitted
# from the pool script, and that omission published gateway endpoints that
# answered /health but could not complete. See that file's header.
if [ "$USER_SIM_MODE" = "gateway" ] || [ "$JUDGE_SHIM_NEEDED" = 1 ]; then
  . "$PROJECT_DIR/x2p_resolve.sh"
  _x2p_resolve || {
    echo "!! gateway unreachable. This job needs --export=ALL,WITH_X2P=1 (it is in" >&2
    echo "   this script's SBATCH header)." >&2; exit 1; }
  # Multi-node caveat: if the winner is a localhost:PORT form, worker nodes only
  # inherit a correct value when their agent happens to use the same port. Verify
  # per-node shim logs before trusting a >1 node gateway run.
fi

# ── Start the API shim (gateway mode only, one per node) ──────────────────────
if [ "$USER_SIM_MODE" = "gateway" ]; then
  SHIM_LOG="$OUTPUT_DIR/$EXPERIMENT_NAME/shim-$(hostname).log"
  echo "Starting openai_shim on $(hostname):$SHIM_PORT -> $SHIM_LOG"
  python3 "$PROJECT_DIR/openai_shim.py" --port "$SHIM_PORT" >"$SHIM_LOG" 2>&1 &
  SHIM_PID=$!
  # The rollout workers on every other node also call 127.0.0.1, so each needs
  # its own copy -- and each must resolve its OWN x2pagentd.
  #
  # This is what killed job 374195 after 4h07m. x2p_resolve.sh runs on the HEAD
  # and exports X2P_PROXY_URL=http://localhost:<head's port>. srun propagates the
  # environment, api.py PREFERS X2P_PROXY_URL over localhost:$X2P_PORT, and the
  # worker's agent listens on a different port -- so every gateway call from the
  # worker timed out. 112 TravelGym timeouts, all from the worker's IP, then
  # `sorted_output_req_list` came back None and the run died. Unset the inherited
  # value and re-resolve per node.
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$NNODES" -gt 1 ]; then
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2); do
      echo "Starting openai_shim on $node:$SHIM_PORT"
      srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -c "
          unset X2P_PROXY_URL X2P_PORT   # both are the HEAD's; this node has its own
          . '$PROJECT_DIR/x2p_resolve.sh'
          _x2p_resolve || { echo '!! no x2pagentd on '\$(hostname) >&2; exit 1; }
          exec python3 '$PROJECT_DIR/openai_shim.py' --port '$SHIM_PORT'" \
        >"$OUTPUT_DIR/$EXPERIMENT_NAME/shim-$node.log" 2>&1 &
    done
  fi
  # Kill the shim however the job ends, including scancel.
  cleanup() { _save_ray_logs; kill "$SHIM_PID" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
fi

# ── Start the judge shim (one per node, same reason as above) ─────────────────
# SHIM_FORCE_MODEL is process-global, so a judge on a different model than the
# simulator needs its own process. In external mode this is the ONLY gateway
# client in the job.
if [ "$JUDGE_SHIM_NEEDED" = 1 ]; then
  JUDGE_LOG="$OUTPUT_DIR/$EXPERIMENT_NAME/judge-$(hostname).log"
  echo "Starting judge shim ($JUDGE_MODEL_NAME) on $(hostname):$JUDGE_PORT -> $JUDGE_LOG"
  SHIM_FORCE_MODEL="$JUDGE_MODEL_NAME" \
  SHIM_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-none}" \
  SHIM_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-128}" \
  AI_GATEWAY_CALLING_PRODUCT=userrl \
    python3 "$PROJECT_DIR/openai_shim.py" --port "$JUDGE_PORT" >"$JUDGE_LOG" 2>&1 &
  JUDGE_PID=$!
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$NNODES" -gt 1 ]; then
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2); do
      echo "Starting judge shim on $node:$JUDGE_PORT"
      # Same per-node x2p resolution as the simulator shim above -- see the
      # comment there for what inheriting the head's X2P_PROXY_URL cost us.
      srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -c "
          unset X2P_PROXY_URL X2P_PORT   # both are the HEAD's; this node has its own
          . '$PROJECT_DIR/x2p_resolve.sh'
          _x2p_resolve || { echo '!! no x2pagentd on '\$(hostname) >&2; exit 1; }
          export SHIM_FORCE_MODEL='$JUDGE_MODEL_NAME'
          export SHIM_REASONING_EFFORT='${JUDGE_REASONING_EFFORT:-none}'
          export SHIM_MAX_CONCURRENCY='${JUDGE_MAX_CONCURRENCY:-128}'
          export AI_GATEWAY_CALLING_PRODUCT=userrl
          exec python3 '$PROJECT_DIR/openai_shim.py' --port '$JUDGE_PORT'" \
        >"$OUTPUT_DIR/$EXPERIMENT_NAME/judge-$node.log" 2>&1 &
    done
  fi
  # Extend, do not replace, whatever trap the simulator shim installed.
  cleanup() { _save_ray_logs; kill "${SHIM_PID:-}" "${JUDGE_PID:-}" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
fi

# Refuse to start training against a dead simulator -- that failure mode is
# invisible once training begins, because the gyms fall back to canned text.
# In external mode this also catches "the vLLM job has not finished loading yet",
# which is the normal case when both jobs are submitted together.
echo "waiting for the simulator at $HEALTH_URL"
for i in $(seq 1 "${SIM_WAIT_TRIES:-60}"); do
  if curl -sf --max-time 10 "$HEALTH_URL" >/dev/null; then
    echo "simulator healthy: $(curl -s --max-time 10 "$HEALTH_URL" | head -c 300)"
    break
  fi
  if [ "$i" -eq "${SIM_WAIT_TRIES:-60}" ]; then
    echo "!! simulator never became healthy at $HEALTH_URL" >&2
    [ "$USER_SIM_MODE" = "gateway" ] && tail -50 "$SHIM_LOG" >&2
    [ "$USER_SIM_MODE" = "external" ] && echo "   is serve_user_sim.sh still loading? raise SIM_WAIT_TRIES." >&2
    exit 1
  fi
  sleep "${SIM_WAIT_SECONDS:-10}"
done

# Same guard for the judge, and stricter: a dead simulator produces canned user
# turns, but a dead judge produces a WRONG REWARD -- GRPO would spend days
# optimising against zeros with nothing in the logs to say so. /health only
# proves the shim process is alive, so make a real completion before training.
if [ -n "${JUDGE_MODEL_NAME:-}" ]; then
  echo "waiting for the judge at ${JUDGE_BASE_URL%/v1}/health"
  for i in $(seq 1 "${JUDGE_WAIT_TRIES:-30}"); do
    curl -sf --max-time 10 "${JUDGE_BASE_URL%/v1}/health" >/dev/null && break
    if [ "$i" -eq "${JUDGE_WAIT_TRIES:-30}" ]; then
      echo "!! judge never became healthy at ${JUDGE_BASE_URL%/v1}/health" >&2
      [ -n "${JUDGE_LOG:-}" ] && tail -50 "$JUDGE_LOG" >&2
      exit 1
    fi
    sleep "${SIM_WAIT_SECONDS:-10}"
  done
  if curl -sf --max-time 180 "$JUDGE_BASE_URL/chat/completions" \
       -H 'Content-Type: application/json' \
       -d '{"model":"probe","messages":[{"role":"user","content":"say OK"}],"max_tokens":16}' \
       2>/dev/null | grep -q '"content"'; then
    echo "judge healthy: $JUDGE_MODEL_NAME answered a live probe"
  else
    echo "!! judge $JUDGE_MODEL_NAME is up but cannot complete -- refusing to train" >&2
    [ -n "${JUDGE_LOG:-}" ] && tail -50 "$JUDGE_LOG" >&2
    exit 1
  fi
fi

# ── Every WORKER node's shims must complete too, not just the head's ──────────
# The checks above only probe 127.0.0.1 on the head. Job 374195 passed them and
# then died 4h07m later: the worker's shim was up and answering /health, but
# could not reach the gateway, so all 112 of its gym calls timed out. A shim that
# is alive but cannot complete is precisely the silent failure the gyms convert
# into canned text -- so make one real completion PER NODE before training.
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$NNODES" -gt 1 ]; then
  # Which ports actually have a shim on the workers. Built explicitly: a
  # ${VAR:+...} test on JUDGE_SHIM_NEEDED expands on the string "0" too, which
  # would probe a port nothing is listening on and abort a healthy run.
  _ports=""
  [ "$USER_SIM_MODE" = "gateway" ] && _ports="$SHIM_PORT"
  [ "$JUDGE_SHIM_NEEDED" = 1 ] && _ports="$_ports $JUDGE_PORT"
  _probe='{"model":"probe","messages":[{"role":"user","content":"say OK"}],"max_tokens":16}'
  for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2); do
    for _p in $_ports; do
      echo "verifying shim on $node:$_p"
      if srun --overlap --nodes=1 --ntasks=1 -w "$node" \
           curl -sf --max-time 180 "http://127.0.0.1:$_p/v1/chat/completions" \
             -H 'Content-Type: application/json' -d "$_probe" 2>/dev/null \
           | grep -q '"content"'; then
        echo "  $node:$_p ok"
      else
        echo "!! shim on $node:$_p is unreachable or cannot complete -- refusing to train" >&2
        echo "   This is the 374195 failure mode: the head is fine, a worker is not," >&2
        echo "   and the gyms would silently substitute canned text for that node." >&2
        tail -30 "$OUTPUT_DIR/$EXPERIMENT_NAME"/*-"$node".log >&2 2>/dev/null || true
        exit 1
      fi
    done
  done
fi

# ── Ray cluster (multi-node only) ─────────────────────────────────────────────
# main_ppo calls ray.init() with no address, so it otherwise builds a node-local
# cluster and ignores the rest of the allocation.
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$NNODES" -gt 1 ]; then
  ray_num_cpus="${SLURM_CPUS_PER_TASK:-96}"
  nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
  head_node=${nodes_array[0]}
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    head_node_ip=$([[ ${#ADDR[0]} -gt 16 ]] && echo "${ADDR[1]}" || echo "${ADDR[0]}")
  fi
  port=6379
  export RAY_ADDRESS="$head_node_ip:$port"
  export MASTER_ADDR=$head_node_ip
  export MASTER_PORT=$port
  echo "Starting Ray HEAD at $head_node ($RAY_ADDRESS)"
  srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --temp-dir "$RAY_TMPDIR" \
    --num-cpus "$ray_num_cpus" --num-gpus "$GPUS_PER_NODE" --block &

  # Wait for the head to actually accept connections instead of sleeping a
  # guess. A worker that runs `ray start --address` too early fails, and it
  # fails INVISIBLY: the srun is backgrounded with &, so its exit status is
  # never checked and its output goes nowhere.
  for _i in $(seq 1 60); do
    ray status --address "$RAY_ADDRESS" >/dev/null 2>&1 && break
    [ "$_i" -eq 60 ] && { echo "!! Ray head never came up at $RAY_ADDRESS" >&2; exit 1; }
    sleep 2
  done
  echo "Ray head is up"

  for node in "${nodes_array[@]:1}"; do
    echo "Starting Ray WORKER at $node"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
      ray start --address "$RAY_ADDRESS" \
      --num-cpus "$ray_num_cpus" --num-gpus "$GPUS_PER_NODE" --block &
  done

  # REFUSE to train unless every GPU actually joined.
  #
  # This is the fix for the 2-node failures (389835/389836/389874/389991/390041,
  # five of six). If a worker's ray start loses the race above, the cluster has
  # 8 GPUs while the trainer asks for nnodes x 8 = 16. Ray then over-places the
  # WorkerDicts, two land on one GPU, and they die during init_model() with a
  # CUDA-level fault that carries NO Python traceback -- the raylet reports only
  # "A worker died ... unexpected system error". Flaky because it is a race, and
  # invisible because the old code ended in `ray status || true`, discarding the
  # single check that would have caught it.
  _want_gpu=$(( NNODES * GPUS_PER_NODE ))
  for _i in $(seq 1 90); do
    _have=$(ray status --address "$RAY_ADDRESS" 2>/dev/null \
            | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 | sed 's|.*/||; s|\.0 GPU||')
    [ "${_have:-0}" -ge "$_want_gpu" ] 2>/dev/null && break
    if [ "$_i" -eq 90 ]; then
      echo "!! Ray cluster has ${_have:-0} GPUs, expected $_want_gpu -- a worker" >&2
      echo "   node failed to join. Training now would over-place WorkerDicts and" >&2
      echo "   die in init_model() with no traceback. Refusing." >&2
      ray status --address "$RAY_ADDRESS" >&2 2>/dev/null || true
      exit 1
    fi
    sleep 2
  done
  echo "Ray cluster ready: $_have/$_want_gpu GPUs across $NNODES nodes"
  ray status --address "$RAY_ADDRESS" || true
fi

# ── Train ─────────────────────────────────────────────────────────────────────
_dyn=()
if [ "$dynamic_bsz" = "1" ]; then
  _dyn=(actor_rollout_ref.actor.use_dynamic_bsz=True
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu")
  echo "dynamic bsz: on, ${max_token_len_per_gpu} tokens/gpu (micro_bsz ignored)"
fi
set -x
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_multiturn' \
    algorithm.adv_estimator=grpo_multiturn \
    algorithm.gamma=$gamma \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bsz \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.sdpo_coef=$sdpo_coef \
    actor_rollout_ref.actor.sdpo_estimator="${SDPO_ESTIMATOR:-ref}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.model.enable_activation_offload=$act_offload \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bsz \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem_util \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.multi_turn.max_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.model_name="$MULTITURN_MODEL_NAME" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/interact_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="$turn_level_method" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="$trajectory_score_method" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='UserRL' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    ${RESUME_PATH:+trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME_PATH} \
    trainer.default_local_dir="$OUTPUT_DIR/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$save_freq \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null} \
    trainer.test_freq=$test_freq \
    trainer.val_before_train=False \
    trainer.total_epochs=$total_epochs "${_dyn[@]}" "$@"
