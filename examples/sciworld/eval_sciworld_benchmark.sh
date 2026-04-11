#!/bin/bash
set -x

# ============================================================
# SciWorld Benchmark: evaluate a model checkpoint on 6 SciWorld tasks
#
# Tasks (matching explorer's eval_sciworld_multi.yaml):
#   Classification: find-animal (75), find-living-thing (75), find-plant (75)
#   Electricity:    power-component (5), renewable-vs-nonrenewable (5)
#   Biology:        identify-life-stages-2 (4)
#   Total: 239 test variations (padded to nearest GPU multiple)
#
# Each run evaluates twice:
#   1) Without reflection (ReAct baseline)
#   2) With reflection (LaMer/Reflexion style)
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint bash examples/sciworld/eval_sciworld_benchmark.sh
#
# Optional env vars:
#   ENGINE           — vllm engine name (default: vllm)
#   EXPERIMENT_TAG   — suffix for experiment names (default: "benchmark")
#   CONDA_ENV        — conda environment name (default: verl-test)
#   N_GPUS           — number of GPUs per node (default: 2)
#   VAL_BATCH_SIZE   — validation batch size (default: 239, all tasks)
# ============================================================

# ---- Environment setup ----
if [ -n "$CONDA_ENV" ]; then
    source /share/pkg.7/miniconda/23.1.0/install/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "$CONDA_ENV"
fi

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

cd "$(dirname "$0")/../.." || exit 1  # cd to LaMer root

# ---- Required: model checkpoint ----
if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is not set."
    echo "Usage: MODEL_PATH=/path/to/checkpoint bash examples/sciworld/eval_sciworld_benchmark.sh"
    exit 1
fi

# ---- Configuration ----
ENGINE=${ENGINE:-vllm}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-benchmark}
N_GPUS=${N_GPUS:-2}
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

# Compute padded val batch size (239 real tasks → nearest multiple of N_GPUS)
REAL_VAL_COUNT=239
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-$(python3 -c "import math; print(math.ceil($REAL_VAL_COUNT / $N_GPUS) * $N_GPUS)")}

SCIWORLD_CONFIG=examples/sciworld/multi_task_config.yaml
DATA_DIR=$HOME/data/sciworld-eval-${EXPERIMENT_TAG}

# ---- Step 1: Prepare evaluation data (Orbit-identical seeds + GPU padding) ----
echo "=========================================="
echo "Preparing SciWorld evaluation data..."
echo "=========================================="
python3 -m examples.gem.prepare_gem_data \
    --config "$SCIWORLD_CONFIG" \
    --seed 42 \
    --pad_to_multiple "$N_GPUS" \
    --output_dir "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed."
    exit 1
fi

# ---- Common parameters matching explorer's evaluation settings ----
# Key settings from explorer:
#   - temperature=0.6, top_p=0.95, do_sample=True (explorer defaults)
#   - val_kwargs.n=1 (single attempt per sample, as in explorer benchmark)
#   - total_epochs=0 + val_before_train=True + val_only=True (eval only)
#   - prompt/response lengths follow LaMer's settings (4096/1024), NOT explorer's
#     (explorer uses MAX_CTX-1024 response), because LaMer's prompt system is
#     inherently different (structured game rules + past trajectories + reflection).
COMMON_PARAMS=(
    # Eval-only mode
    trainer.total_epochs=0
    trainer.val_before_train=True
    +trainer.val_only=True

    # Data — prompt/response lengths follow LaMer's original settings
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/test.parquet"
    data.train_batch_size=4
    data.val_batch_size=$VAL_BATCH_SIZE
    data.max_prompt_length=4096
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=left
    data.return_raw_chat=True

    # Model
    actor_rollout_ref.model.path="$MODEL_PATH"
    +actor_rollout_ref.model.enable_thinking=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

    # Rollout / generation — matching explorer's eval settings
    actor_rollout_ref.rollout.name=$ENGINE
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.max_num_batched_tokens=32768

    # Validation generation parameters (matching explorer: temp=0.6, top_p=0.95)
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    +actor_rollout_ref.rollout.val_kwargs.seed=20
    actor_rollout_ref.rollout.val_kwargs.n=1

    # Ref policy
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32

    # Algorithm (needed for init but not used in eval)
    algorithm.adv_estimator=gigpo
    algorithm.use_kl_in_reward=False
    +algorithm.step_gamma=0.95
    +algorithm.traj_gamma=0.6

    # Reward
    reward_model.reward_manager=episode

    # Environment
    env.env_name=sciworld
    env.seed=0
    env.max_steps=15
    env.max_turns=15

    # Trainer
    trainer.critic_warmup=0
    trainer.n_gpus_per_node=$N_GPUS
    trainer.nnodes=1
    trainer.log_val_generations=1
)

# Append any extra overrides passed as positional args
COMMON_PARAMS+=("$@")


# ======================== Run 1: Without Reflection ========================
echo ""
echo "=========================================="
echo "[1/2] SciWorld — no reflection (ReAct)"
echo "=========================================="
python3 -m verl.trainer.main_ppo \
    "${COMMON_PARAMS[@]}" \
    env.rollout.n=1 \
    env.num_attempts=3 \
    +env.do_reflection=False \
    +env.reflection_type=reflection_only \
    trainer.logger=['console','wandb'] \
    trainer.project_name=sciworld-benchmark \
    trainer.experiment_name="sciworld-${MODEL_NAME}-${EXPERIMENT_TAG}-no-reflection"

echo "SciWorld (no reflection) completed with exit code: $?"


# ======================== Run 2: With Reflection ========================
echo ""
echo "=========================================="
echo "[2/2] SciWorld — with reflection (LaMer)"
echo "=========================================="
python3 -m verl.trainer.main_ppo \
    "${COMMON_PARAMS[@]}" \
    env.rollout.n=1 \
    env.num_attempts=3 \
    +env.do_reflection=True \
    +env.reflection_type=reflection_only \
    trainer.logger=['console','wandb'] \
    trainer.project_name=sciworld-benchmark \
    trainer.experiment_name="sciworld-${MODEL_NAME}-${EXPERIMENT_TAG}-reflection"

echo "SciWorld (reflection) completed with exit code: $?"


# ======================== Summary ========================
echo ""
echo "=========================================="
echo "SciWorld benchmark complete."
echo "  Model:      $MODEL_PATH"
echo "  Tasks:      6 (239 real + padding to $VAL_BATCH_SIZE)"
echo "  Runs:       2 (no-reflection + reflection)"
echo "  Logging:    wandb project=sciworld-benchmark"
echo "=========================================="
