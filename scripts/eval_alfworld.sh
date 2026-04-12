#!/bin/bash
set -x

# ── Evaluation script: ALFWorld with trained model ───────────────────────────
# Runs validation-only (total_epochs=0, val_before_train=True) on ALFWorld
# using the specified trained model checkpoint.
#
# Usage (in pod or locally):
#   bash eval_alfworld.sh
# Optional env overrides:
#   MODEL_PATH=<path>  GPUS=<n>  VAL_BATCH_SIZE=<n>

# ── MLflow setup (Flyte-only: no-op outside cluster) ─────────────────────────
if [ -n "$FLYTE_INTERNAL_EXECUTION_PROJECT" ]; then
    export MLFLOW_TRACKING_URI='http://mlflow-service.mlflow:5000'
    WORKSPACE_NAME=$(echo "$FLYTE_INTERNAL_EXECUTION_WORKFLOW" | awk -F'.' '{print $(NF)}')
    export MLFLOW_EXPERIMENT_NAME="${FLYTE_INTERNAL_EXECUTION_PROJECT}/${WORKSPACE_NAME}"
fi

# ── Activate lamer venv ───────────────────────────────────────────────────────
source /home/jobuser/.venv/lamer/bin/activate


# ── LAMER_ROOT: where the repo was uploaded by openconnect ───────────────────
LAMER_ROOT=/home/jobuser/resources
export PYTHONPATH="${LAMER_ROOT}:${PYTHONPATH}"
cd "${LAMER_ROOT}"

# ── ALFWorld data: must be non-empty (baked into Docker via alfworld-download -f) ─
export ALFWORLD_DATA=${ALFWORLD_DATA:-/home/jobuser/.cache/alfworld}
echo "ALFWORLD_DATA: $ALFWORLD_DATA"
ls "${ALFWORLD_DATA}/json_2.1.1" 2>/dev/null || echo "[WARN] ALFWORLD_DATA json_2.1.1 not found at ${ALFWORLD_DATA}"
# ── Model checkpoint ──────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH}

# ── GPU config ────────────────────────────────────────────────────────────────
GPUS=${GPUS:-$(nvidia-smi --list-gpus | wc -l)}
TP_SIZE=1

# ── Eval data: create minimal dummy parquets if not present ──────────────────
# The env provides real observations; geometry3k data is only used as scaffolding
# for verl's data loader (the actual prompts come from ALFWorld env).
TRAIN_DATA_SIZE=8
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-134}
DATA_DIR="${HOME}/data/verl-agent/text"
mkdir -p "${DATA_DIR}"
python3 - <<EOF
import os, pandas as pd
data_dir = os.path.expandvars("${HOME}/data/verl-agent/text")
os.makedirs(data_dir, exist_ok=True)
def make_rows(split, n):
    return [{"data_source": "text",
             "prompt": [{"role": "user", "content": ""}],
             "ability": "agent",
             "extra_info": {"split": split, "index": i}} for i in range(n)]
pd.DataFrame(make_rows("train", ${TRAIN_DATA_SIZE})).to_parquet(os.path.join(data_dir, "train.parquet"))
pd.DataFrame(make_rows("test", ${VAL_BATCH_SIZE})).to_parquet(os.path.join(data_dir, "test.parquet"))
print(f"Created dummy scaffold data: train={${TRAIN_DATA_SIZE}}, test={${VAL_BATCH_SIZE}}")
EOF

# ── Reflection type: reflection_only | history_only | history_and_reflection ──
REFLECTION_TYPE=${REFLECTION_TYPE:-reflection_only}

echo "=========================================="
echo "  eval_alfworld.sh args from workflow"
echo "=========================================="
echo "  MODEL_PATH:       ${MODEL_PATH}"
echo "  GPUS:             ${GPUS}"
echo "  VAL_BATCH_SIZE:   ${VAL_BATCH_SIZE}"
echo "  REFLECTION_TYPE:  ${REFLECTION_TYPE}"
echo "  ALFWORLD_DATA:    ${ALFWORLD_DATA}"
echo "=========================================="

# ── Experiment name ───────────────────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-unknown}
EXPERIMENT_NAME="lamer-eval-alfworld-${MODEL_NAME}-${REFLECTION_TYPE}-${FLYTE_INTERNAL_EXECUTION_ID:-local}"

# ── Run evaluation (val_before_train=True, total_epochs=0 → validation only) ─
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${TRAIN_DATA_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=7168 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    env.env_name="alfworld/AlfredTWEnv" \
    env.alfworld.eval_dataset=eval_out_of_distribution \
    env.seed=0 \
    env.rollout.n=1 \
    env.num_attempts=3 \
    env.max_steps=30 \
    env.max_turns=10 \
    +env.reflection_type="${REFLECTION_TYPE}" \
    reward_model.reward_manager=episode \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name="${MLFLOW_EXPERIMENT_NAME:-lamer-eval}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.total_epochs=0 \
    trainer.log_val_generations=1 \
    trainer.default_local_dir=/shared/public/sharing/sirzhu/eval_alfworld/${EXPERIMENT_NAME}

