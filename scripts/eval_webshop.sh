#!/bin/bash
set -x

# ── Evaluation script: Webshop with trained model ────────────────────────────
# Runs validation-only (total_epochs=0, val_before_train=True) on Webshop.
#
# PREREQUISITE: Webshop data must be pre-downloaded to shared storage.
# On a machine with internet/Google Drive access, run:
#   cd agent_system/environments/webshop/webshop && ./setup.sh -d small
# Then copy/move the resulting data/ directory to:
#   /shared/public/sharing/sirzhu/webshop_data/
# This script will symlink it into the expected location.
#
# Usage:
#   WEBSHOP_DATA_SRC=/shared/public/sharing/sirzhu/webshop_data bash eval_webshop.sh

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

# ── Java (required by pyserini's LuceneSearcher) ──────────────────────────────
# Prefer LinkedIn-cluster Java; fall back to yum-installed java-11-openjdk in image
if [ -d "/export/apps/jdk/JDK-11_0_13-msft" ]; then
    export JAVA_HOME=/export/apps/jdk/JDK-11_0_13-msft
else
    export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
fi
export PATH="${JAVA_HOME}/bin:${PATH}"

# ── Webshop symlinks ──────────────────────────────────────────────────────────
# Flask server expects data/, search_engine/, and web_agent_site/templates/
# inside the webshop package dir. Repo ships empty stubs; link real files from NFS.
WEBSHOP_DIR="${LAMER_ROOT}/agent_system/environments/webshop/webshop"
# NFS source: verl-agent pre-built webshop environment
WEBSHOP_NFS_ROOT=${WEBSHOP_NFS_ROOT:-/shared/public/sharing/sirzhu/verl-agent/agent_system/environments/env_package/webshop/webshop}
WEBSHOP_DATA_SRC=${WEBSHOP_DATA_SRC:-${WEBSHOP_NFS_ROOT}/data}
WEBSHOP_DATA_DST="${WEBSHOP_DIR}/data"
WEBSHOP_SE_SRC="${WEBSHOP_NFS_ROOT}/search_engine"
WEBSHOP_SE_DST="${WEBSHOP_DIR}/search_engine"
WEBSHOP_TMPL_SRC="${WEBSHOP_NFS_ROOT}/web_agent_site/templates"
WEBSHOP_TMPL_DST="${WEBSHOP_DIR}/web_agent_site/templates"

# data/ and search_engine/ are required (hard fail if missing); templates/ is optional
# (repo now ships real templates; NFS version overrides if available).
for pair in "${WEBSHOP_DATA_SRC}:${WEBSHOP_DATA_DST}:required" "${WEBSHOP_SE_SRC}:${WEBSHOP_SE_DST}:required" "${WEBSHOP_TMPL_SRC}:${WEBSHOP_TMPL_DST}:optional"; do
    src="${pair%%:*}"; rest="${pair#*:}"; dst="${rest%%:*}"; req="${rest##*:}"
    if [ -d "${src}" ]; then
        rm -rf "${dst}"
        ln -sf "${src}" "${dst}"
        echo "Symlinked: ${src} -> ${dst}"
    elif [ "${req}" = "required" ]; then
        echo "ERROR: Webshop source not found at ${src}"
        exit 1
    else
        echo "INFO: Optional NFS source not found at ${src}, using repo version"
    fi
done

# ── Model checkpoint ──────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH}
echo "MODEL_PATH: $MODEL_PATH"

# ── GPU config ────────────────────────────────────────────────────────────────
GPUS=${GPUS:-$(nvidia-smi --list-gpus | wc -l)}
echo "GPUS: $GPUS"
TP_SIZE=1

# ── Eval data: create minimal dummy parquets if not present ──────────────────
REFLECTION_TYPE=${REFLECTION_TYPE:-reflection_only}
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}   # parallel workers (memory constraint: each loads full catalog)
VAL_TOTAL_SIZE=${VAL_TOTAL_SIZE:-100}  # total evaluations; must be a multiple of VAL_BATCH_SIZE
DATA_DIR="${HOME}/data/verl-agent/text"
mkdir -p "${DATA_DIR}"
python3 - <<EOF
import os, pandas as pd
data_dir = os.path.expanduser("~/data/verl-agent/text")
os.makedirs(data_dir, exist_ok=True)
def make_rows(split, n):
    return [{"data_source": "text",
             "prompt": [{"role": "user", "content": ""}],
             "ability": "agent",
             "extra_info": {"split": split, "index": i}} for i in range(n)]
pd.DataFrame(make_rows("train", ${TRAIN_BATCH_SIZE})).to_parquet(os.path.join(data_dir, "train.parquet"))
pd.DataFrame(make_rows("test", ${VAL_TOTAL_SIZE})).to_parquet(os.path.join(data_dir, "test.parquet"))
print("Created dummy scaffold data: train=${TRAIN_BATCH_SIZE} test=${VAL_TOTAL_SIZE} at", data_dir)
EOF

# ── Experiment name ───────────────────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-unknown}
EXPERIMENT_NAME="lamer-eval-webshop-${MODEL_NAME}-${REFLECTION_TYPE}-${FLYTE_INTERNAL_EXECUTION_ID:-local}"

# ── Run evaluation (val_before_train=True, total_epochs=0 → validation only) ─
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=16384 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    env.env_name=Webshop \
    env.webshop.use_small=False \
    env.webshop.human_goals=True \
    +env.reflection_type="${REFLECTION_TYPE}" \
    env.seed=0 \
    env.rollout.n=1 \
    env.num_attempts=3 \
    env.max_steps=30 \
    env.max_turns=10 \
    reward_model.reward_manager=episode \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name="${MLFLOW_EXPERIMENT_NAME:-lamer-eval}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.total_epochs=0 \
    trainer.log_val_generations=${VAL_TOTAL_SIZE} \
    trainer.default_local_dir=/shared/public/sharing/sirzhu/eval_webshop/${EXPERIMENT_NAME}

