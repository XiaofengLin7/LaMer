set -x
ENGINE=${ENGINE:-${1:-vllm}}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
GPUS_PER_NODE=$(nvidia-smi --list-gpus|wc -l)
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
source /home/jobuser/resources/scripts/setup_mlflow_hf.sh

# ── Activate lamer venv and set repo root ────────────────────────────────────
source /home/jobuser/.venv/lamer/bin/activate
LAMER_ROOT=/home/jobuser/resources
export PYTHONPATH="${LAMER_ROOT}:${PYTHONPATH}"
cd "${LAMER_ROOT}"

# ── NLTK words corpus ─────────────────────────────────────────────────────────
# Corpus is pre-installed in the Docker image (NLTK_DATA=/home/jobuser/nltk_data).
# Pre-load in main thread to avoid Ray worker race condition on nltk.corpus.words.
export NLTK_DATA=/home/jobuser/nltk_data
python -c "
import nltk
nltk.corpus.words.words('en-basic')
print('NLTK words corpus pre-loaded')
"

# Per-task train_size=512, 5 tasks => 2560 total train tasks
# Per-task test_size=256, 2 val tasks => 512 total val tasks
# train_batch_size = number of unique tasks per batch (before group repeat)
train_batch_size=64
val_batch_size=128
group_size=4
reflection_type="${REFLECTION_TYPE:-history_and_reflection}"

# Multi-episode is handled inside GEM wrappers.
# LaMer adds reflection on top: num_attempts=3 with reflection between attempts.
# max_turns = max(total_step_cap) across all tasks = 30
GEM_CONFIG=examples/gem/multi_task_multi_episode_config.yaml
DATA_DIR=$HOME/data/gem-multi-task
DISABLE_THINKING=False
ENABLE_REFLECTION=True
# MODEL_NAME=Qwen3-8B
# MODEL_PATH="/shared/public/elr-models/Qwen/Qwen3-8B/2069b3fae1114555f3c020c81410e51fa0f656f2_130k_context"
MODEL_NAME=GLM-4-9B-0414
# MODEL_PATH="/shared/public/models/zai-org-GLM-4-9B-0414"
MODEL_PATH="/shared/public/sharing/sirzhu/Lamer/gem-multi-task-multi-task-multi-episode-config-GLM-4-9B-0414-disable-thinking-False-enable-reflection-True-reflection-type-reflection_only-ff8496b108549420db5f/global_step_100/actor_hf"
CONFIG_NAME=$(basename "$GEM_CONFIG" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
EXPERIMENT_NAME="gem-multi-task-${CONFIG_NAME}-${MODEL_NAME}-disable-thinking-${DISABLE_THINKING}-enable-reflection-${ENABLE_REFLECTION}-reflection-type-${reflection_type}-${FLYTE_INTERNAL_EXECUTION_ID}"

# Step 1: Prepare data with Orbit-identical seeds
python3 -m examples.gem.prepare_gem_data \
    --config $GEM_CONFIG \
    --seed 42 \
    --output_dir $DATA_DIR

# Step 2: Train
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=7168 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.enable_thinking=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=9216 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.seed=20 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    reward_model.reward_manager=episode \
    env.env_name=GEM \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.num_attempts=3 \
    +env.do_reflection=True \
    env.max_steps=10 \
    env.max_turns=10 \
    +env.reflection_type=$reflection_type \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name=$MLFLOW_EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=4 \
    trainer.val_before_train=False \
    trainer.log_val_generations=1 \
    trainer.default_local_dir=/shared/public/sharing/sirzhu/Lamer/${EXPERIMENT_NAME}