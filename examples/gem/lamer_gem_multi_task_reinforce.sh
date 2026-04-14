set -x
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
export NLTK_DATA=/home/jobuser/nltk_data
python -c "
import nltk
nltk.corpus.words.words('en-basic')
print('NLTK words corpus pre-loaded')
"

train_batch_size=64
val_batch_size=128
group_size=4
reflection_type="${REFLECTION_TYPE:-history_and_reflection}"

GEM_CONFIG=examples/gem/multi_task_multi_episode_config.yaml
DATA_DIR=$HOME/data/gem-multi-task
MODEL_NAME=Qwen3-8B
DISABLE_THINKING=False
ENABLE_REFLECTION=True
CONFIG_NAME=$(basename "$GEM_CONFIG" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
EXPERIMENT_NAME="gem-multi-task-reinforce-${CONFIG_NAME}-${MODEL_NAME}-disable-thinking-${DISABLE_THINKING}-enable-reflection-${ENABLE_REFLECTION}-reflection-type-${reflection_type}-${FLYTE_INTERNAL_EXECUTION_ID}"
MODEL_PATH="/shared/public/elr-models/Qwen/Qwen3-8B/2069b3fae1114555f3c020c81410e51fa0f656f2_130k_context"

python3 -m examples.gem.prepare_gem_data \
    --config $GEM_CONFIG \
    --seed 42 \
    --output_dir $DATA_DIR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=7168 \
    data.max_response_length=1024 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=8192 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
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
    trainer.total_epochs=8 \
    trainer.val_before_train=True \
    trainer.log_val_generations=1 \
    trainer.default_local_dir=/shared/public/sharing/sirzhu/Lamer/${EXPERIMENT_NAME}
