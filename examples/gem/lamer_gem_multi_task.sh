set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Per-task train_size=512, 5 tasks => 2560 total train tasks
# Per-task test_size=256, 2 val tasks => 512 total val tasks
# train_batch_size = number of unique tasks per batch (before group repeat)
train_data_size=16
val_data_size=512
group_size=4
mode="mean_norm"
reflection_type="reflection_only"

# Multi-episode is handled inside GEM wrappers.
# LaMer adds reflection on top: num_attempts=3 with reflection between attempts.
# max_turns = max(total_step_cap) across all tasks = 30
GEM_CONFIG=examples/gem/multi_task_multi_episode_config.yaml
DATA_DIR=$HOME/data/gem-multi-task

# Step 1: Prepare data with Orbit-identical seeds
python3 -m examples.gem.prepare_gem_data \
    --config $GEM_CONFIG \
    --seed 42 \
    --output_dir $DATA_DIR

# Step 2: Train
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
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
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.seed=20 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name='lamer' \
    trainer.experiment_name=gem_multi_task_lamer_qwen3_4b \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=4 \
    trainer.val_before_train=False \
    trainer.log_val_generations=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    2>&1 | tee -a ../gem_multi_task_lamer_qwen3_4b.log
