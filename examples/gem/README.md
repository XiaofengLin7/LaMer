# GEM Multi-Task Multi-Episode Training & Evaluation

## Prerequisites

```bash
# Install LaMer
cd /path/to/LaMer
pip install -e .

# Install GEM environments
pip install gem-llm

# For ALFWorld OOD evaluation
pip install alfworld
```

## 1. Prepare Data

Generate train/val parquet files with Orbit-identical seeds:

```bash
python -m examples.gem.prepare_gem_data \
    --config examples/gem/multi_task_multi_episode_config.yaml \
    --seed 42 \
    --output_dir ~/data/gem-multi-task
```

This produces:
- `~/data/gem-multi-task/train.parquet` — 2560 tasks (512 per task x 5 train tasks)
- `~/data/gem-multi-task/test.parquet` — 512 tasks (256 maze + 256 mastermind)

**Train tasks**: RockPaperScissors, Minesweeper-only-reveal, Hangman-easy, Wordle-hard, Blackjack
**Val tasks**: Maze, Mastermind-random

## 2. Train

```bash
bash examples/gem/lamer_gem_multi_task.sh
```

Key config knobs you may want to adjust in the script:
- `train_data_size` — number of unique tasks per training batch (before group repeat)
- `group_size` — GiGPO group size (`env.rollout.n`), each task repeated N times
- `actor_rollout_ref.model.path` — model to train (default: `Qwen/Qwen3-4B`)
- `trainer.n_gpus_per_node` / `trainer.nnodes` — GPU resources
- `trainer.total_epochs` — total training epochs

The multi-episode logic runs inside the environment wrappers (`num_attempts=1`, `do_reflection=False` at the rollout level). Each task has its own step budget (e.g., Blackjack: 12 steps, Wordle: 30 steps). The rollout loop's `max_turns=30` accommodates the longest task; shorter tasks return `done=True` earlier.

## 3. Evaluate on GEM Val Tasks (Maze + Mastermind)

Validation runs automatically during training at every `trainer.test_freq` epochs on the maze and mastermind tasks from the val parquet. Check WandB or console logs for per-task success rates.

To run standalone validation on a checkpoint:

```bash
# Same script but with val_before_train=True and total_epochs=0
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=~/data/gem-multi-task/train.parquet \
    data.val_files=~/data/gem-multi-task/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/path/to/checkpoint \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward_model.reward_manager=episode \
    env.env_name=GEM \
    env.seed=0 \
    env.rollout.n=1 \
    env.num_attempts=1 \
    env.do_reflection=False \
    env.max_steps=30 \
    env.max_turns=30 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1
```

## 4. Evaluate on ALFWorld (134 OOD Tasks)

ALFWorld uses LaMer's existing ALFWorld environment, not the GEM wrapper. Run it separately with the GEM-trained checkpoint:

```bash
# Prepare ALFWorld data (if not already done)
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 8 \
    --val_data_size 134

# Evaluate
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=134 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/path/to/gem-trained-checkpoint \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.seed=20 \
    reward_model.reward_manager=episode \
    env.env_name="alfworld/AlfredTWEnv" \
    env.seed=0 \
    env.rollout.n=1 \
    env.num_attempts=1 \
    env.max_steps=30 \
    env.max_turns=10 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1
```

## Task Configuration Reference

| Task | turns/ep | step cap | Adapter |
|------|----------|----------|---------|
| rockpaperscissors | 5 | 15 | RockPaperScissorsEnvAdapter |
| Minesweeper-v0-only-reveal | 8 | 24 | GEMEnvAdapter |
| Hangman-v0-easy | 10 | 30 | GEMEnvAdapter |
| Wordle-v0-hard | 10 | 30 | GEMEnvAdapter |
| Blackjack-v0 | 4 | 12 | BlackjackEnvAdapter |
| maze (val) | 9 | 27 | MazeEnvAdapter |
| Mastermind-v0-random (val) | 3 | 9 | GEMEnvAdapter |

## Notes

- Seeds are generated using SHA256 hashing, identical to Orbit's `prepare_gem_data.py`, ensuring fair comparison.
- Multi-episode logic is inside the environment wrapper. From the rollout loop's perspective, each GEM env looks like a single long episode with up to `total_step_cap` steps.
- To customize task mix or sizes, edit `examples/gem/multi_task_multi_episode_config.yaml` and re-run data preparation.
