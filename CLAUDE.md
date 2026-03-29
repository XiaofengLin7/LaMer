# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LaMer (Meta-RL Induces Exploration in Language Agents) is a Meta-RL framework for training LLM agents to actively explore and adapt to environments at test time (ICLR '26). Paper: https://arxiv.org/abs/2512.16848

Built on top of Bytedance's veRL (v0.3.1.dev) framework, extended with agent-specific multi-turn rollout and environment support.

## Common Commands

### Training
```bash
# LaMer meta-RL training (minesweeper example)
bash examples/minesweeper/lamer_minesweeper_qwen3_4b.sh

# GiGPO baseline
bash examples/minesweeper/gigpo_minesweeper_qwen3_4b.sh

# GRPO baseline
bash examples/minesweeper/grpo_minesweeper_qwen3_4b.sh
```
Environments: `minesweeper`, `sokoban`, `alfworld`, `webshop` — each has its own scripts under `examples/`.

### Testing Environments (no GPU required)
```bash
python examples/test_env.py
```

### Installation
```bash
pip install -e .
# ALFWorld: pip install alfworld
# WebShop: requires Python <=3.10, separate conda env
```

## Architecture

### Two-Layer Structure

**`verl/`** — Core distributed RL training framework (forked from veRL):
- `trainer/main_ppo.py` — Hydra entry point; everything starts here
- `trainer/ppo/ray_trainer.py` — `RayPPOTrainer`: orchestrates the PPO loop (rollout → reward → advantage → actor update → critic update)
- `trainer/ppo/core_algos.py` — GAE, KL control, standard PPO math
- `trainer/ppo/core_gigpo.py` — GiGPO (group-wise GRPO) advantage estimation
- `protocol.py` — `DataProto`: the universal data container (TensorDict batch + non-tensor dict + metadata) passed between all workers
- `workers/` — FSDP-sharded actor, critic, rollout (vLLM), reward workers
- `single_controller/` — Ray-based distributed coordination across machines

**`agent_system/`** — LaMer-specific agent logic layered on top of veRL:
- `environments/` — Environment wrappers (minesweeper, sokoban, alfworld, webshop), all implement `EnvironmentManagerBase` from `environments/base.py`
- `multi_turn_rollout/metarl_rollout_loop.py` — LaMer's multi-turn trajectory collection with reflection/history tracking (the key differentiator from standard RL)
- `multi_turn_rollout/rollout_loop.py` — Standard GiGPO single-turn rollout
- `reward_manager/episode.py` — Episode-level reward computation

### Training Flow

1. Hydra config loads from `verl/trainer/config/ppo_trainer.yaml` + CLI overrides
2. Ray cluster initializes, environment manager created via `make_envs()`
3. `RayPPOTrainer` runs the loop:
   - **Rollout**: Actor generates actions → environment steps → multi-turn trajectories (LaMer adds reflection between turns)
   - **Reward**: Episode-level reward computed
   - **Advantage**: GiGPO groups trajectories (group size = `env.rollout.n`), computes step-level + trajectory-level advantages
   - **Update**: FSDP-sharded actor/critic gradient updates

### Key Configuration

All training is configured via Hydra (YAML + CLI overrides). Important config knobs:
- `algorithm.adv_estimator` — `gigpo` or `gae`
- `algorithm.step_gamma` / `algorithm.traj_gamma` — step vs trajectory discount factors
- `env.reflection_type` — `reflection_only`, controls LaMer's meta-RL strategy
- `env.rollout.n` — group size for GiGPO advantage estimation
- `env.max_turns` — maximum interaction turns per episode
- `actor_rollout_ref.rollout.tensor_model_parallel_size` — vLLM tensor parallelism

### Adding a New Environment

1. Implement `EnvironmentManagerBase` (see `agent_system/environments/base.py`) with `reset()` and `step()` methods
2. Register it in `make_envs()` within the trainer
3. Create training scripts under `examples/<env_name>/`

### Model Support

Qwen2/3, LLaMA, and any HuggingFace transformers model. Model weights are loaded via the registry in `verl/models/registry.py`.
