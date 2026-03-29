"""Multi-episode wrapper for GEM environments.

Manages multiple episodes within a single trajectory:
- When inner env returns done=True, resets for next episode
- Outer trajectory ends when total_steps >= total_step_cap
- Reward shaping: success_reward on episode success, 0.0 otherwise
- Tracks episode successes for metrics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .env_adapters import resolve_adapter_class


class GEMMultiEpisodeWrapper:
    """Wraps a single inner environment with multi-episode logic.

    Each wrapper instance manages one task (one env_id + one seed),
    running multiple episodes until the total step budget is exhausted.
    """

    def __init__(
        self,
        task_dict: Dict[str, Any],
        success_reward: float = 1.0,
    ):
        """Initialize wrapper from a task dictionary.

        Args:
            task_dict: Must contain 'env_id', 'seed', 'total_step_cap',
                      'max_turns_per_episode', 'inner_env_class'.
                      May contain additional env-specific kwargs.
            success_reward: Reward on episode success.
        """
        self.task_dict = task_dict
        self.env_id = task_dict['env_id']
        self.seed = task_dict.get('seed', 0)
        self.total_step_cap = int(task_dict.get('total_step_cap', 30))
        self.max_turns_per_episode = int(task_dict.get('max_turns_per_episode', 10))
        self.success_reward = float(success_reward)

        # Resolve and create inner environment
        inner_cls_name = task_dict.get('inner_env_class', 'GEMEnvAdapter')
        inner_cls = resolve_adapter_class(inner_cls_name)

        # Build env_kwargs from task_dict, excluding meta fields
        meta_keys = {'env_id', 'seed', 'uid', 'data_source', 'total_step_cap',
                     'max_turns_per_episode', 'inner_env_class', 'train_size', 'test_size'}
        env_specific_kwargs = {k: v for k, v in task_dict.items() if k not in meta_keys}

        # Different adapters have different constructor signatures
        if inner_cls_name.endswith('GEMEnvAdapter') or 'GEMEnvAdapter' in inner_cls_name:
            self.inner_env = inner_cls(env_id=self.env_id, env_kwargs=env_specific_kwargs)
        elif 'RockPaperScissors' in inner_cls_name:
            rps_kwargs = {
                'max_turns': self.max_turns_per_episode,
                'min_dom': task_dict.get('min_dom', 0.4),
            }
            self.inner_env = inner_cls(env_id=self.env_id, env_kwargs=rps_kwargs)
        elif 'Blackjack' in inner_cls_name:
            self.inner_env = inner_cls(
                env_id=self.env_id,
                max_turns=self.max_turns_per_episode,
                seed=self.seed,
            )
        elif 'Maze' in inner_cls_name:
            maze_kwargs = {
                'shapes': [tuple(s) for s in task_dict.get('shapes', [[6, 6]])],
                'max_turns': self.max_turns_per_episode,
                'shortest_path_min_length': task_dict.get('shortest_path_min_length', 7),
                'shortest_path_max_length': task_dict.get('shortest_path_max_length', 8),
            }
            self.inner_env = inner_cls(env_id=self.env_id, env_kwargs=maze_kwargs)
        else:
            self.inner_env = inner_cls(env_id=self.env_id, env_kwargs=env_specific_kwargs)

        # Episode tracking state
        self._total_steps: int = 0
        self._episode_index: int = 0
        self._episode_step: int = 0
        self._episode_successes: List[bool] = []
        self._episode_lengths: List[int] = []
        self._init_observation: str = ""
        self._game_rules: str = ""

    @property
    def is_correct(self) -> bool:
        """Whether any episode in this trajectory succeeded."""
        return any(self._episode_successes)

    def reset(self) -> Tuple[str, dict]:
        """Reset for a new trajectory.

        Returns:
            observation: Initial observation with episode header.
            info: Info dict with 'won' = False.
        """
        self._total_steps = 0
        self._episode_index = 0
        self._episode_step = 0
        self._episode_successes = []
        self._episode_lengths = []

        observation, info = self._reset_inner_env()
        self._init_observation = observation
        observation = self._format_episode_header(observation, self._episode_index)

        info['won'] = False
        info['episode_index'] = self._episode_index
        info['total_steps'] = self._total_steps
        return observation, info

    def restart(self) -> Tuple[str, dict]:
        """Restart for MetaRL: reset to initial state, same task/seed.

        Returns:
            observation: Fresh initial observation.
            info: Info dict.
        """
        self._total_steps = 0
        self._episode_index = 0
        self._episode_step = 0
        self._episode_successes = []
        self._episode_lengths = []

        observation, info = self._reset_inner_env()
        self._init_observation = observation
        observation = self._format_episode_header(observation, self._episode_index)

        info['won'] = False
        info['episode_index'] = self._episode_index
        info['total_steps'] = self._total_steps
        return observation, info

    def step(self, text_action: str) -> Tuple[str, float, bool, dict]:
        """Execute one step in the multi-episode trajectory.

        Args:
            text_action: Raw text action from the LLM.

        Returns:
            observation: Next observation string.
            reward: Shaped reward (success_reward on success, 0.0 otherwise).
            done: True when total step budget is exhausted.
            info: Dict with 'won', 'episode_index', etc.
        """
        # Guard: if already done, return no-op (rollout loop may call step on done envs)
        if self._total_steps >= self.total_step_cap:
            info = {
                'won': self.is_correct,
                'episode_index': self._episode_index,
                'total_steps': self._total_steps,
                'episode_successes': list(self._episode_successes),
                'is_action_valid': False,
            }
            return "", 0.0, True, info

        observation, env_reward, inner_done, info = self.inner_env.step(text_action)

        self._total_steps += 1
        self._episode_step += 1

        success = self._is_episode_success(
            done=inner_done, info=info, reward=env_reward
        )

        shaped_reward = self.success_reward if success else 0.0
        outer_done = self._total_steps >= self.total_step_cap

        if inner_done:
            self._episode_successes.append(success)
            self._episode_lengths.append(self._episode_step)

            if not outer_done:
                # Reset inner env for next episode
                next_obs, reset_info = self._reset_inner_env()
                self._episode_index += 1
                self._episode_step = 0

                # Short transition header (game rules already in init_observation)
                ep_num = self._episode_index + 1
                observation = f"{observation}\n\n[Episode {ep_num}] New episode starts; reuse prior knowledge."

        # Budget exhausted mid-episode
        if outer_done and not inner_done and self._episode_step > 0:
            self._episode_successes.append(False)
            self._episode_lengths.append(self._episode_step)

        # Build info
        info['won'] = self.is_correct
        info['episode_index'] = self._episode_index
        info['total_steps'] = self._total_steps
        info['episode_successes'] = list(self._episode_successes)
        info['is_action_valid'] = True

        return observation, shaped_reward, outer_done, info

    def get_metrics(self) -> dict:
        """Return episode-level metrics for this trajectory."""
        num_episodes = len(self._episode_successes)
        success_count = sum(self._episode_successes)
        return {
            "episode/success_rate": 1.0 if self.is_correct else 0.0,
            "episode/num_episodes": num_episodes,
            "episode/success_count": success_count,
            "episode/total_steps": self._total_steps,
        }

    def get_init_observation(self) -> str:
        """Return the stored initial observation."""
        return self._init_observation

    def get_rules(self) -> str:
        """Return static game rules from the inner environment."""
        return self._game_rules

    def close(self):
        if hasattr(self.inner_env, 'close'):
            self.inner_env.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_inner_env(self) -> Tuple[str, dict]:
        """Reset the inner environment with the stored seed."""
        result = self.inner_env.reset(seed=self.seed, task=self.task_dict)
        # Capture rules on first reset
        if not self._game_rules and hasattr(self.inner_env, 'get_rules'):
            self._game_rules = self.inner_env.get_rules()
        return result

    @staticmethod
    def _is_episode_success(done: bool, info: dict, reward: float) -> bool:
        """Determine if an episode was successful."""
        if not done:
            return False
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if truncated:
            return False
        return terminated and reward > 0

    @staticmethod
    def _format_episode_header(observation: str, episode_index: int) -> str:
        """Prepend episode header to observation."""
        ep_num = episode_index + 1
        header = f"[Episode {ep_num}] New episode starts; reuse prior knowledge."
        return f"{header}\n{observation}"
