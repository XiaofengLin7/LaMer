"""Single-game wrapper for GEM environments.

Each wrapper instance manages one task (one env_id + one seed),
playing ONE game round per LaMer episode (attempt). The meta-RL loop
handles multiple episodes with reflection between them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .env_adapters import resolve_adapter_class


class GEMMultiEpisodeWrapper:
    """Wraps a single inner environment for one game round per episode.

    Each call to reset()/restart() starts a fresh game. step() plays
    that game until it ends or max_turns_per_episode is reached.
    """

    def __init__(
        self,
        task_dict: Dict[str, Any],
        success_reward: float = 1.0,
    ):
        self.task_dict = task_dict
        self.env_id = task_dict['env_id']
        self.seed = task_dict.get('seed', 0)
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
        self._episode_step: int = 0
        self._is_done: bool = False
        self._won: bool = False
        self._init_observation: str = ""
        self._game_rules: str = ""

    @property
    def is_correct(self) -> bool:
        return self._won

    def reset(self) -> Tuple[str, dict]:
        self._episode_step = 0
        self._is_done = False
        self._won = False

        observation, info = self._reset_inner_env()
        self._init_observation = observation

        info['won'] = False
        return observation, info

    def restart(self) -> Tuple[str, dict]:
        """Restart for MetaRL: reset inner env for next episode, same task/seed."""
        self._episode_step = 0
        self._is_done = False
        self._won = False

        observation, info = self._reset_inner_env()
        self._init_observation = observation

        info['won'] = False
        return observation, info

    def step(self, text_action: str) -> Tuple[str, float, bool, dict]:
        # Guard: if already done, return no-op
        if self._is_done:
            info = {
                'won': self._won,
                'is_action_valid': False,
            }
            return "", 0.0, True, info

        observation, env_reward, inner_done, info = self.inner_env.step(text_action)
        self._episode_step += 1

        success = self._is_episode_success(
            done=inner_done, info=info, reward=env_reward
        )

        shaped_reward = self.success_reward if success else 0.0

        # Done when: agent wins OR step limit reached.
        # NOT on inner_done alone — gem games terminate after each guess
        # but allow multiple guesses within one episode (e.g., Hangman 10 guesses).
        outer_done = success or self._episode_step >= self.max_turns_per_episode
        if outer_done:
            self._is_done = True
        if success:
            self._won = True

        # If inner env is done but episode continues, reset for next guess
        if inner_done and not outer_done:
            self.inner_env.reset(seed=self.seed, task=self.task_dict)

        info['won'] = self._won
        info['is_action_valid'] = True

        return observation, shaped_reward, outer_done, info

    def get_init_observation(self) -> str:
        return self._init_observation

    def get_rules(self) -> str:
        return self._game_rules

    def close(self):
        if hasattr(self.inner_env, 'close'):
            self.inner_env.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_inner_env(self) -> Tuple[str, dict]:
        result = self.inner_env.reset(seed=self.seed, task=self.task_dict)
        if not self._game_rules and hasattr(self.inner_env, 'get_rules'):
            self._game_rules = self.inner_env.get_rules()
        return result

    @staticmethod
    def _is_episode_success(done: bool, info: dict, reward: float) -> bool:
        if not done:
            return False
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if truncated:
            return False
        return terminated and reward > 0
