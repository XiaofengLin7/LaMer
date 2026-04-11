"""Single-task wrapper for SciWorld environments.

Each wrapper instance manages one task (one env_id + one seed),
playing ONE game round per LaMer episode (attempt). The meta-RL loop
handles multiple episodes with reflection between them.

Unlike GEM games that terminate after each guess and auto-reset,
SciWorld runs one continuous game per episode until success or max_turns.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .env_adapter import SciWorldEnvAdapter


class SciWorldMultiEpisodeWrapper:
    """Wraps a single SciWorld environment for one game per episode.

    Each call to reset()/restart() starts a fresh game (same task/seed).
    step() plays that game until it ends or max_turns is reached.
    """

    def __init__(
        self,
        task_dict: Dict[str, Any],
        success_reward: float = 1.0,
    ):
        self.task_dict = task_dict
        self.env_id = task_dict['env_id']
        self.seed = task_dict.get('seed', 0)
        self.max_turns_per_episode = int(task_dict.get('max_turns_per_episode', 15))
        self.success_reward = float(success_reward)

        # Create inner SciWorld adapter
        self.inner_env = SciWorldEnvAdapter(
            env_id=self.env_id,
            env_kwargs={
                'max_turns': self.max_turns_per_episode,
                'task_name': task_dict.get('task_name'),
                'split': task_dict.get('split', 'test'),
                'seed': self.seed,
            },
        )

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

        observation, info = self.inner_env.reset(seed=self.seed, task=self.task_dict)
        self._init_observation = observation

        if not self._game_rules:
            self._game_rules = self.inner_env.get_rules()

        info['won'] = False
        return observation, info

    def restart(self) -> Tuple[str, dict]:
        """Restart for MetaRL: reset inner env for next episode, same task/seed."""
        self._episode_step = 0
        self._is_done = False
        self._won = False

        observation, info = self.inner_env.reset(seed=self.seed, task=self.task_dict)
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

        success = info.get('is_correct', False)
        shaped_reward = self.success_reward if success else 0.0

        # SciWorld adapter handles truncation internally (done when
        # terminated or turn >= max_turns). Trust its done flag.
        if inner_done:
            self._is_done = True
        if success:
            self._won = True

        info['won'] = self._won
        info['is_action_valid'] = True

        return observation, shaped_reward, inner_done, info

    def get_init_observation(self) -> str:
        return self._init_observation

    def get_rules(self) -> str:
        return self._game_rules

    def close(self):
        if hasattr(self, 'inner_env'):
            self.inner_env.close()
