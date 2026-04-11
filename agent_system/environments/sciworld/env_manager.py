"""SciWorld environment manager for LaMer.

Reuses GEMEnvironmentManager directly since it is wrapper-agnostic.
Only the vectorized environment class (SciWorldMultiProcessEnv) and
the make_envs() factory are SciWorld-specific.
"""

from typing import List, Dict, Any

import gym

from .multi_episode_wrapper import SciWorldMultiEpisodeWrapper
from ..gem.env_manager import GEMEnvironmentManager
from ..gem.projection import gem_projection


# ===========================================================================
# Vectorized Environment
# ===========================================================================

class SciWorldMultiProcessEnv(gym.Env):
    """Vectorized SciWorld environment using local objects (no Ray actors).

    Each slot holds one SciWorldMultiEpisodeWrapper instance.
    Supports dynamic reconfiguration of tasks per batch.
    """

    def __init__(self, env_num=1, group_n=1, is_train=True, success_reward=1.0):
        super().__init__()
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.success_reward = success_reward

        # Local wrapper slots (created on reconfigure)
        self.wrappers: List[SciWorldMultiEpisodeWrapper | None] = [None] * self.num_processes

    def reconfigure(self, task_dicts: List[Dict[str, Any]]):
        """Assign task dicts to wrapper slots."""
        assert len(task_dicts) == self.num_processes, \
            f"Expected {self.num_processes} task dicts, got {len(task_dicts)}"

        for i, task_dict in enumerate(task_dicts):
            if self.wrappers[i] is not None:
                self.wrappers[i].close()
            self.wrappers[i] = SciWorldMultiEpisodeWrapper(
                task_dict=task_dict,
                success_reward=self.success_reward,
            )

    def step(self, actions: List[str]):
        """Sequential step across all wrappers."""
        assert len(actions) == self.num_processes
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, action in enumerate(actions):
            obs, reward, done, info = self.wrappers[i].step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """Sequential reset across all wrappers."""
        obs_list, info_list = [], []
        for w in self.wrappers:
            obs, info = w.reset()
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def restart(self):
        """Sequential restart for MetaRL."""
        obs_list, info_list = [], []
        for w in self.wrappers:
            obs, info = w.restart()
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def get_init_observations(self):
        """Get initial observations from all wrappers."""
        return [w.get_init_observation() for w in self.wrappers]

    def get_rules(self):
        """Get static game rules from all wrappers."""
        return [w.get_rules() for w in self.wrappers]

    def close(self):
        for w in self.wrappers:
            if w is not None:
                w.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ===========================================================================
# Factory function
# ===========================================================================

def make_envs(config):
    """Create SciWorld train and validation environments.

    Reuses GEMEnvironmentManager since it is wrapper-agnostic.

    Args:
        config: Hydra config with env settings.

    Returns:
        (envs, val_envs): Tuple of GEMEnvironmentManager instances.
    """
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    success_reward = config.env.get('sciworld', {}).get('success_reward', 1.0) if hasattr(config.env, 'sciworld') else 1.0

    # Create vectorized environments
    _envs = SciWorldMultiProcessEnv(
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        success_reward=success_reward,
    )
    _val_envs = SciWorldMultiProcessEnv(
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        success_reward=success_reward,
    )

    num_attempts = config.env.get('num_attempts', 1)
    do_reflection = config.env.get('do_reflection', False)
    val_num_attempts = config.env.get('val_num_attempts', num_attempts)
    val_do_reflection = config.env.get('val_do_reflection', do_reflection)

    projection_f = gem_projection

    envs = GEMEnvironmentManager(_envs, projection_f, num_attempts, do_reflection, config)
    val_envs = GEMEnvironmentManager(_val_envs, projection_f, val_num_attempts, val_do_reflection, config)

    return envs, val_envs
