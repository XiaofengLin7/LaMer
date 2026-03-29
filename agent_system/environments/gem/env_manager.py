"""GEM multi-task environment manager for LaMer.

Follows the same pattern as MineSweeperEnvironmentManager:
- Ray-based vectorized environments
- MetaRL support (restart, reflect)
- Memory-based history tracking
- LaMer-style prompt building
"""

from typing import List, Dict, Any
from collections import defaultdict
from functools import partial

import gym
import numpy as np

from .multi_episode_wrapper import GEMMultiEpisodeWrapper
from .prompt import get_gem_prompt
from .memory import SimpleMemoryGEM as SimpleMemory
from .projection import gem_projection
from ..base import EnvironmentManagerBase, to_numpy


# ===========================================================================
# Ray Worker
# ===========================================================================

class GEMMultiProcessEnv(gym.Env):
    """Vectorized GEM environment using local objects (no Ray actors).

    Each slot holds one GEMMultiEpisodeWrapper instance.
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
        self.wrappers: List[GEMMultiEpisodeWrapper | None] = [None] * self.num_processes

    def reconfigure(self, task_dicts: List[Dict[str, Any]]):
        """Assign task dicts to wrapper slots."""
        assert len(task_dicts) == self.num_processes, \
            f"Expected {self.num_processes} task dicts, got {len(task_dicts)}"

        for i, task_dict in enumerate(task_dicts):
            if self.wrappers[i] is not None:
                self.wrappers[i].close()
            self.wrappers[i] = GEMMultiEpisodeWrapper(
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
# Environment Manager
# ===========================================================================

class GEMEnvironmentManager(EnvironmentManagerBase):
    """Manages GEM multi-task multi-episode environments for LaMer.

    Follows the same interface as MineSweeperEnvironmentManager:
    - reset(), restart(), reflect(), step(), build_text_obs()
    - success_evaluator()
    - reconfigure_from_batch() for dynamic task assignment
    """

    def __init__(self, envs: GEMMultiProcessEnv, projection_f, num_attempts, do_reflection, config):
        self.num_attempts = num_attempts
        self.num_processes = envs.num_processes
        self.do_reflection = do_reflection
        self.reflection_type = config.env.get('reflection_type', 'reflection_only')
        assert self.reflection_type in ['history_and_reflection', 'reflection_only', 'history_only']

        # Parse max_turns from config; this should be max(total_step_cap) across all tasks
        self.max_turns = config.env.get('max_turns', 30)

        # Init states (initial observations from inner envs)
        self.init_states = [None for _ in range(self.num_processes)]

        # Static game rules per worker
        self.game_rules = ['' for _ in range(self.num_processes)]

        # Per-worker data_source for per-task metric grouping
        self.data_sources = [None for _ in range(self.num_processes)]

        # Memories for MetaRL
        self.memories = [SimpleMemory() for _ in range(self.num_attempts)]
        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0

        super().__init__(envs, projection_f, config)

    def reconfigure_from_batch(self, gen_batch):
        """Extract task info from gen_batch and reconfigure environment workers.

        Args:
            gen_batch: DataProto with non_tensor_batch['extra_info'] containing
                      task dicts for each environment instance.
        """
        if 'extra_info' not in gen_batch.non_tensor_batch:
            return  # Not a GEM batch, skip

        extra_infos = gen_batch.non_tensor_batch['extra_info']
        batch_size = len(extra_infos)

        # Extract task dicts
        task_dicts = []
        for i in range(batch_size):
            info = extra_infos[i]
            if isinstance(info, dict):
                task_dicts.append(info)
            else:
                task_dicts.append(dict(info))

        assert len(task_dicts) == self.num_processes, \
            f"Expected {self.num_processes} tasks, got {len(task_dicts)}"

        # Store data_sources and max_episodes for per-task metrics
        self.data_sources = [td.get('data_source', 'unknown') for td in task_dicts]
        self.max_episodes = [
            int(td.get('total_step_cap', 30)) // int(td.get('max_turns_per_episode', 10))
            for td in task_dicts
        ]

        # Reconfigure workers
        self.envs.reconfigure(task_dicts)

    def reset(self):
        """Reset all environments."""
        obs, infos = self.envs.reset()

        # Store initial observations and game rules
        self.init_states = self.envs.get_init_observations()
        self.game_rules = self.envs.get_rules()

        # Reset memories and reflections
        for memory in self.memories:
            memory.reset(self.num_processes)
        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0

        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs,
        }
        return observations, infos

    def restart(self):
        """Restart for MetaRL 2nd+ attempts."""
        obs, infos = self.envs.restart()

        # Store fresh initial observations
        self.init_states = self.envs.get_init_observations()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': obs,
        }
        return observations, infos

    def reflect(self):
        """Get prompts for reflect phase."""
        infos = [{"action_is_valid": True, "won": False} for _ in range(self.num_processes)]
        observations = {
            'text': self.build_text_obs(phase='reflect'),
            'image': None,
            'anchor': ['reflection' for _ in range(self.num_processes)],
        }
        return observations, infos

    def step(self, text_actions: List[str], phase: str = 'play'):
        assert phase in ['play', 'reflect']

        if phase == 'reflect':
            reflections, valids = self.projection_f(text_actions, phase='reflect')
            for i, reflection in enumerate(reflections):
                self.reflections[i][self.curr_traj_idx] = reflection

            infos = [{"action_is_valid": False, "won": False} for _ in range(self.num_processes)]
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])

            next_observations = {'text': '', 'image': None, 'anchor': ''}
            rewards = np.array(valids, dtype=float)
            dones = np.array([False] * len(text_actions))
            return next_observations, rewards, dones, infos

        else:
            # Forward raw text to wrappers (they handle parsing internally)
            thoughts, actions, valids = self.projection_f(text_actions, phase='play')
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # Add action validity to infos
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])

            # Store in memory
            self.memories[self.curr_traj_idx].store({
                'text_obs': next_obs,
                'thought': thoughts,
                'action': actions,
                'reward': rewards,
                'dones': dones,
                'won': [info.get('won', False) for info in infos],
            })
            self.curr_turn_idx += 1

            next_observations = {
                'text': self.build_text_obs(phase='play'),
                'image': None,
                'anchor': next_obs,
            }

            rewards = to_numpy(np.array(rewards, dtype=float))
            dones = to_numpy(np.array(dones))

            return next_observations, rewards, dones, infos

    def build_text_obs(self, phase: str = 'play') -> List[str]:
        """Build text observations with LaMer-style prompts.

        Uses obs_length to control how many recent steps show full observations
        (older steps are truncated to '...'). This keeps prompt length bounded
        even for long multi-episode trajectories across multiple attempts.
        """
        postprocess_text_obs = []
        assert phase in ['play', 'reflect']

        obs_length = 2 if phase == 'play' else 5
        history_length = 7 if phase == 'play' else 10
        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            curr_trajs, _ = self.memories[self.curr_traj_idx].fetch(
                history_length=history_length, obs_length=obs_length)

        past_trajs = [{} for _ in range(self.num_processes)]
        for traj_idx in range(self.curr_traj_idx):
            # Past trajectories use short summaries (obs_length=1)
            trajectories, _ = self.memories[traj_idx].fetch(
                history_length=history_length, obs_length=1)
            for i in range(self.num_processes):
                past_trajs[i][traj_idx] = trajectories[i]

        for i in range(self.num_processes):
            obs = get_gem_prompt(
                phase=phase,
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                game_rules=self.game_rules[i] if i < len(self.game_rules) else '',
                init_observation=self.init_states[i] if self.init_states[i] is not None else '',
                curr_traj=curr_trajs[i],
                past_traj=past_trajs[i],
                reflection=self.reflections[i],
                reflection_type=self.reflection_type,
            )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Evaluate episode success with per-task and per-episode breakdown.

        Reports:
        - success_rate[0]: overall any-episode success (for compatibility)
        - success_rate/{data_source}: per-task overall success rate
        - success_rate/{data_source}/episode_{n}: per-task per-episode success rate
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        # Per-task per-episode tracking: {data_source: {ep_idx: [bool, ...]}}
        per_task_episode = defaultdict(lambda: defaultdict(list))
        # Per-task overall tracking
        per_task_overall = defaultdict(list)

        for bs in range(batch_size):
            data_source = self.data_sources[bs] if bs < len(self.data_sources) else 'unknown'

            # Find the last active play step to get final episode_successes
            episode_successes = []
            won = False
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item['active_masks'] and batch_item['phase'] == 'play':
                    info = total_infos[bs][i]
                    episode_successes = info.get('episode_successes', [])
                    won = info.get('won', False)
                    break

            # Overall success (any episode succeeded)
            success['success_rate[0]'].append(won)

            # Per-task overall
            per_task_overall[data_source].append(won)

            # Per-task per-episode (capped at total_step_cap // max_turns_per_episode)
            max_ep = self.max_episodes[bs] if bs < len(self.max_episodes) else 3
            for ep_idx, ep_success in enumerate(episode_successes):
                if ep_idx >= max_ep:
                    break
                per_task_episode[data_source][ep_idx].append(ep_success)

        # Add per-task overall success rates
        for ds, vals in per_task_overall.items():
            success[f'success_rate/{ds}'] = vals

        # Add per-task per-episode success rates
        for ds, ep_dict in per_task_episode.items():
            for ep_idx in sorted(ep_dict.keys()):
                success[f'success_rate/{ds}/episode_{ep_idx + 1}'] = ep_dict[ep_idx]

        return {key: np.array(value) for key, value in success.items()}


# ===========================================================================
# Factory function
# ===========================================================================

def make_envs(config):
    """Create GEM train and validation environments.

    Args:
        config: Hydra config with env.gem settings.

    Returns:
        (envs, val_envs): Tuple of GEMEnvironmentManager instances.
    """
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    success_reward = config.env.get('gem', {}).get('success_reward', 1.0) if hasattr(config.env, 'gem') else 1.0

    # Create vectorized environments
    _envs = GEMMultiProcessEnv(
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        success_reward=success_reward,
    )
    _val_envs = GEMMultiProcessEnv(
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
