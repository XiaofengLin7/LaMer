"""Tests for GEMMultiEpisodeWrapper stop-on-success and guard behavior.

Uses a mock inner env to avoid gem dependency. Run with:
    python -m agent_system.environments.gem.test_multi_episode_wrapper
"""

import sys
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Mock inner env that succeeds on a configurable step
# ---------------------------------------------------------------------------

class MockInnerEnv:
    """Deterministic mock env: succeeds on step `win_on_step`, fails otherwise."""

    def __init__(self, win_on_step=1, max_steps=4):
        self._win_on_step = win_on_step
        self._max_steps = max_steps
        self._step_count = 0
        self._done = False

    def reset(self, seed=None, task=None):
        self._step_count = 0
        self._done = False
        return "obs_reset", {"terminated": False, "truncated": False}

    def step(self, action):
        if self._done:
            raise RuntimeError("Environment is done. Call reset().")
        self._step_count += 1
        if self._step_count == self._win_on_step:
            self._done = True
            return "obs_win", 1.0, True, {"terminated": True, "truncated": False}
        if self._step_count >= self._max_steps:
            self._done = True
            return "obs_lose", 0.0, True, {"terminated": False, "truncated": True}
        return f"obs_step{self._step_count}", 0.0, False, {"terminated": False, "truncated": False}

    def get_rules(self):
        return "mock rules"

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helper to build a wrapper with the mock env
# ---------------------------------------------------------------------------

def _make_wrapper(win_on_step=1, total_step_cap=12, max_turns_per_episode=4):
    """Create a GEMMultiEpisodeWrapper with a MockInnerEnv patched in."""
    from agent_system.environments.gem.multi_episode_wrapper import GEMMultiEpisodeWrapper

    task_dict = {
        "env_id": "mock",
        "seed": 42,
        "total_step_cap": total_step_cap,
        "max_turns_per_episode": max_turns_per_episode,
        "inner_env_class": "GEMEnvAdapter",  # will be overridden
    }

    # Patch resolve_adapter_class so __init__ doesn't try to import gem
    with patch("agent_system.environments.gem.multi_episode_wrapper.resolve_adapter_class"):
        wrapper = GEMMultiEpisodeWrapper.__new__(GEMMultiEpisodeWrapper)
        # Manually init fields
        wrapper.task_dict = task_dict
        wrapper.env_id = "mock"
        wrapper.seed = 42
        wrapper.total_step_cap = total_step_cap
        wrapper.max_turns_per_episode = max_turns_per_episode
        wrapper.success_reward = 1.0
        wrapper.inner_env = MockInnerEnv(win_on_step=win_on_step, max_steps=max_turns_per_episode)
        wrapper._total_steps = 0
        wrapper._episode_index = 0
        wrapper._episode_step = 0
        wrapper._is_done = False
        wrapper._episode_successes = []
        wrapper._episode_lengths = []
        wrapper._init_observation = ""
        wrapper._game_rules = ""

    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stop_on_first_success():
    """Wrapper should set done=True immediately when an episode succeeds."""
    wrapper = _make_wrapper(win_on_step=1, total_step_cap=12)
    wrapper.reset()

    obs, reward, done, info = wrapper.step("action1")
    assert done, "Should be done after first episode success"
    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    assert info["won"], "info['won'] should be True"
    assert wrapper._is_done, "_is_done flag should be set"
    print("  PASS: test_stop_on_first_success")


def test_guard_after_success():
    """After success-done, further step() calls should return no-op, not crash."""
    wrapper = _make_wrapper(win_on_step=1, total_step_cap=12)
    wrapper.reset()

    # First step: success
    wrapper.step("action1")

    # Subsequent steps should NOT raise RuntimeError
    for i in range(5):
        obs, reward, done, info = wrapper.step(f"extra_action_{i}")
        assert done, f"Step {i}: should still be done"
        assert reward == 0.0, f"Step {i}: guard should return 0.0 reward"
        assert not info["is_action_valid"], f"Step {i}: guard should mark action invalid"
    print("  PASS: test_guard_after_success")


def test_no_reward_accumulation():
    """Total reward from a single attempt should be exactly 1.0, not accumulated."""
    wrapper = _make_wrapper(win_on_step=1, total_step_cap=12)
    wrapper.reset()

    total_reward = 0.0
    for _ in range(15):  # more steps than step cap
        obs, reward, done, info = wrapper.step("action")
        total_reward += reward
        if done:
            break

    assert total_reward == 1.0, f"Expected total reward 1.0, got {total_reward}"
    print("  PASS: test_no_reward_accumulation")


def test_failure_continues_episodes():
    """If episode fails, wrapper should continue to next episode (not stop)."""
    # win_on_step=999 means never wins
    wrapper = _make_wrapper(win_on_step=999, total_step_cap=12, max_turns_per_episode=4)
    wrapper.reset()

    steps = 0
    done = False
    while not done and steps < 20:
        obs, reward, done, info = wrapper.step("action")
        steps += 1

    assert steps == 12, f"Should exhaust all 12 steps, took {steps}"
    assert not info["won"], "Should not have won"
    # 12 steps / 4 per episode = 3 episodes
    assert len(wrapper._episode_successes) == 3, \
        f"Expected 3 episodes, got {len(wrapper._episode_successes)}"
    print("  PASS: test_failure_continues_episodes")


def test_restart_clears_done():
    """restart() should clear _is_done so the wrapper can be stepped again."""
    wrapper = _make_wrapper(win_on_step=1, total_step_cap=12)
    wrapper.reset()

    # Win first attempt
    wrapper.step("action1")
    assert wrapper._is_done

    # Restart for next MetaRL attempt
    wrapper.restart()
    assert not wrapper._is_done, "_is_done should be cleared after restart"

    # Should be able to step again without error
    obs, reward, done, info = wrapper.step("action1")
    assert done, "Should succeed again on first step"
    assert reward == 1.0
    print("  PASS: test_restart_clears_done")


def test_reset_clears_done():
    """reset() should clear _is_done."""
    wrapper = _make_wrapper(win_on_step=1, total_step_cap=12)
    wrapper.reset()
    wrapper.step("action1")
    assert wrapper._is_done

    wrapper.reset()
    assert not wrapper._is_done, "_is_done should be cleared after reset"
    print("  PASS: test_reset_clears_done")


if __name__ == "__main__":
    tests = [
        test_stop_on_first_success,
        test_guard_after_success,
        test_no_reward_accumulation,
        test_failure_continues_episodes,
        test_restart_clears_done,
        test_reset_clears_done,
    ]
    print(f"Running {len(tests)} tests...")
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
