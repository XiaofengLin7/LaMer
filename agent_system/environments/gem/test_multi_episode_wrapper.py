"""Tests for GEMMultiEpisodeWrapper single-game-per-episode behavior.

Uses a mock inner env to avoid gem dependency. Run with:
    python -m agent_system.environments.gem.test_multi_episode_wrapper
"""

import sys
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Mock inner env
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

def _make_wrapper(win_on_step=999, max_turns_per_episode=5, inner_max_steps=10):
    """Create a GEMMultiEpisodeWrapper with a MockInnerEnv patched in."""
    from agent_system.environments.gem.multi_episode_wrapper import GEMMultiEpisodeWrapper

    with patch("agent_system.environments.gem.multi_episode_wrapper.resolve_adapter_class"):
        wrapper = GEMMultiEpisodeWrapper.__new__(GEMMultiEpisodeWrapper)
        wrapper.task_dict = {"env_id": "mock", "seed": 42}
        wrapper.env_id = "mock"
        wrapper.seed = 42
        wrapper.max_turns_per_episode = max_turns_per_episode
        wrapper.success_reward = 1.0
        wrapper.inner_env = MockInnerEnv(win_on_step=win_on_step, max_steps=inner_max_steps)
        wrapper._episode_step = 0
        wrapper._is_done = False
        wrapper._won = False
        wrapper._init_observation = ""
        wrapper._game_rules = ""

    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_done_on_inner_done():
    """Wrapper returns done when inner game ends (win or loss)."""
    # Win on step 2
    wrapper = _make_wrapper(win_on_step=2, max_turns_per_episode=10)
    wrapper.reset()

    obs, reward, done, info = wrapper.step("action1")
    assert not done, "Step 1: game not over yet"
    assert reward == 0.0

    obs, reward, done, info = wrapper.step("action2")
    assert done, "Step 2: inner game ended (win)"
    assert reward == 1.0
    assert info["won"]
    print("  PASS: test_done_on_inner_done")


def test_done_on_step_limit():
    """Wrapper returns done at max_turns_per_episode even if inner game not finished."""
    # Never wins, inner game has 10 max steps, but wrapper limits to 5
    wrapper = _make_wrapper(win_on_step=999, max_turns_per_episode=5, inner_max_steps=10)
    wrapper.reset()

    for i in range(4):
        obs, reward, done, info = wrapper.step(f"action{i}")
        assert not done, f"Step {i+1}: should not be done yet"

    obs, reward, done, info = wrapper.step("action5")
    assert done, "Step 5: should be done (max_turns_per_episode reached)"
    assert not info["won"]
    print("  PASS: test_done_on_step_limit")


def test_success_reward():
    """Success gives reward 1.0, failure gives 0.0."""
    # Win on step 1
    wrapper = _make_wrapper(win_on_step=1, max_turns_per_episode=5)
    wrapper.reset()
    obs, reward, done, info = wrapper.step("action1")
    assert reward == 1.0
    assert info["won"]

    # Never win, exhaust steps
    wrapper2 = _make_wrapper(win_on_step=999, max_turns_per_episode=3, inner_max_steps=10)
    wrapper2.reset()
    total_reward = 0.0
    for i in range(3):
        obs, reward, done, info = wrapper2.step(f"action{i}")
        total_reward += reward
    assert total_reward == 0.0, f"Expected 0.0 total reward for failure, got {total_reward}"
    assert not info["won"]
    print("  PASS: test_success_reward")


def test_guard_after_done():
    """After done, further step() calls return no-op without crashing."""
    wrapper = _make_wrapper(win_on_step=1, max_turns_per_episode=5)
    wrapper.reset()
    wrapper.step("action1")  # wins, done

    for i in range(5):
        obs, reward, done, info = wrapper.step(f"extra_{i}")
        assert done
        assert reward == 0.0
        assert not info["is_action_valid"]
    print("  PASS: test_guard_after_done")


def test_restart_clears_done():
    """restart() clears done state for next meta-RL episode."""
    wrapper = _make_wrapper(win_on_step=1, max_turns_per_episode=5)
    wrapper.reset()
    wrapper.step("action1")  # wins, done
    assert wrapper._is_done

    wrapper.restart()
    assert not wrapper._is_done
    assert not wrapper._won

    # Can step again
    obs, reward, done, info = wrapper.step("action1")
    assert done and reward == 1.0
    print("  PASS: test_restart_clears_done")


def test_reset_clears_done():
    """reset() clears done state."""
    wrapper = _make_wrapper(win_on_step=1, max_turns_per_episode=5)
    wrapper.reset()
    wrapper.step("action1")
    assert wrapper._is_done

    wrapper.reset()
    assert not wrapper._is_done
    assert not wrapper._won
    print("  PASS: test_reset_clears_done")


def test_no_multi_episode_within_attempt():
    """Wrapper does NOT auto-reset for another game within a single attempt.
    When inner game ends, wrapper is done — no second game round."""
    # Inner game ends on step 2 (loss via truncation at max_steps=2)
    wrapper = _make_wrapper(win_on_step=999, max_turns_per_episode=10, inner_max_steps=2)
    wrapper.reset()

    wrapper.step("action1")
    obs, reward, done, info = wrapper.step("action2")
    assert done, "Should be done when inner game ends"

    # Guard returns no-op, doesn't start a new game
    obs, reward, done, info = wrapper.step("action3")
    assert done
    assert not info["is_action_valid"]
    assert wrapper._episode_step == 2, f"Episode step should stay at 2, got {wrapper._episode_step}"
    print("  PASS: test_no_multi_episode_within_attempt")


if __name__ == "__main__":
    tests = [
        test_done_on_inner_done,
        test_done_on_step_limit,
        test_success_reward,
        test_guard_after_done,
        test_restart_clears_done,
        test_reset_clears_done,
        test_no_multi_episode_within_attempt,
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
