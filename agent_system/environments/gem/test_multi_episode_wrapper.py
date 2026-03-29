"""Tests for GEMMultiEpisodeWrapper behavior.

Gem games terminate after each guess (terminated=True). The wrapper
auto-resets the inner env and continues until max_turns_per_episode
or success (win). Run with:
    python -m agent_system.environments.gem.test_multi_episode_wrapper
"""

import sys
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Mock inner env: terminates after each step (like gem Hangman/Wordle)
# ---------------------------------------------------------------------------

class MockGemEnv:
    """Mock env that terminates after every step. Wins on step `win_on_step`."""

    def __init__(self, win_on_step=999):
        self._win_on_step = win_on_step
        self._total_steps = 0

    def reset(self, seed=None, task=None):
        return "obs_reset", {"terminated": False, "truncated": False}

    def step(self, action):
        self._total_steps += 1
        if self._total_steps == self._win_on_step:
            return "obs_win", 1.0, True, {"terminated": True, "truncated": False}
        # Wrong guess: terminated=True, negative reward (like gem games)
        return "obs_wrong", -0.1, True, {"terminated": True, "truncated": False}

    def get_rules(self):
        return "mock rules"

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_wrapper(win_on_step=999, max_turns_per_episode=10):
    from agent_system.environments.gem.multi_episode_wrapper import GEMMultiEpisodeWrapper

    with patch("agent_system.environments.gem.multi_episode_wrapper.resolve_adapter_class"):
        wrapper = GEMMultiEpisodeWrapper.__new__(GEMMultiEpisodeWrapper)
        wrapper.task_dict = {"env_id": "mock", "seed": 42}
        wrapper.env_id = "mock"
        wrapper.seed = 42
        wrapper.max_turns_per_episode = max_turns_per_episode
        wrapper.success_reward = 1.0
        wrapper.inner_env = MockGemEnv(win_on_step=win_on_step)
        wrapper._episode_step = 0
        wrapper._is_done = False
        wrapper._won = False
        wrapper._init_observation = ""
        wrapper._game_rules = ""

    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_continues_after_wrong_guess():
    """Inner env terminates on wrong guess, but wrapper continues until max_turns."""
    wrapper = _make_wrapper(win_on_step=999, max_turns_per_episode=10)
    wrapper.reset()

    for i in range(9):
        obs, reward, done, info = wrapper.step(f"guess{i}")
        assert not done, f"Step {i+1}: should NOT be done (wrong guess, still have turns)"
        assert reward == 0.0

    obs, reward, done, info = wrapper.step("guess10")
    assert done, "Step 10: should be done (max_turns_per_episode reached)"
    assert not info["won"]
    print("  PASS: test_continues_after_wrong_guess")


def test_stops_on_success():
    """Wrapper stops immediately when agent wins."""
    wrapper = _make_wrapper(win_on_step=3, max_turns_per_episode=10)
    wrapper.reset()

    obs, reward, done, info = wrapper.step("guess1")
    assert not done
    obs, reward, done, info = wrapper.step("guess2")
    assert not done

    obs, reward, done, info = wrapper.step("guess3")
    assert done, "Should stop on success"
    assert reward == 1.0
    assert info["won"]
    assert wrapper._episode_step == 3
    print("  PASS: test_stops_on_success")


def test_episode_length_matches_max_turns():
    """Episode step count equals max_turns_per_episode when no success."""
    for max_turns in [4, 5, 8, 10]:
        wrapper = _make_wrapper(win_on_step=999, max_turns_per_episode=max_turns)
        wrapper.reset()
        steps = 0
        done = False
        while not done:
            obs, reward, done, info = wrapper.step("guess")
            steps += 1
        assert steps == max_turns, \
            f"max_turns={max_turns}: expected {max_turns} steps, got {steps}"
    print("  PASS: test_episode_length_matches_max_turns")


def test_guard_after_done():
    """After done, further step() calls return no-op without crashing."""
    wrapper = _make_wrapper(win_on_step=1, max_turns_per_episode=5)
    wrapper.reset()
    wrapper.step("guess1")  # wins, done

    for i in range(5):
        obs, reward, done, info = wrapper.step(f"extra_{i}")
        assert done
        assert reward == 0.0
        assert not info["is_action_valid"]
    print("  PASS: test_guard_after_done")


def test_restart_clears_done():
    """restart() allows a new episode on the same task."""
    wrapper = _make_wrapper(win_on_step=1, max_turns_per_episode=5)
    wrapper.reset()
    wrapper.step("guess1")
    assert wrapper._is_done

    wrapper.restart()
    assert not wrapper._is_done
    assert not wrapper._won
    # Inner env step counter persists, so win_on_step=1 already fired.
    # But restart resets inner env, so a new game starts.
    print("  PASS: test_restart_clears_done")


def test_reset_clears_done():
    """reset() clears done state."""
    wrapper = _make_wrapper(win_on_step=1, max_turns_per_episode=5)
    wrapper.reset()
    wrapper.step("guess1")
    assert wrapper._is_done

    wrapper.reset()
    assert not wrapper._is_done
    assert not wrapper._won
    print("  PASS: test_reset_clears_done")


def test_reward_only_on_success():
    """Reward is 1.0 only on success step, 0.0 for all wrong guesses."""
    wrapper = _make_wrapper(win_on_step=5, max_turns_per_episode=10)
    wrapper.reset()

    total_reward = 0.0
    for i in range(10):
        obs, reward, done, info = wrapper.step("guess")
        total_reward += reward
        if done:
            break

    assert total_reward == 1.0, f"Expected 1.0 total reward, got {total_reward}"
    print("  PASS: test_reward_only_on_success")


if __name__ == "__main__":
    tests = [
        test_continues_after_wrong_guess,
        test_stops_on_success,
        test_episode_length_matches_max_turns,
        test_guard_after_done,
        test_restart_clears_done,
        test_reset_clears_done,
        test_reward_only_on_success,
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
