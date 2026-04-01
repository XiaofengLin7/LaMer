"""Simulate all GEM games: bug check + prompt report.

Runs each game through multiple steps, including the multi-episode wrapper
and prompt builder, to verify correctness and show sample prompts.

Usage:
    conda run -n verl-agent python -m examples.gem.simulate_all_games
"""

from __future__ import annotations
import traceback
import sys

from agent_system.environments.gem.multi_episode_wrapper import GEMMultiEpisodeWrapper
from agent_system.environments.gem.prompt import get_gem_prompt
from agent_system.environments.gem.memory import SimpleMemoryGEM as SimpleMemory
from agent_system.environments.gem.projection import gem_projection


# ── Task configs matching multi_task_multi_episode_config.yaml ──────────────

TASKS = [
    {
        "name": "RockPaperScissors",
        "task_dict": {
            "env_id": "rockpaperscissors",
            "seed": 42,
            "max_turns_per_episode": 5,
            "total_step_cap": 15,
            "inner_env_class": "RockPaperScissorsEnvAdapter",
            "min_dom": 0.4,
        },
        "actions": ["rock", "paper", "scissors", "rock", "paper"],
    },
    {
        "name": "Minesweeper (only-reveal)",
        "task_dict": {
            "env_id": "game:Minesweeper-v0-only-reveal",
            "seed": 42,
            "max_turns_per_episode": 8,
            "total_step_cap": 24,
            "inner_env_class": "GEMEnvAdapter",
        },
        "actions": ["reveal 0 0", "reveal 1 1", "reveal 2 2", "reveal 0 1"],
    },
    {
        "name": "Hangman (easy)",
        "task_dict": {
            "env_id": "game:Hangman-v0-easy",
            "seed": 42,
            "max_turns_per_episode": 10,
            "total_step_cap": 30,
            "inner_env_class": "GEMEnvAdapter",
        },
        "actions": ["E", "A", "S", "T", "R"],
    },
    {
        "name": "Wordle (hard)",
        "task_dict": {
            "env_id": "game:Wordle-v0-hard",
            "seed": 42,
            "max_turns_per_episode": 10,
            "total_step_cap": 30,
            "inner_env_class": "GEMEnvAdapter",
        },
        "actions": ["CRANE", "SLOTH", "BLUFF", "PYGMY"],
    },
    {
        "name": "Blackjack",
        "task_dict": {
            "env_id": "game:Blackjack-v0",
            "seed": 42,
            "max_turns_per_episode": 4,
            "total_step_cap": 12,
            "inner_env_class": "BlackjackEnvAdapter",
        },
        "actions": ["hit 3", "stand", "hit 5", "stand"],
    },
    {
        "name": "Maze",
        "task_dict": {
            "env_id": "maze",
            "seed": 42,
            "max_turns_per_episode": 9,
            "total_step_cap": 27,
            "inner_env_class": "MazeEnvAdapter",
            "shapes": [[6, 6]],
            "shortest_path_min_length": 7,
            "shortest_path_max_length": 8,
        },
        "actions": ["right", "down", "right", "down", "right"],
    },
    {
        "name": "Mastermind",
        "task_dict": {
            "env_id": "game:Mastermind-v0-random",
            "seed": 42,
            "max_turns_per_episode": 3,
            "total_step_cap": 9,
            "inner_env_class": "GEMEnvAdapter",
            "code_length": 3,
            "num_numbers": 6,
            "duplicate_numbers": False,
        },
        "actions": ["1 2 3", "4 5 6", "1 3 5"],
    },
]


def simulate_game(task_cfg: dict) -> dict:
    """Run one game through wrapper + prompt pipeline. Returns report dict."""
    name = task_cfg["name"]
    task_dict = task_cfg["task_dict"]
    actions = task_cfg["actions"]
    report = {"name": name, "error": None, "steps": [], "prompts": {}}

    try:
        # ── Create wrapper and reset ────────────────────────────────
        wrapper = GEMMultiEpisodeWrapper(task_dict)
        init_obs, init_info = wrapper.reset()
        rules = wrapper.get_rules()

        report["rules_len"] = len(rules)
        report["init_obs"] = init_obs
        report["max_turns_per_episode"] = wrapper.max_turns_per_episode

        # ── Set up memory (like env_manager does) ───────────────────
        memory = SimpleMemory()
        memory.reset(1)  # 1 env

        # ── Build initial prompt (turn 0, episode 0) ────────────────
        prompt_t0 = get_gem_prompt(
            phase="play", turn_idx=0, traj_idx=0,
            game_rules=rules, init_observation=init_obs,
            curr_traj="", past_traj={}, reflection={},
            reflection_type="reflection_only",
        )
        report["prompts"]["episode1_turn0"] = prompt_t0

        # ── Step through actions ────────────────────────────────────
        for i, action in enumerate(actions):
            obs, reward, done, info = wrapper.step(action)
            step_info = {
                "action": action,
                "obs": obs[:150] + ("..." if len(obs) > 150 else ""),
                "reward": reward,
                "done": done,
                "won": info.get("won", False),
                "is_action_valid": info.get("is_action_valid", None),
                "terminated": info.get("terminated", None),
            }
            report["steps"].append(step_info)

            # Store in memory
            memory.store({
                "text_obs": [obs],
                "thought": [""],
                "action": [action],
                "reward": [reward],
                "dones": [done],
                "won": [info.get("won", False)],
            })

            if done:
                break

        # ── Build prompt after some steps (current trajectory) ──────
        # Use max_turns_per_episode as history/obs length (matches env_manager)
        max_turns = wrapper.max_turns_per_episode
        curr_trajs, _ = memory.fetch(history_length=max_turns, obs_length=max_turns)
        prompt_mid = get_gem_prompt(
            phase="play", turn_idx=len(report["steps"]), traj_idx=0,
            game_rules=rules, init_observation=init_obs,
            curr_traj=curr_trajs[0], past_traj={}, reflection={},
            reflection_type="reflection_only",
        )
        report["prompts"]["episode1_mid"] = prompt_mid

        # ── Build reflect prompt ────────────────────────────────────
        prompt_reflect = get_gem_prompt(
            phase="reflect", turn_idx=len(report["steps"]), traj_idx=0,
            game_rules=rules, init_observation=init_obs,
            curr_traj=curr_trajs[0], past_traj={}, reflection={},
            reflection_type="reflection_only",
        )
        report["prompts"]["reflect"] = prompt_reflect

        # ── Simulate episode 2 (after reflection) ──────────────────
        wrapper.restart()
        new_obs, new_info = wrapper.reset()

        # Build episode 2 initial prompt with mock reflection
        mock_reflection = {0: "I should try a different strategy. Last time I made suboptimal moves."}
        prompt_ep2 = get_gem_prompt(
            phase="play", turn_idx=0, traj_idx=1,
            game_rules=rules, init_observation=new_obs,
            curr_traj="", past_traj={}, reflection=mock_reflection,
            reflection_type="reflection_only",
        )
        report["prompts"]["episode2_turn0"] = prompt_ep2

        report["total_steps"] = len(report["steps"])
        report["final_done"] = report["steps"][-1]["done"] if report["steps"] else False
        report["final_won"] = report["steps"][-1]["won"] if report["steps"] else False

        # ── Verify: no truncation in current episode history ───────
        import re
        curr_traj_text = curr_trajs[0]
        num_steps = len(report["steps"])
        # Count step labels (e.g. "Action 1:", "Observation 3:") not occurrences in obs text
        action_labels = len(re.findall(r"^Action \d+:", curr_traj_text, re.MULTILINE))
        obs_truncated = len(re.findall(r"^Observation \d+: \.\.\.$", curr_traj_text, re.MULTILINE))
        report["history_check"] = {
            "steps_taken": num_steps,
            "actions_in_history": action_labels,
            "obs_truncated": obs_truncated,
            "has_truncation": obs_truncated > 0,
            "max_turns_per_episode": max_turns,
        }

    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return report


def print_report(report: dict):
    """Pretty-print a game simulation report."""
    name = report["name"]
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  GAME: {name}")
    print(sep)

    if report["error"]:
        print(f"  *** ERROR ***")
        print(f"  {report['error']}")
        return

    print(f"  max_turns_per_episode: {report['max_turns_per_episode']}")
    print(f"  rules length: {report['rules_len']} chars")
    print(f"  total steps taken: {report['total_steps']}")
    print(f"  final done: {report['final_done']}, final won: {report['final_won']}")

    hc = report.get("history_check")
    if hc:
        ok = not hc["has_truncation"] and hc["actions_in_history"] == hc["steps_taken"]
        status = "PASS" if ok else "FAIL"
        print(f"\n  --- History Check [{status}] ---")
        print(f"  steps_taken={hc['steps_taken']}, actions_in_history={hc['actions_in_history']}, "
              f"obs_truncated={hc['obs_truncated']}, max_turns={hc['max_turns_per_episode']}")

    print(f"\n  --- Steps ---")
    for i, step in enumerate(report["steps"]):
        print(f"  Step {i+1}: action='{step['action']}' reward={step['reward']:.1f} "
              f"done={step['done']} won={step['won']} valid={step['is_action_valid']}")
        print(f"          obs: {step['obs'][:120]}")

    # Print prompts
    for label, prompt in report["prompts"].items():
        print(f"\n  --- Prompt: {label} ({len(prompt)} chars) ---")
        # Indent each line for readability
        for line in prompt.split("\n"):
            print(f"  | {line}")


def main():
    print("Simulating all GEM games...")
    reports = []
    errors = []

    for task_cfg in TASKS:
        report = simulate_game(task_cfg)
        reports.append(report)
        if report["error"]:
            errors.append(report["name"])

    for report in reports:
        print_report(report)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for r in reports:
        status = "ERROR" if r["error"] else "OK"
        steps = r.get("total_steps", "?")
        won = r.get("final_won", "?")
        hc = r.get("history_check", {})
        hist_ok = "PASS" if (hc and not hc.get("has_truncation") and hc.get("actions_in_history") == hc.get("steps_taken")) else ("FAIL" if hc else "?")
        print(f"  [{status}] {r['name']:25s} steps={steps} won={won} history={hist_ok}")

    if errors:
        print(f"\n  *** {len(errors)} ERRORS: {', '.join(errors)} ***")
        return 1
    else:
        print(f"\n  All {len(reports)} games simulated successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
