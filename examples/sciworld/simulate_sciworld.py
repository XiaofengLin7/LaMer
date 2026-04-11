"""Simulate SciWorld tasks: bug check + prompt report.

Runs SciWorld tasks through the multi-episode wrapper and prompt builder
to verify correctness and show sample prompts/feedback.

Usage:
    conda run -n verl-agent python -m examples.sciworld.simulate_sciworld
"""

from __future__ import annotations
import traceback
import re
import sys

from agent_system.environments.sciworld.env_adapter import SciWorldEnvAdapter
from agent_system.environments.sciworld.multi_episode_wrapper import SciWorldMultiEpisodeWrapper
from agent_system.environments.gem.prompt import get_gem_prompt
from agent_system.environments.gem.memory import SimpleMemoryGEM as SimpleMemory


# ── Task configs matching multi_task_config.yaml ──────────────────────────

TASKS = [
    {
        "name": "find-animal",
        "task_dict": {
            "env_id": "sciworld:find-animal",
            "seed": 42,
            "max_turns_per_episode": 15,
            "total_step_cap": 45,
            "task_name": "find-animal",
            "split": "test",
            "inner_env_class": "SciWorldEnvAdapter",
        },
        "actions": [
            "look around",
            "go to hallway",
            "go to kitchen",
            "look around",
            "focus on frog",
        ],
    },
    {
        "name": "power-component",
        "task_dict": {
            "env_id": "sciworld:power-component",
            "seed": 42,
            "max_turns_per_episode": 15,
            "total_step_cap": 45,
            "task_name": "power-component",
            "split": "test",
            "inner_env_class": "SciWorldEnvAdapter",
        },
        "actions": [
            "look around",
            "pick up battery",
            "connect battery to light bulb",
            "activate battery",
        ],
    },
]


def simulate_task(task_cfg: dict) -> dict:
    """Run one SciWorld task through wrapper + prompt pipeline."""
    name = task_cfg["name"]
    task_dict = task_cfg["task_dict"]
    actions = task_cfg["actions"]
    report = {"name": name, "error": None, "steps": [], "prompts": {}}

    try:
        # ── Create wrapper and reset ────────────────────────────────
        wrapper = SciWorldMultiEpisodeWrapper(task_dict)
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
                "obs": obs[:200] + ("..." if len(obs) > 200 else ""),
                "reward": reward,
                "done": done,
                "won": info.get("won", False),
                "is_action_valid": info.get("is_action_valid", None),
                "score": info.get("score", None),
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

        # ── Build prompt with trajectory history ──────────────────
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

        # Verify same task: init_obs should be identical
        report["cross_episode_same_task"] = (init_obs == new_obs)

        # Build episode 2 initial prompt with mock reflection
        mock_reflection = {0: "I should explore more rooms and use the focus command on animals."}
        prompt_ep2 = get_gem_prompt(
            phase="play", turn_idx=0, traj_idx=1,
            game_rules=rules, init_observation=new_obs,
            curr_traj="", past_traj={0: curr_trajs[0]}, reflection=mock_reflection,
            reflection_type="reflection_only",
        )
        report["prompts"]["episode2_turn0"] = prompt_ep2

        report["total_steps"] = len(report["steps"])
        report["final_done"] = report["steps"][-1]["done"] if report["steps"] else False
        report["final_won"] = report["steps"][-1]["won"] if report["steps"] else False

        # ── Verify history: no truncation within episode ──────────
        curr_traj_text = curr_trajs[0]
        num_steps = len(report["steps"])
        action_labels = len(re.findall(r"^Action \d+:", curr_traj_text, re.MULTILINE))
        obs_truncated = len(re.findall(r"^Observation \d+: \.\.\.$", curr_traj_text, re.MULTILINE))
        report["history_check"] = {
            "steps_taken": num_steps,
            "actions_in_history": action_labels,
            "obs_truncated": obs_truncated,
            "has_truncation": obs_truncated > 0,
            "max_turns_per_episode": max_turns,
        }

        wrapper.close()

    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return report


def print_report(report: dict):
    """Pretty-print a task simulation report."""
    name = report["name"]
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  TASK: {name}")
    print(sep)

    if report["error"]:
        print(f"  *** ERROR ***")
        print(f"  {report['error']}")
        return

    print(f"  max_turns_per_episode: {report['max_turns_per_episode']}")
    print(f"  rules length: {report['rules_len']} chars")
    print(f"  total steps taken: {report['total_steps']}")
    print(f"  final done: {report['final_done']}, final won: {report['final_won']}")
    print(f"  cross-episode same task: {report.get('cross_episode_same_task', '?')}")

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
              f"done={step['done']} won={step['won']} score={step['score']}")
        print(f"          obs: {step['obs'][:150]}")

    # Print prompts
    for label, prompt in report["prompts"].items():
        print(f"\n  --- Prompt: {label} ({len(prompt)} chars) ---")
        for line in prompt.split("\n"):
            print(f"  | {line}")


def main():
    print("Simulating SciWorld tasks...")
    reports = []
    errors = []

    for task_cfg in TASKS:
        report = simulate_task(task_cfg)
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
        same_task = r.get("cross_episode_same_task", "?")
        hc = r.get("history_check", {})
        hist_ok = "PASS" if (hc and not hc.get("has_truncation") and
                             hc.get("actions_in_history") == hc.get("steps_taken")) else (
                  "FAIL" if hc else "?")
        print(f"  [{status}] {r['name']:45s} steps={steps} won={won} "
              f"same_task={same_task} history={hist_ok}")

    if errors:
        print(f"\n  *** {len(errors)} ERRORS: {', '.join(errors)} ***")
        return 1
    else:
        print(f"\n  All {len(reports)} tasks simulated successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
