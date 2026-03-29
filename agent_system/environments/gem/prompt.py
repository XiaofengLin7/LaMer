"""LaMer-style prompt templates for GEM multi-task environments.

Follows the same pattern as minesweeper/prompt.py (Appendix B.1 style):
- Structured sections: Game Rules → Observation → Action prompt
- Game rules come from each adapter's get_rules() method
- Play prompt with init observation + past reflections/history + current trajectory
- Reflect prompt for meta-RL reflection phase
"""

# ---------------------------------------------------------------------------
# Play prompt: structured with explicit game rules and observation sections
# ---------------------------------------------------------------------------

GEM_PLAY_PROMPT = """You are an expert agent playing a game environment.

{game_rules}

# Observation
The initial game state is:
{init_observation}{past_trajectories_reflections}{current_trajectory}
Now it's your turn to make a move.
- First reason step-by-step about the current situation.
- Then choose your action following the action format specified above."""


# ---------------------------------------------------------------------------
# Reflect prompt: for meta-RL reflection phase
# ---------------------------------------------------------------------------

GEM_REFLECT_PROMPT = '''You are an expert agent playing a game environment.

{game_rules}

# Your Task
You will be given the history of a past experience.
Your job now is to **reflect on the past experience**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** for your next try starting from the original initial state.

# Past Experience
The initial game state is:
{init_observation}{current_trajectory}
The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.'''


# ---------------------------------------------------------------------------
# Templates for parsing past trajectories and reflections
# ---------------------------------------------------------------------------

PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
The task is NOT successfully completed. Your reflection is:
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
The task is NOT successfully completed.'''

REFLECTION_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is:
{reflection}'''


def parse_reflection(traj_idx, past_traj, reflection, reflection_type):
    """Format past trial reflections/history for injection into the prompt."""
    if traj_idx == 0 or len(reflection) == 0:
        return '\n'
    else:
        memories = []
        for _idx in range(traj_idx):
            if reflection_type == 'history_and_reflection':
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only':
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == 'reflection_only':
                memory = REFLECTION_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx]
                )
            else:
                raise ValueError(f"Unknown reflection_type: {reflection_type}")
            memories.append(memory)
        return ''.join(memories)


# ---------------------------------------------------------------------------
# Templates for parsing current trajectory
# ---------------------------------------------------------------------------

CURR_TRAJ_AT_TRAJ1 = '''
You have already taken the following actions:
{current_trajectory}
'''

CURR_TRAJ_AT_TRAJ2toN = '''

Currently you're on trial #{traj_idx}. You have already taken the following actions:
{current_trajectory}
'''

TRAJ_2toN_INIT = '''

Currently you're on trial #{traj_idx}, starting from the initial state.'''


def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    """Format current trajectory state for the prompt."""
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        else:
            return CURR_TRAJ_AT_TRAJ1.format(current_trajectory=curr_traj)
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1,
                current_trajectory=curr_traj
            )


def get_gem_prompt(phase: str = 'play',
                   turn_idx: int = 0,
                   traj_idx: int = 0,
                   game_rules: str = '',
                   init_observation: str = '',
                   curr_traj: str = '',
                   past_traj: str = '',
                   reflection: str = '',
                   reflection_type: str = 'reflection_only',
                   ):
    """Build the full prompt for a GEM environment step.

    Args:
        phase: 'play' or 'reflect'
        turn_idx: Current turn within the trajectory
        traj_idx: Current trial index (0-based, for MetaRL)
        game_rules: Static game rules from adapter.get_rules()
        init_observation: Dynamic initial game state from adapter.reset()
        curr_traj: Formatted current trajectory history
        past_traj: Dict of past trajectory histories (keyed by trial idx)
        reflection: Dict of past reflections (keyed by trial idx)
        reflection_type: One of 'reflection_only', 'history_only', 'history_and_reflection'
    """
    assert phase in ['play', 'reflect']

    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = GEM_PLAY_PROMPT.format(
            game_rules=game_rules,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
        )
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = GEM_REFLECT_PROMPT.format(
            game_rules=game_rules,
            init_observation=init_observation,
            current_trajectory=current_trajectory,
        )
    return prompt.strip()
