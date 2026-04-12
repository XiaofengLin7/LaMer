ALFWORLD_PLAY_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.
{init_observation}{past_trajectories_reflections}{current_trajectory}

Your admissible actions of the current situation are:
[{admissible_actions}]

Now it's your turn to take an action.

- Your response should first be step-by-step reasoning about the current situation.
- Once you've finished your reasoning, you should choose an admissible action for the current step and present it as \\boxed{{your action}}.
"""


ALFWORLD_REFLECT_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.
{init_observation}

You will be given the history of a past experience.
Your job is to **reflect on the past sequence**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** starting from the original initial state.

Below are the actions you took and the corresponding observations:
{current_trajectory}
{completion_status}

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.
"""

# Prompt templates for parsing past trajectories and reflections
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE = '''

On trial #{traj_idx}, the actions you took and the corresponding observations are:
{past_trajectory}
{completion_status} Your reflection is:
{reflection}'''

HISTORY_ONLY_TEMPLATE = '''

On trial #{traj_idx}, the actions you took and the corresponding observations are:
{past_trajectory}
{completion_status}
'''

REFLECTION_ONLY_TEMPLATE = '''

On trial #{traj_idx}, {completion_status_lower} Your reflection is:
{reflection}'''

import logging as _logging
_prompt_logger = _logging.getLogger(__name__)

def _completion_status(won: bool):
    return "The task is successfully completed." if won else "The task is NOT successfully completed."

def parse_reflection(traj_idx, past_traj, reflection, reflection_type='reflection_only', wons=None):
    if traj_idx == 0 or len(reflection) == 0:
        return '\n'
    else:
        _prompt_logger.info(f'[prompt] reflection_type={reflection_type!r} | traj_idx={traj_idx} | template={"PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE" if reflection_type == "history_and_reflection" else "HISTORY_ONLY_TEMPLATE" if reflection_type == "history_only" else "REFLECTION_ONLY_TEMPLATE"}')
        memories = []
        for _idx in range(traj_idx):
            won = wons[_idx] if wons is not None else False
            status = _completion_status(won)
            if reflection_type == 'history_and_reflection':
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    completion_status=status,
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only':
                memory = HISTORY_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    completion_status=status,
                )
            elif reflection_type == 'reflection_only':
                memory = REFLECTION_ONLY_TEMPLATE.format(
                    traj_idx=_idx + 1,
                    completion_status_lower=status.lower(),
                    reflection=reflection[_idx]
                )
            memories.append(memory)
        return ''.join(memories)


CURR_TRAJ_AT_TRAJ1 = '''
Below are the actions you took and the corresponding observations: 
{current_trajectory}'''

CURR_TRAJ_AT_TRAJ2toN = '''

Currently you're on trial #{traj_idx}, below are the actions you took and the corresponding observations: 
{current_trajectory}'''

TRAJ_2toN_INIT = '''

Currently you're on trial #{traj_idx}, starting from the initial state.'''


def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        else:
            return CURR_TRAJ_AT_TRAJ1.format(
                current_trajectory=curr_traj
            )
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1,
                current_trajectory=curr_traj
            )
        
def get_alfworld_prompt(phase: str = 'play',
                        turn_idx: int = 0,
                        traj_idx: int = 0,
                        init_observation: str = '',
                        curr_traj: str='',
                        past_traj: dict={},
                        admissible_actions: str='',
                        reflection: str='',
                        reflection_type: str='reflection_only',
                        wons: list=None,
                        ):
    assert phase in ['play', 'reflect']
    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, reflection_type, wons=wons)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)

        prompt = ALFWORLD_PLAY_PROMPT.format(
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
            admissible_actions=admissible_actions,
        )

    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        curr_won = wons[traj_idx] if wons is not None else False
        prompt = ALFWORLD_REFLECT_PROMPT.format(
            init_observation=init_observation,
            current_trajectory=current_trajectory,
            completion_status=_completion_status(curr_won),
        )

    return prompt.strip()
