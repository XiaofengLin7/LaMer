"""Action parsing for ALFWorld environments.

Aligned with GEM projection: uses \\boxed{...} action format.
The projection extracts the boxed action from the full LLM response
(stripping thinking/reasoning), so that only the concise action is
stored in memory and included in future prompts.
"""

import re
import copy
from typing import List, Tuple


def _strip_thinking(text: str) -> str:
    """Remove all thinking content from LLM output.

    Handles both complete <think>...</think> blocks and truncated
    responses where </think> is missing (model hit max_response_length).
    """
    # First: strip complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Then: strip truncated <think>... (no closing tag, rest of text is thinking)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _extract_boxed_action(text: str) -> Tuple[str, bool]:
    """Extract the action from \\boxed{...} in LLM output.

    Returns:
        (action_text, valid) where action_text is the raw text forwarded
        to the environment, and valid=True if \\boxed{} was found.
    """
    # Find \boxed{...} in the FULL text first (before stripping thinking)
    pattern = r"\\boxed\{([^}]*)\}"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip().lower(), True

    # No \boxed{} found — strip thinking and return short fallback
    cleaned = _strip_thinking(text)
    if cleaned:
        # Cap at 100 chars to prevent bloat from truncated responses
        return cleaned[-100:].lower(), False
    return "", False


def _extract_reflection(text: str) -> Tuple[str, bool]:
    """Extract reflection from <remark>...</remark> tags.

    Returns:
        (reflection_text, valid) where valid=True if tags were found.
    """
    cleaned = _strip_thinking(text)

    match = re.search(r"<remark>(.*?)</remark>", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:2000], True
    return cleaned[:2000], False


def alfworld_projection(text_actions: List[str], action_pools: List[List[str]] = None, phase: str = 'play'):
    """Parse LLM text output for ALFWorld environments.

    Args:
        text_actions: Raw text outputs from the LLM.
        action_pools: Admissible actions per environment (kept for interface
                      compatibility but not used for extraction).
        phase: 'play' or 'reflect'.

    Returns:
        If phase == 'play':
            (actions, valids) where actions are the extracted
            \\boxed{...} content (lowercased).
        If phase == 'reflect':
            (reflections, valids) where reflections are extracted from
            <remark>...</remark> tags.
    """
    assert phase in ['play', 'reflect']

    if phase == 'reflect':
        reflections = []
        valids = []
        for text in text_actions:
            reflection, valid = _extract_reflection(text)
            reflections.append(reflection)
            valids.append(valid)
        return reflections, valids
    else:
        actions = []
        valids = []
        for text in text_actions:
            action, valid = _extract_boxed_action(text)
            actions.append(action)
            valids.append(valid)
        return actions, valids


if __name__ == '__main__':
    # Quick sanity check
    print(alfworld_projection([r'I think I should \boxed{take mug 1 from countertop 1}'], phase='play'))
    print(alfworld_projection(['<remark>I should have gone to shelf first</remark>'], phase='reflect'))
