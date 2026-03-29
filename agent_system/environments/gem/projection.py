"""Action parsing for GEM environments.

GEM tasks use \\boxed{...} action format. The projection extracts the boxed
action from the full LLM response (stripping thinking/reasoning), so that
only the concise action is stored in memory and included in future prompts.
"""

import re
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


def gem_projection(text_actions: List[str], phase: str = 'play'):
    """Parse LLM text output for GEM environments.

    Args:
        text_actions: Raw text outputs from the LLM.
        phase: 'play' or 'reflect'.

    Returns:
        If phase == 'play':
            (thoughts, actions, valids) where actions are the extracted
            \\boxed{...} content (concise action only, no thinking).
        If phase == 'reflect':
            (reflections, valids) where reflections are extracted from
            <remark>...</remark> tags.
    """
    if phase == 'reflect':
        reflections = []
        valids = []
        for text in text_actions:
            reflection, valid = _extract_reflection(text)
            reflections.append(reflection)
            valids.append(valid)
        return reflections, valids
    else:
        thoughts = []
        actions = []
        valids = []
        for text in text_actions:
            action, valid = _extract_boxed_action(text)
            thoughts.append("")
            actions.append(action)
            valids.append(valid)
        return thoughts, actions, valids


def _extract_boxed_action(text: str) -> Tuple[str, bool]:
    """Extract the action from \\boxed{...} in LLM output.

    Returns:
        (action_text, valid) where action_text is the raw text forwarded
        to the inner adapter, and valid=True if \\boxed{} was found.
    """
    # Find \boxed{...} in the FULL text first (before stripping thinking)
    pattern = r"\\boxed\{([^}]*)\}"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip(), True

    # No \boxed{} found — strip thinking and return short fallback
    cleaned = _strip_thinking(text)
    if cleaned:
        # Cap at 100 chars to prevent bloat from truncated responses
        return cleaned[:100], False
    return "", False


def _extract_reflection(text: str) -> Tuple[str, bool]:
    """Extract reflection from <remark>...</remark> tags.

    Returns:
        (reflection_text, valid) where valid=True if tags were found.
    """
    cleaned = _strip_thinking(text)

    match = re.search(r"<remark>(.*?)</remark>", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:2000], True  # Cap at 2000 chars like minesweeper
    return cleaned[:2000], False
