"""Tests for ALFWorld projection (boxed action format).

Run: python -m agent_system.environments.alfworld.test_projection
"""

import unittest
from .projection import alfworld_projection, _extract_boxed_action, _extract_reflection, _strip_thinking


class TestStripThinking(unittest.TestCase):

    def test_complete_think_block(self):
        text = "<think>some reasoning</think>The answer is \\boxed{go to desk 1}"
        assert _strip_thinking(text) == "The answer is \\boxed{go to desk 1}"

    def test_truncated_think_block(self):
        text = "<think>partial reasoning without closing tag"
        assert _strip_thinking(text) == ""

    def test_multiple_think_blocks(self):
        text = "<think>first</think>middle<think>second</think>end"
        assert _strip_thinking(text) == "middleend"

    def test_no_think_block(self):
        text = "just plain text"
        assert _strip_thinking(text) == "just plain text"

    def test_empty_think_block(self):
        text = "<think></think>result"
        assert _strip_thinking(text) == "result"


class TestExtractBoxedAction(unittest.TestCase):

    def test_simple_boxed(self):
        action, valid = _extract_boxed_action(r"\boxed{take mug 1 from countertop 1}")
        assert action == "take mug 1 from countertop 1"
        assert valid is True

    def test_boxed_with_reasoning(self):
        text = "I need to pick up the mug first.\n\\boxed{take mug 1 from countertop 1}"
        action, valid = _extract_boxed_action(text)
        assert action == "take mug 1 from countertop 1"
        assert valid is True

    def test_boxed_with_thinking(self):
        text = "<think>Let me think about this...</think>\n\\boxed{go to shelf 1}"
        action, valid = _extract_boxed_action(text)
        assert action == "go to shelf 1"
        assert valid is True

    def test_boxed_inside_thinking(self):
        """Boxed action inside <think> block should still be found (searched before stripping)."""
        text = "<think>I'll do \\boxed{open drawer 1}</think>"
        action, valid = _extract_boxed_action(text)
        assert action == "open drawer 1"
        assert valid is True

    def test_multiple_boxed_takes_last(self):
        text = "\\boxed{wrong action}\nActually: \\boxed{go to desk 1}"
        action, valid = _extract_boxed_action(text)
        assert action == "go to desk 1"
        assert valid is True

    def test_boxed_with_whitespace(self):
        action, valid = _extract_boxed_action(r"\boxed{  take lamp 2 from shelf 3  }")
        assert action == "take lamp 2 from shelf 3"
        assert valid is True

    def test_action_is_lowercased(self):
        action, valid = _extract_boxed_action(r"\boxed{Go To Desk 1}")
        assert action == "go to desk 1"
        assert valid is True

    def test_no_boxed_fallback(self):
        text = "I should take the mug"
        action, valid = _extract_boxed_action(text)
        assert valid is False
        assert action == "i should take the mug"

    def test_no_boxed_fallback_strips_thinking(self):
        text = "<think>long reasoning</think>take mug 1"
        action, valid = _extract_boxed_action(text)
        assert valid is False
        assert action == "take mug 1"

    def test_no_boxed_fallback_truncated_thinking(self):
        text = "<think>very long reasoning that got truncated"
        action, valid = _extract_boxed_action(text)
        assert valid is False
        assert action == ""

    def test_no_boxed_fallback_caps_at_100(self):
        text = "x" * 200
        action, valid = _extract_boxed_action(text)
        assert valid is False
        assert len(action) == 100

    def test_empty_boxed(self):
        action, valid = _extract_boxed_action(r"\boxed{}")
        assert action == ""
        assert valid is True

    def test_empty_input(self):
        action, valid = _extract_boxed_action("")
        assert action == ""
        assert valid is False

    def test_boxed_with_trailing_text(self):
        text = "\\boxed{examine book 1}\nDone."
        action, valid = _extract_boxed_action(text)
        assert action == "examine book 1"
        assert valid is True


class TestExtractReflection(unittest.TestCase):

    def test_simple_remark(self):
        text = "<remark>I should go to shelf first</remark>"
        reflection, valid = _extract_reflection(text)
        assert reflection == "I should go to shelf first"
        assert valid is True

    def test_remark_with_reasoning(self):
        text = "Let me think about what went wrong.\n<remark>Next time go to desk first</remark>"
        reflection, valid = _extract_reflection(text)
        assert reflection == "Next time go to desk first"
        assert valid is True

    def test_remark_with_thinking(self):
        text = "<think>analyzing</think>\n<remark>Better plan: go left</remark>"
        reflection, valid = _extract_reflection(text)
        assert reflection == "Better plan: go left"
        assert valid is True

    def test_no_remark_fallback(self):
        text = "I should have gone to the shelf."
        reflection, valid = _extract_reflection(text)
        assert valid is False
        assert reflection == "I should have gone to the shelf."

    def test_no_remark_strips_thinking(self):
        text = "<think>some thinking</think>plain reflection"
        reflection, valid = _extract_reflection(text)
        assert valid is False
        assert reflection == "plain reflection"

    def test_remark_truncated_at_2000(self):
        long_text = "<remark>" + "a" * 3000 + "</remark>"
        reflection, valid = _extract_reflection(long_text)
        assert valid is True
        assert len(reflection) == 2000

    def test_multiline_remark(self):
        text = "<remark>Line 1.\nLine 2.\nLine 3.</remark>"
        reflection, valid = _extract_reflection(text)
        assert valid is True
        assert "Line 1." in reflection
        assert "Line 3." in reflection

    def test_empty_remark(self):
        text = "<remark></remark>"
        reflection, valid = _extract_reflection(text)
        assert valid is True
        assert reflection == ""


class TestAlfworldProjectionPlay(unittest.TestCase):

    def test_single_action(self):
        actions, valids = alfworld_projection([r"\boxed{take mug 1 from countertop 1}"], phase='play')
        assert actions == ["take mug 1 from countertop 1"]
        assert valids == [True]

    def test_multiple_actions(self):
        texts = [
            r"Reasoning here. \boxed{go to shelf 1}",
            r"\boxed{take book 1 from desk 1}",
            r"No boxed action here",
        ]
        actions, valids = alfworld_projection(texts, phase='play')
        assert actions[0] == "go to shelf 1"
        assert valids[0] is True
        assert actions[1] == "take book 1 from desk 1"
        assert valids[1] is True
        assert valids[2] is False

    def test_action_pools_ignored(self):
        """action_pools param is kept for interface compat but not used."""
        actions, valids = alfworld_projection(
            [r"\boxed{go to desk 1}"],
            action_pools=[["go to desk 1", "go to shelf 1"]],
            phase='play',
        )
        assert actions == ["go to desk 1"]
        assert valids == [True]

    def test_actions_lowercased(self):
        actions, valids = alfworld_projection([r"\boxed{Go To DESK 1}"], phase='play')
        assert actions == ["go to desk 1"]


class TestAlfworldProjectionReflect(unittest.TestCase):

    def test_single_reflection(self):
        reflections, valids = alfworld_projection(
            ["<remark>I should have gone to shelf first</remark>"],
            phase='reflect',
        )
        assert reflections == ["I should have gone to shelf first"]
        assert valids == [True]

    def test_invalid_reflection(self):
        reflections, valids = alfworld_projection(
            ["Just some text without remark tags"],
            phase='reflect',
        )
        assert valids == [False]
        assert reflections == ["Just some text without remark tags"]

    def test_multiple_reflections(self):
        texts = [
            "<remark>Plan A</remark>",
            "no tags here",
            "<think>hmm</think><remark>Plan B</remark>",
        ]
        reflections, valids = alfworld_projection(texts, phase='reflect')
        assert reflections[0] == "Plan A"
        assert valids[0] is True
        assert valids[1] is False
        assert reflections[2] == "Plan B"
        assert valids[2] is True


if __name__ == '__main__':
    unittest.main()
