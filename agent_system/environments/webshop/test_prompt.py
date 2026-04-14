"""Tests for WebShop prompt rendering and projection.

These tests only exercise prompt.py and projection.py — no gym/Flask/webshop
server is required. They run with any Python that has the standard library.

Run (from the LaMer repo root):
    python -m agent_system.environments.webshop.test_prompt

Or directly:
    python agent_system/environments/webshop/test_prompt.py
"""

import sys
import os
import unittest
import importlib.util

# Import prompt.py and projection.py directly so that the package __init__.py
# (which transitively imports gym/Flask) is never triggered.
_HERE = os.path.dirname(os.path.abspath(__file__))

def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_ws_{name}", os.path.join(_HERE, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_prompt_mod = _load("prompt")
_proj_mod   = _load("projection")

get_webshop_prompt       = _prompt_mod.get_webshop_prompt
get_webshop_prompt_short = _prompt_mod.get_webshop_prompt_short
webshop_projection       = _proj_mod.webshop_projection
_extract_boxed_action    = _proj_mod._extract_boxed_action
_extract_reflection      = _proj_mod._extract_reflection
_strip_thinking          = _proj_mod._strip_thinking


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestWebshopPromptRendering(unittest.TestCase):
    """Core regression: .format() must not raise KeyError: 'your action'."""

    BASE_KWARGS = dict(
        phase='play',
        turn_idx=0,
        traj_idx=0,
        task_description='find a red shirt under $30',
        curr_traj='',
        past_traj={},
        admissible_actions='search[red shirt], click[back to search]',
        reflection='',
        reflection_type='reflection_only',
    )

    def test_no_key_error_on_play(self):
        """Regression: \\boxed{your action} must not be treated as a format key."""
        prompt = get_webshop_prompt(**self.BASE_KWARGS)
        self.assertIsInstance(prompt, str)

    def test_boxed_literal_in_output(self):
        """The rendered prompt must contain the literal string \\boxed{your action}."""
        prompt = get_webshop_prompt(**self.BASE_KWARGS)
        self.assertIn(r'\boxed{your action}', prompt)

    def test_task_description_interpolated(self):
        prompt = get_webshop_prompt(**self.BASE_KWARGS)
        self.assertIn('find a red shirt under $30', prompt)

    def test_admissible_actions_interpolated(self):
        prompt = get_webshop_prompt(**self.BASE_KWARGS)
        self.assertIn('search[red shirt], click[back to search]', prompt)

    def test_reflect_phase_no_key_error(self):
        kwargs = dict(self.BASE_KWARGS)
        kwargs['phase'] = 'reflect'
        kwargs['curr_traj'] = 'Obs: ...  Action: search[shirt]'
        kwargs['turn_idx'] = 3
        prompt = get_webshop_prompt(**kwargs)
        self.assertIsInstance(prompt, str)
        self.assertIn('find a red shirt under $30', prompt)

    def test_multi_traj_play(self):
        """Prompt with traj_idx > 0 should include reflection preamble."""
        kwargs = dict(self.BASE_KWARGS)
        kwargs['traj_idx'] = 1
        kwargs['turn_idx'] = 2
        kwargs['curr_traj'] = 'Obs: search results  Action: click[item 1]'
        kwargs['past_traj'] = {0: 'Obs: search  Action: search[shirt]'}
        kwargs['reflection'] = {0: 'I should have filtered by price.'}
        prompt = get_webshop_prompt(**kwargs)
        self.assertIsInstance(prompt, str)
        self.assertIn('I should have filtered by price.', prompt)

    def test_short_prompt_no_key_error(self):
        """get_webshop_prompt_short also uses WEBSHOP_PLAY_PROMPT.format()."""
        prompt = get_webshop_prompt_short(**self.BASE_KWARGS)
        self.assertIsInstance(prompt, str)
        self.assertIn(r'\boxed{your action}', prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Projection helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStripThinking(unittest.TestCase):

    def test_complete_think_block(self):
        text = "<think>reasoning</think>The answer is \\boxed{search[shirt]}"
        self.assertEqual(_strip_thinking(text), "The answer is \\boxed{search[shirt]}")

    def test_truncated_think_block(self):
        text = "<think>partial reasoning without closing tag"
        self.assertEqual(_strip_thinking(text), "")

    def test_no_think_block(self):
        text = "plain response"
        self.assertEqual(_strip_thinking(text), "plain response")

    def test_multiple_think_blocks(self):
        text = "<think>first</think>middle<think>second</think>end"
        self.assertEqual(_strip_thinking(text), "middleend")


class TestExtractBoxedAction(unittest.TestCase):

    def test_simple_boxed(self):
        action, valid = _extract_boxed_action(r"\boxed{search[red shirt]}")
        self.assertEqual(action, "search[red shirt]")
        self.assertTrue(valid)

    def test_boxed_with_reasoning(self):
        text = "I should search first.\n\\boxed{search[red shirt under 30]}"
        action, valid = _extract_boxed_action(text)
        self.assertEqual(action, "search[red shirt under 30]")
        self.assertTrue(valid)

    def test_boxed_with_thinking(self):
        text = "<think>Let me decide</think>\n\\boxed{click[b076hg3h3h]}"
        action, valid = _extract_boxed_action(text)
        self.assertEqual(action, "click[b076hg3h3h]")
        self.assertTrue(valid)

    def test_multiple_boxed_takes_last(self):
        text = "\\boxed{search[shirt]}\nActually: \\boxed{click[item 2]}"
        action, valid = _extract_boxed_action(text)
        self.assertEqual(action, "click[item 2]")
        self.assertTrue(valid)

    def test_action_is_lowercased(self):
        action, valid = _extract_boxed_action(r"\boxed{Click[Back To Search]}")
        self.assertEqual(action, "click[back to search]")
        self.assertTrue(valid)

    def test_no_boxed_fallback(self):
        text = "click the first item"
        action, valid = _extract_boxed_action(text)
        self.assertFalse(valid)
        self.assertEqual(action, "click the first item")

    def test_no_boxed_fallback_caps_at_100(self):
        text = "x" * 200
        action, valid = _extract_boxed_action(text)
        self.assertFalse(valid)
        self.assertEqual(len(action), 100)

    def test_empty_input(self):
        action, valid = _extract_boxed_action("")
        self.assertFalse(valid)
        self.assertEqual(action, "")

    def test_empty_boxed(self):
        action, valid = _extract_boxed_action(r"\boxed{}")
        self.assertTrue(valid)
        self.assertEqual(action, "")

    def test_boxed_with_whitespace(self):
        action, valid = _extract_boxed_action(r"\boxed{  search[shirt]  }")
        self.assertEqual(action, "search[shirt]")
        self.assertTrue(valid)


class TestExtractReflection(unittest.TestCase):

    def test_simple_remark(self):
        text = "<remark>I should have filtered by price first.</remark>"
        reflection, valid = _extract_reflection(text)
        self.assertEqual(reflection, "I should have filtered by price first.")
        self.assertTrue(valid)

    def test_remark_with_reasoning(self):
        text = "Analysis: went wrong at step 2.\n<remark>Next time filter by price.</remark>"
        reflection, valid = _extract_reflection(text)
        self.assertEqual(reflection, "Next time filter by price.")
        self.assertTrue(valid)

    def test_no_remark_fallback(self):
        text = "I should have filtered by size."
        reflection, valid = _extract_reflection(text)
        self.assertFalse(valid)
        self.assertEqual(reflection, "I should have filtered by size.")

    def test_remark_truncated_at_2000(self):
        long_text = "<remark>" + "a" * 3000 + "</remark>"
        reflection, valid = _extract_reflection(long_text)
        self.assertTrue(valid)
        self.assertEqual(len(reflection), 2000)

    def test_multiline_remark(self):
        text = "<remark>Line 1.\nLine 2.\nLine 3.</remark>"
        reflection, valid = _extract_reflection(text)
        self.assertTrue(valid)
        self.assertIn("Line 1.", reflection)
        self.assertIn("Line 3.", reflection)


# ─────────────────────────────────────────────────────────────────────────────
# webshop_projection end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestWebshopProjectionPlay(unittest.TestCase):

    def test_single_action(self):
        actions, valids = webshop_projection([r"\boxed{search[red shirt]}"], phase='play')
        self.assertEqual(actions, ["search[red shirt]"])
        self.assertEqual(valids, [True])

    def test_multiple_actions(self):
        texts = [
            r"Reasoning. \boxed{click[item 1]}",
            r"\boxed{search[blue jeans]}",
            "no boxed here",
        ]
        actions, valids = webshop_projection(texts, phase='play')
        self.assertEqual(actions[0], "click[item 1]")
        self.assertTrue(valids[0])
        self.assertEqual(actions[1], "search[blue jeans]")
        self.assertTrue(valids[1])
        self.assertFalse(valids[2])

    def test_actions_lowercased(self):
        actions, valids = webshop_projection([r"\boxed{Click[BACK TO SEARCH]}"], phase='play')
        self.assertEqual(actions, ["click[back to search]"])


class TestWebshopProjectionReflect(unittest.TestCase):

    def test_single_reflection(self):
        reflections, valids = webshop_projection(
            ["<remark>I should have sorted by price.</remark>"],
            phase='reflect',
        )
        self.assertEqual(reflections, ["I should have sorted by price."])
        self.assertEqual(valids, [True])

    def test_invalid_reflection(self):
        reflections, valids = webshop_projection(
            ["Just plain text without tags"],
            phase='reflect',
        )
        self.assertEqual(valids, [False])
        self.assertEqual(reflections, ["Just plain text without tags"])

    def test_multiple_reflections(self):
        texts = [
            "<remark>Plan A</remark>",
            "no tags",
            "<think>hmm</think><remark>Plan B</remark>",
        ]
        reflections, valids = webshop_projection(texts, phase='reflect')
        self.assertEqual(reflections[0], "Plan A")
        self.assertTrue(valids[0])
        self.assertFalse(valids[1])
        self.assertEqual(reflections[2], "Plan B")
        self.assertTrue(valids[2])


if __name__ == '__main__':
    unittest.main()
