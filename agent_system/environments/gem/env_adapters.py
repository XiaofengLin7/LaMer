"""Self-contained environment adapters for GEM multi-task training.

All adapters are standalone — no imports from explorer or rLLM.
Each adapter implements:
    get_rules() -> str              (static game rules, symbols, goal, action format)
    reset(seed, task) -> (obs_str, info_dict)   (dynamic game state only, no rules)
    step(action_str) -> (obs_str, reward, done, info_dict)
    close()
"""

from __future__ import annotations

import re
import random
import hashlib
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Utility: extract \boxed{...} content from LLM output
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} in text."""
    # Find all \boxed{...} occurrences (handling nested braces)
    pattern = r"\\boxed\{([^}]*)\}"
    matches = list(re.finditer(pattern, str(text), re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return None


# ===========================================================================
# GEM Environment Adapter (wraps gem.make())
# ===========================================================================

# Auto-register custom minesweeper on first import
_gem_registered = False

def _ensure_gem_registered():
    global _gem_registered
    if _gem_registered:
        return
    try:
        import gem  # noqa
        import gem.envs  # noqa: F401
        from gem.envs.registration import register
        from gem.envs.game_env.minesweeper import MinesweeperEnv as BaseMinesweeperEnv

        class OnlyRevealMinesweeperEnv(BaseMinesweeperEnv):
            """Minesweeper variant that only requires revealing all non-mine cells."""
            def _is_solved(self) -> bool:
                return all(
                    self.grid[r][c] == -1 or self.revealed[r][c]
                    for r in range(self.rows)
                    for c in range(self.cols)
                )

        # Store for reference
        globals()['OnlyRevealMinesweeperEnv'] = OnlyRevealMinesweeperEnv

        # Register with GEM
        # Use a try/except in case it's already registered
        try:
            register(
                "game:Minesweeper-v0-only-reveal",
                f"{__name__}:OnlyRevealMinesweeperEnv",
                rows=5, cols=5, num_mines=5, max_turns=25,
            )
        except Exception:
            pass  # Already registered

        _gem_registered = True
    except ImportError:
        pass  # gem not available; GEMEnvAdapter will fail on use


class GEMEnvAdapter:
    """Adapter wrapping a GEM environment via gem.make().

    Separates static rules (prefix/suffix from first reset) from dynamic observations.
    """

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        _ensure_gem_registered()
        import gem
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        self._env = gem.make(env_id, **self.env_kwargs)
        self._seed = self.env_kwargs.get("seed")
        self._rules: Optional[str] = None

    def get_rules(self) -> str:
        """Return static game rules extracted from the gem environment.

        For gem games, the observation from reset() contains the rules text,
        while suffix contains the dynamic game state.
        """
        if self._rules is not None:
            return self._rules
        return ""

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[Any, dict]:
        reset_seed = seed if seed is not None else self._seed
        self._seed = reset_seed
        try:
            observation, info = self._env.reset(seed=reset_seed)
        except TypeError:
            observation, info = self._env.reset(task=task)
        # In gem: observation = rules text (static), suffix = game state (dynamic)
        suffix = info.get("suffix", "")
        if self._rules is None:
            self._rules = observation.strip()
        # Return the suffix as the dynamic game state;
        # if no suffix, return a minimal state indicator
        game_state = suffix.strip() if suffix.strip() else "Game started. Make your first move."
        return game_state, info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        observation, reward, terminated, truncated, info = self._env.step(str(action))
        done = bool(terminated or truncated)
        enriched_info = dict(info or {})
        enriched_info.update({
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "raw_reward": float(reward),
            "max_turns": getattr(self._env, "max_turns", None),
        })
        # observation = action feedback, suffix = updated game state
        suffix = enriched_info.get("suffix", "")
        dynamic_obs = observation
        if suffix.strip():
            dynamic_obs = observation + "\n" + suffix.strip()
        return dynamic_obs, float(reward), done, enriched_info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()


# ===========================================================================
# Rock-Paper-Scissors Adapter
# ===========================================================================

class _PolarizedAdversaryDistribution:
    """Hidden adversary distribution with a dominant action."""
    LABELS = ("rock", "paper", "scissors")

    def __init__(self, *, min_dom: float):
        if not (0.0 < min_dom < 1.0):
            raise ValueError("min_dom must be in (0, 1)")
        self.min_dom = float(min_dom)
        self._shape = 2.0

    def sample(self, seed: int) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        dom_idx = int(rng.integers(0, 3))
        t = float(rng.beta(self._shape, self._shape))
        dom_prob = self.min_dom + (1.0 - self.min_dom) * t
        remainder = 1.0 - dom_prob
        rest = rng.dirichlet([1.0, 1.0]) * remainder
        probs = np.zeros(3, dtype=float)
        others = [i for i in (0, 1, 2) if i != dom_idx]
        probs[dom_idx] = dom_prob
        probs[others[0]] = float(rest[0])
        probs[others[1]] = float(rest[1])
        return {label: float(p) for label, p in zip(self.LABELS, probs)}


class RockPaperScissorsEnvAdapter:
    """Multi-turn Rock-Paper-Scissors with a hidden polarized adversary."""
    ACTIONS = {"rock", "paper", "scissors"}

    _RULES = """# Game Description
You are playing a multi-turn Rock-Paper-Scissors game against an adversary.

# Actions and Outcomes
- Available actions: rock, paper, scissors.
- Rock beats scissors, scissors beats paper, paper beats rock.
- Equal actions result in a draw.

# Your Goal
Maximize your win rate over multiple turns. The adversary's action each turn is sampled from a fixed hidden distribution. Observe outcomes to infer the distribution and exploit it.

# Action Format
Output your action within \\boxed{...}.
Example: \\boxed{rock}"""

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        self.env_id = env_id
        if env_kwargs is None:
            raise ValueError("env_kwargs must include: max_turns, min_dom")
        self.max_turns = env_kwargs.get("max_turns", None)
        self.min_dom = env_kwargs.get("min_dom", None)
        if not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("env_kwargs['max_turns'] must be a positive int")
        if not isinstance(self.min_dom, (int, float)) or not (0.0 < float(self.min_dom) < 1.0):
            raise ValueError("env_kwargs['min_dom'] must be a float in (0, 1)")
        self._dist_gen = _PolarizedAdversaryDistribution(min_dom=float(self.min_dom))
        self.turn: int = 0
        self.terminated: bool = False
        self._probs: Optional[Dict[str, float]] = None
        self._rng_turn: Optional[np.random.Generator] = None
        self._wins: int = 0
        self._losses: int = 0
        self._draws: int = 0

    def get_rules(self) -> str:
        """Return static game rules."""
        return self._RULES

    def _parse_action(self, raw_text: str) -> str:
        boxed = extract_boxed_answer(raw_text)
        return (boxed if boxed else str(raw_text)).strip().lower()

    @staticmethod
    def _outcome(agent: str, adv: str) -> str:
        if agent not in {"rock", "paper", "scissors"}:
            return "lose"
        if agent == adv:
            return "draw"
        if (agent, adv) in {("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")}:
            return "win"
        return "lose"

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[str, Dict[str, Any]]:
        if seed is None:
            raise ValueError("Seed must be provided.")
        self.turn = 0
        self.terminated = False
        self._wins = self._losses = self._draws = 0
        self._probs = self._dist_gen.sample(seed)
        self._rng_turn = np.random.default_rng()

        initial = f"Game started. You have {self.max_turns} turns. Make your first move."
        info = {
            "turn": self.turn, "max_turns": self.max_turns,
            "terminated": False, "truncated": False,
        }
        return initial, info

    def step(self, action: Any) -> tuple[str, float, bool, Dict[str, Any]]:
        if self._probs is None or self._rng_turn is None:
            raise ValueError("Call reset() first.")
        labels = np.array(["rock", "paper", "scissors"])
        p = np.array([self._probs["rock"], self._probs["paper"], self._probs["scissors"]], dtype=float)
        p = p / p.sum()
        adv_action = str(self._rng_turn.choice(labels, p=p))

        agent_action = self._parse_action(str(action))
        outcome = self._outcome(agent_action, adv_action)
        if outcome == "win":
            self._wins += 1
        elif outcome == "lose":
            self._losses += 1
        else:
            self._draws += 1

        self.turn += 1
        is_last = self.turn >= self.max_turns
        if is_last:
            self.terminated = True

        reward = 0.0
        if is_last:
            win_rate = self._wins / float(self.max_turns)
            reward = 1.0 if win_rate >= 0.5 else 0.0

        obs = f"You {outcome}" + (". Now play the next episode." if is_last else "")
        info = {
            "turn": self.turn, "max_turns": self.max_turns,
            "terminated": is_last, "truncated": False,
            "raw_reward": float(reward),
        }
        return obs, float(reward), is_last, info

    def close(self) -> None:
        pass


# ===========================================================================
# Blackjack Adapter
# ===========================================================================

Card = int

def _hand_value(hand: List[Card]) -> Tuple[int, bool]:
    total = sum(hand)
    usable_ace = 1 in hand and total + 10 <= 21
    if usable_ace:
        total += 10
    return total, usable_ace


class BlackjackEnv:
    """Minimal Blackjack environment."""
    def __init__(self, max_turns: int = 30, seed: Optional[int] = None):
        self.max_turns = max_turns
        self._rng = random.Random(seed)
        self._seed = seed
        self.player: List[Card] = []
        self.dealer: List[Card] = []
        self.deck: List[Card] = []
        self._taken_indices: set[int] = set()
        self._available_indices: List[int] = []
        self.turn: int = 0
        self._done: bool = False

    def reset(self, seed: Optional[int] = None, task: Optional[dict] = None) -> tuple[Dict[str, Any], dict]:
        if seed is None and task is not None:
            seed = task.get("seed")
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            self._rng.seed(self._seed)

        max_attempts = 50
        attempts = 0
        while True:
            self.deck = self._build_deck()
            self.dealer = [self.deck.pop(0), self.deck.pop(0)]
            dealer_total, _ = _hand_value(self.dealer)
            if dealer_total >= 17 or attempts >= max_attempts:
                break
            attempts += 1
        self.player = [self.deck.pop(0), self.deck.pop(0)]
        self._taken_indices = set()
        self._available_indices = list(range(len(self.deck)))
        self.turn = 0
        self._done = False

        player_total, _ = _hand_value(self.player)
        dealer_total, _ = _hand_value(self.dealer)
        terminated = False
        reward = 0.0
        observation = self._format_obs(reveal_dealer=terminated)
        info = self._build_info(terminated=terminated, truncated=False, reward=reward,
                                natural=False, player_total=player_total,
                                dealer_total=dealer_total if terminated else None)
        self._done = terminated
        return observation, info

    def step(self, action: Any) -> tuple[Dict[str, Any], float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset().")
        normalized_action = self._normalize_action(action)
        if normalized_action.get("action") == "hit":
            idx = normalized_action.get("card_index")
            if idx is None or not self._available_indices or idx not in self._available_indices:
                normalized_action = {"action": "stand"}

        self.turn += 1
        truncated = self.turn >= self.max_turns
        card = None

        if normalized_action["action"] == "hit":
            card = self._take_card(normalized_action["card_index"])
            self.player.append(card)
            player_total, _ = _hand_value(self.player)
            terminated = player_total > 21
            reward = 0.0
        else:
            player_total, _ = _hand_value(self.player)
            terminated = True
            reward = 0.0
            dealer_total, _ = _hand_value(self.dealer)
            if player_total <= 21 and dealer_total <= 21 and player_total > dealer_total:
                reward = 1.0

        done = terminated or truncated
        self._done = done
        observation = self._format_obs(reveal_dealer=done)
        info = self._build_info(terminated=terminated, truncated=truncated, reward=reward,
                                natural=False, player_total=_hand_value(self.player)[0],
                                dealer_total=_hand_value(self.dealer)[0] if done else None,
                                last_drawn_card=card)
        info["last_action"] = normalized_action
        return observation, reward, terminated, truncated, info

    def _normalize_action(self, action: Any) -> Dict[str, Union[str, int]]:
        if isinstance(action, str):
            extracted = extract_boxed_answer(action)
            if extracted is not None:
                action = extracted.strip()
            else:
                action = action.strip()
        if action is None:
            return {"action": "stand"}
        if isinstance(action, str):
            cleaned = action.strip().strip(".!").lower()
            if not cleaned:
                return {"action": "stand"}
            parts = cleaned.split()
            if parts[0] in {"stand", "s", "stick", "stay"}:
                return {"action": "stand"}
            if parts[0] in {"hit", "h"}:
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        return {"action": "hit", "card_index": idx}
                    except ValueError:
                        pass
                if self._available_indices:
                    return {"action": "hit", "card_index": self._available_indices[0]}
                return {"action": "stand"}
        if isinstance(action, dict):
            act = str(action.get("action", "")).lower()
            if act in {"stand", "s", "stick", "stay"}:
                return {"action": "stand"}
            if act in {"hit", "h"}:
                if "card_index" in action:
                    try:
                        return {"action": "hit", "card_index": int(action["card_index"])}
                    except Exception:
                        return {"action": "stand"}
                if self._available_indices:
                    return {"action": "hit", "card_index": self._available_indices[0]}
                return {"action": "stand"}
        return {"action": "stand"}

    def _format_obs(self, reveal_dealer: bool) -> Dict[str, Any]:
        player_total, player_soft = _hand_value(self.player)
        dealer_total, dealer_soft = _hand_value(self.dealer)
        deck_view = [
            self._card_to_str(self.deck[idx]) if idx in self._taken_indices else "?"
            for idx in range(len(self.deck))
        ]
        return {
            "player_hand": list(self.player),
            "dealer_upcard": self.dealer[0],
            "dealer_hand": list(self.dealer) if reveal_dealer else [self.dealer[0], None],
            "player_total": player_total,
            "dealer_total": dealer_total if reveal_dealer else None,
            "player_soft": player_soft,
            "dealer_soft": dealer_soft if reveal_dealer else None,
            "turn": self.turn,
            "legal_actions": ["hit <card_index>", "stand"],
            "deck": deck_view,
            "available_indices": list(self._available_indices),
        }

    def _build_info(self, terminated, truncated, reward, natural, player_total, dealer_total, last_drawn_card=None):
        return {
            "terminated": bool(terminated), "truncated": bool(truncated),
            "raw_reward": float(reward), "natural": bool(natural),
            "player_total": player_total, "dealer_total": dealer_total,
            "max_turns": self.max_turns,
            "available_indices": list(self._available_indices),
            "dealer_hand": list(self.dealer), "player_hand": list(self.player),
            "last_drawn_card": last_drawn_card,
        }

    def _build_deck(self) -> List[Card]:
        deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self._rng.shuffle(deck)
        return deck

    @staticmethod
    def _card_to_str(card: Card) -> str:
        if card == 1:
            return "A"
        return str(card)

    def _take_card(self, explicit_index: Optional[int] = None) -> Card:
        if not self._available_indices:
            raise RuntimeError("No cards available.")
        idx = explicit_index if explicit_index is not None else self._available_indices[0]
        if idx < 0 or idx >= len(self.deck):
            raise ValueError(f"Card index {idx} out of range.")
        if idx in self._taken_indices:
            raise ValueError(f"Card index {idx} already taken.")
        card = self.deck[idx]
        self._taken_indices.add(idx)
        self._available_indices = [i for i in range(len(self.deck)) if i not in self._taken_indices]
        return card


class BlackjackEnvAdapter:
    """Adapter exposing BlackjackEnv."""

    _RULES = """# Game Description
You are playing a simplified game of Blackjack against a dealer.

# Card Values
- Number cards (2-10): face value
- Face cards (J, Q, K): value 10
- Ace (A): value 1 or 11, chosen to give the highest possible hand value not exceeding 21

# Your Goal
Get a hand value as close to 21 as possible without exceeding 21.
- If your hand value exceeds 21, you bust and immediately lose.
- After you stand, the game ends and your hand is compared with the dealer's hand.

# Dealer Rules
- The dealer has exactly two cards (one visible, one hidden).
- The dealer does not draw additional cards.

# Action Format
Available actions: 'hit <card_index>' or 'stand'.
Output your action within \\boxed{...}.
- Hit example: \\boxed{hit 9}
- Stand example: \\boxed{stand}"""

    def __init__(self, env_id: Optional[str] = None, env_kwargs: Optional[dict] = None,
                 max_turns: int = 30, seed: Optional[int] = None, **_: Any):
        self.max_turns = max_turns
        self._seed = seed
        self._env = BlackjackEnv(max_turns=max_turns, seed=seed)

    def get_rules(self) -> str:
        """Return static game rules."""
        return self._RULES

    def reset(self, seed: Optional[int] = None, task: Optional[dict] = None) -> tuple[Any, dict]:
        reset_seed = seed if seed is not None else self._seed
        self._seed = reset_seed
        obs, info = self._env.reset(seed=reset_seed, task=task)
        return self._render_observation(obs), info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        enriched_info = dict(info)
        enriched_info.update({
            "terminated": bool(terminated), "truncated": bool(truncated),
            "raw_reward": float(reward), "max_turns": self.max_turns,
        })

        last_action = info.get("last_action", {})
        action_name = last_action.get("action") if isinstance(last_action, dict) else None
        action_idx = last_action.get("card_index") if isinstance(last_action, dict) else None
        drawn_card = info.get("last_drawn_card")
        drawn_card_str = self._card_to_str(drawn_card) if drawn_card is not None else None
        player_total_info = info.get("player_total")
        busted = bool(info.get("terminated")) and player_total_info is not None and player_total_info > 21

        if action_name == "hit":
            action_summary = f"You chose to hit{f' {action_idx}' if action_idx is not None else ''}"
            if drawn_card_str is not None:
                action_summary += f" and drew {drawn_card_str}"
            action_summary += "."
        elif action_name:
            action_summary = f"You chose to {action_name}."
        else:
            action_summary = "You chose an action."

        if done:
            dealer_hand = info.get("dealer_hand", [])
            player_hand = info.get("player_hand", [])
            dealer_str = ", ".join(self._card_to_str(c) for c in dealer_hand)
            player_str = ", ".join(self._card_to_str(c) for c in player_hand)
            outcome = "win" if reward > 0 else "lose or tie"
            if busted:
                obs_text = (f"{action_summary} Dealer cards: {dealer_str}. "
                            f"Your cards: {player_str}. You lose because you bust.")
            else:
                obs_text = (f"{action_summary} Dealer cards: {dealer_str}. "
                            f"Your cards: {player_str}. You {outcome}.")
        else:
            obs_text = f"{action_summary}\n{self._render_observation(obs)}"
        return obs_text, float(reward), done, enriched_info

    def close(self) -> None:
        pass

    def _render_observation(self, obs: Dict[str, Any]) -> str:
        player_cards = ", ".join(self._card_to_str(c) for c in obs["player_hand"])
        dealer_cards = (
            ", ".join(self._card_to_str(c) for c in obs["dealer_hand"])
            if obs["dealer_total"] is not None
            else f"{self._card_to_str(obs['dealer_upcard'])}, ?"
        )
        deck_display = " ".join(f"{i}:{c}" for i, c in enumerate(obs["deck"]))
        lines = [
            f"Dealer: {dealer_cards}",
            f"Your hand ({obs['player_total']}): {player_cards}",
            f"Deck (index:value, ?=hidden, drawn=revealed): {deck_display}",
            f"Available indices: {obs['available_indices']}",
            "Actions: 'stand' or 'hit <card_index>' (choose an available index)",
        ]
        return "\n".join(lines)

    @staticmethod
    def _card_to_str(card: Card) -> str:
        if card == 1:
            return "A"
        return str(card)


# ===========================================================================
# Maze Adapter
# ===========================================================================

class MazeGenerator:
    """Standalone maze generator."""
    def __init__(self, shapes: List[Tuple[int, int]]):
        if not isinstance(shapes, list) or len(shapes) == 0:
            raise ValueError("shapes must be a non-empty list of (width, height) tuples")
        self.shapes = shapes

    def generate(self, seed: int) -> Tuple[np.ndarray, Tuple[int, int], int]:
        random.seed(seed)
        np.random.seed(seed)
        maze_width, maze_height = random.choice(self.shapes)
        maze = np.ones((maze_height, maze_width), dtype=int)
        start_x = random.randint(1, maze_height - 2)
        start_y = random.randint(1, maze_width - 2)
        init_position = (start_x, start_y)
        self._create_branched_network(maze, start_x, start_y)
        self._add_branch_extensions(maze, maze_height, maze_width)
        self._create_strategic_loops(maze, maze_height, maze_width, seed)
        self._ensure_connectivity(maze, init_position)
        goal_position, shortest_path_len = self._find_farthest_position(maze, init_position)
        maze[goal_position[0], goal_position[1]] = -1
        return maze, init_position, shortest_path_len

    def _create_branched_network(self, maze, start_x, start_y):
        maze[start_x, start_y] = 0
        active_fronts = [(start_x, start_y)]
        visited = {(start_x, start_y)}
        while active_fronts:
            front_idx = random.randint(0, len(active_fronts) - 1)
            current_x, current_y = active_fronts[front_idx]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)
            expanded = False
            branches_created = 0
            max_branches_per_node = 1 if random.random() < 0.7 else 2
            for dx, dy in directions:
                if branches_created >= max_branches_per_node:
                    break
                new_x, new_y = current_x + dx, current_y + dy
                if (1 <= new_x < maze.shape[0] - 1 and 1 <= new_y < maze.shape[1] - 1
                        and (new_x, new_y) not in visited):
                    if self._count_open_neighbors(maze, new_x, new_y) <= 1:
                        maze[new_x, new_y] = 0
                        visited.add((new_x, new_y))
                        active_fronts.append((new_x, new_y))
                        expanded = True
                        branches_created += 1
            if not expanded:
                active_fronts.pop(front_idx)

    def _add_branch_extensions(self, maze, height, width):
        dead_ends = []
        for x in range(1, height - 1):
            for y in range(1, width - 1):
                if maze[x, y] == 0 and self._count_open_neighbors(maze, x, y) == 1:
                    dead_ends.append((x, y))
        num_extensions = min(len(dead_ends), max(2, len(dead_ends) // 3))
        random.shuffle(dead_ends)
        for i in range(num_extensions):
            x, y = dead_ends[i]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (1 <= new_x < height - 1 and 1 <= new_y < width - 1
                        and maze[new_x, new_y] == 1
                        and self._count_open_neighbors(maze, new_x, new_y) <= 1):
                    maze[new_x, new_y] = 0
                    if random.random() < 0.6:
                        self._extend_branch_randomly(maze, new_x, new_y, height, width,
                                                     max_length=random.randint(2, 5))
                    break

    def _extend_branch_randomly(self, maze, start_x, start_y, height, width, max_length):
        current_x, current_y = start_x, start_y
        for _ in range(max_length):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)
            extended = False
            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                if (1 <= new_x < height - 1 and 1 <= new_y < width - 1
                        and maze[new_x, new_y] == 1
                        and self._count_open_neighbors(maze, new_x, new_y) <= 1):
                    maze[new_x, new_y] = 0
                    current_x, current_y = new_x, new_y
                    extended = True
                    break
            if not extended:
                break

    def _create_strategic_loops(self, maze, height, width, seed):
        random.seed(seed + 42)
        num_loops = random.randint(2, max(3, (height * width) // 30))
        for _ in range(num_loops):
            for _ in range(100):
                x = random.randint(1, height - 2)
                y = random.randint(1, width - 2)
                if maze[x, y] == 1:
                    path_neighbors = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and maze[nx, ny] == 0:
                            path_neighbors.append((nx, ny))
                    if len(path_neighbors) == 2:
                        maze[x, y] = 0
                        break

    def _count_open_neighbors(self, maze, x, y):
        count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                count += 1
        return count

    def _ensure_connectivity(self, maze, start_pos):
        reachable = self._get_reachable_positions(maze, start_pos)
        all_paths = set()
        for x in range(1, maze.shape[0] - 1):
            for y in range(1, maze.shape[1] - 1):
                if maze[x, y] == 0:
                    all_paths.add((x, y))
        unreachable = all_paths - reachable
        for pos in unreachable:
            self._connect_to_reachable(maze, pos, reachable)

    def _get_reachable_positions(self, maze, start):
        queue = deque([start])
        visited = {start}
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1]
                        and maze[new_x, new_y] == 0 and (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))
        return visited

    def _connect_to_reachable(self, maze, pos, reachable):
        x, y = pos
        min_distance = float("inf")
        best_connection = None
        for rx, ry in reachable:
            distance = abs(x - rx) + abs(y - ry)
            if distance < min_distance:
                min_distance = distance
                best_connection = (rx, ry)
        if best_connection is None:
            return
        cx, cy = best_connection
        current_x, current_y = x, y
        while current_x != cx:
            current_x += 1 if current_x < cx else -1
            if 0 <= current_x < maze.shape[0] and 0 <= current_y < maze.shape[1]:
                maze[current_x, current_y] = 0
        while current_y != cy:
            current_y += 1 if current_y < cy else -1
            if 0 <= current_x < maze.shape[0] and 0 <= current_y < maze.shape[1]:
                maze[current_x, current_y] = 0

    def _find_farthest_position(self, maze, start):
        queue = deque([(start, 0)])
        visited = {start}
        farthest_pos = start
        max_distance = 0
        while queue:
            (x, y), distance = queue.popleft()
            if distance > max_distance:
                max_distance = distance
                farthest_pos = (x, y)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1]
                        and maze[new_x, new_y] == 0 and (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    queue.append(((new_x, new_y), distance + 1))
        return farthest_pos, max_distance


class MazeEnvAdapter:
    """Maze navigation environment adapter."""
    ACTIONS = {"up", "down", "left", "right"}
    DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    _RULES = """# Game Description
You are a maze-solving agent navigating from the START position to the GOAL position.

# Observations
- Your current position is shown as (row, col) coordinates.
- For each direction (up, down, left, right), you are told whether it leads to a wall, a path, or the goal.

# Your Goal
Navigate from the START position to the GOAL position in the fewest turns possible.

# Rules
- You can move in four directions: up, down, left, right.
- Moving into a wall keeps you at your current position.

# Action Format
Output your next move within \\boxed{}.
Example: \\boxed{up}"""

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        self.env_id = env_id
        if env_kwargs is None:
            raise ValueError("env_kwargs must include: shapes, max_turns")
        self.shapes = env_kwargs.get("shapes", [])
        self.max_turns = env_kwargs.get("max_turns", -1)
        self.shortest_path_min_length = env_kwargs.get("shortest_path_min_length", 8)
        self.shortest_path_max_length = env_kwargs.get("shortest_path_max_length", 8)
        try:
            self.shapes = [tuple(x) for x in self.shapes]
        except Exception:
            raise ValueError("shapes must be a list of [width, height]")
        self.generator = MazeGenerator(self.shapes)
        self.map: Optional[np.ndarray] = None
        self.init_position: Optional[Tuple[int, int]] = None
        self.current_position: Optional[Tuple[int, int]] = None
        self.shortest_path_len: Optional[int] = None
        self.current_turn = 0
        self.achieve_goal = False

    def get_rules(self) -> str:
        """Return static game rules."""
        return self._RULES

    def _get_cell_type(self, x, y):
        if self.map is None or x < 0 or x >= self.map.shape[0] or y < 0 or y >= self.map.shape[1]:
            return "wall"
        cell_value = self.map[x, y]
        if cell_value == 1:
            return "wall"
        elif cell_value == -1:
            return "goal"
        return "path"

    def _generate_observation(self):
        x, y = self.current_position
        up = self._get_cell_type(x - 1, y)
        down = self._get_cell_type(x + 1, y)
        left = self._get_cell_type(x, y - 1)
        right = self._get_cell_type(x, y + 1)
        return (
            f"Now you are at position ({x}, {y}) in the maze. "
            f"Around you, up leads to {up}, down leads to {down}, left leads to {left}, and right leads to {right}. "
        )

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[str, Dict[str, Any]]:
        if seed is None:
            raise ValueError("Seed must be provided.")
        max_tries = 1000
        for _ in range(max_tries):
            self.map, self.init_position, self.shortest_path_len = self.generator.generate(seed)
            if self.shortest_path_min_length <= self.shortest_path_len <= self.shortest_path_max_length:
                break
            seed += 1
        else:
            raise ValueError(f"Failed to generate maze within path length bounds after {max_tries} attempts.")

        self.current_turn = 0
        self.current_position = self.init_position
        self.achieve_goal = False
        # Return only the dynamic game state (position + surroundings)
        observation = "You are at the START position. " + self._generate_observation()
        info = {
            "turn": self.current_turn, "max_turns": self.max_turns,
            "terminated": False, "truncated": False,
        }
        return observation, info

    def step(self, action: Any) -> tuple[str, float, bool, Dict[str, Any]]:
        parsed_action = self._parse_action(str(action))
        self.current_turn += 1
        observation_text, reward = self._execute_action(parsed_action)

        terminated = False
        if self.achieve_goal:
            terminated = True
        if not terminated and self.current_turn >= self.max_turns:
            terminated = True
            observation_text = f"\nYou have reached the maximum number of turns ({self.max_turns}). Let's play again. "

        info = {
            "turn": self.current_turn, "max_turns": self.max_turns,
            "is_correct": self.achieve_goal,
            "terminated": terminated, "truncated": False,
            "raw_reward": float(reward),
        }
        return observation_text, float(reward), terminated, info

    def _parse_action(self, raw_text):
        boxed = extract_boxed_answer(raw_text)
        return (boxed if boxed else raw_text).strip().lower()

    def _execute_action(self, action):
        if action not in self.DELTA:
            return "Invalid action. Please try again.", 0.0
        x, y = self.current_position
        dx, dy = self.DELTA[action]
        new_x, new_y = x + dx, y + dy
        if new_x < 0 or new_x >= self.map.shape[0] or new_y < 0 or new_y >= self.map.shape[1]:
            return "You hit the wall. Please choose another direction.", 0.0
        if self.map[new_x, new_y] == 1:
            return "You hit the wall. Please choose another direction.", 0.0
        self.current_position = (new_x, new_y)
        if self.map[new_x, new_y] == -1:
            self.achieve_goal = True
            return "Congratulations! You arrived at the goal! Let's play again. ", 1.0
        return f"You move to {action}. " + self._generate_observation(), 0.0

    def close(self) -> None:
        pass


# ===========================================================================
# Adapter Registry
# ===========================================================================

ADAPTER_REGISTRY = {
    'GEMEnvAdapter': GEMEnvAdapter,
    'RockPaperScissorsEnvAdapter': RockPaperScissorsEnvAdapter,
    'BlackjackEnvAdapter': BlackjackEnvAdapter,
    'MazeEnvAdapter': MazeEnvAdapter,
}


def resolve_adapter_class(class_name: str):
    """Resolve an adapter class from a string name (short or dotted path)."""
    # Try short name first
    short_name = class_name.rsplit(".", 1)[-1] if "." in class_name else class_name
    if short_name in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[short_name]
    raise ValueError(f"Unknown adapter class: {class_name}. Available: {list(ADAPTER_REGISTRY.keys())}")
