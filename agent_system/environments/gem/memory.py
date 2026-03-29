"""Memory manager for GEM environments.

Follows the same pattern as SimpleMemoryMineSweeper:
stores per-step history and formats it for prompt injection.
"""

from typing import List, Dict, Any


class SimpleMemoryGEM:
    """Memory manager: stores & fetches per-environment history records."""

    def __init__(self, num_processes=0):
        self._data = [{} for _ in range(num_processes)]
        self.keys = None
        self.num_processes = num_processes

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, num_processes: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(num_processes)]
        self.num_processes = num_processes
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """Store a new record (one step of history) for each environment instance.

        Args:
            record: Dict where each key maps to a list of length num_processes.
                    Expected keys: 'text_obs', 'action', 'reward', 'dones', 'won'
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.num_processes):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int = 7,
        obs_key: str = "text_obs",
        action_key: str = "action",
        obs_length: int = 2,
    ) -> tuple:
        """Fetch and format recent interaction history for each environment.

        Args:
            history_length: Maximum past steps to retrieve.
            obs_key: Key for observations in stored records.
            action_key: Key for actions in stored records.
            obs_length: Number of recent steps to show full observations for;
                        older steps show truncated observations.

        Returns:
            memory_contexts: List of formatted history strings per environment.
            valid_lengths: List of actual valid history step counts.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.num_processes):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]

                if len(recent) - j > obs_length:
                    lines.append(
                        f"Action {step_num}: {act}\nObservation {step_num}: ..."
                    )
                else:
                    lines.append(
                        f"Action {step_num}: {act}\nObservation {step_num}:\n{obs}"
                    )
                if 'dones' in rec and rec['dones']:
                    valid_len = step_num
                    break

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
