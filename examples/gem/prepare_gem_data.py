"""Prepare GEM multi-task multi-episode parquet data for LaMer training.

Generates train/val parquet files with task dicts in extra_info.
Uses SHA256-based deterministic seeding identical to Orbit's prepare_gem_data.py
for fair comparison.

Usage:
    python -m examples.gem.prepare_gem_data \
        --config examples/gem/multi_task_multi_episode_config.yaml \
        --seed 42 \
        --output_dir ~/data/gem-multi-task
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


def get_task_specific_seed(task_cfg: Dict[str, Any], global_seed: int) -> int:
    """Generate a deterministic seed for a task based on its configuration.

    Identical to Orbit's implementation for fair comparison.
    Excludes train_size/test_size so seed depends only on task identity.
    """
    task_params = {
        k: v for k, v in task_cfg.items()
        if k not in ["train_size", "test_size"]
    }
    sorted_items = sorted(task_params.items())
    param_str = str(sorted_items) + str(global_seed)
    hash_bytes = hashlib.sha256(param_str.encode()).digest()
    task_seed = int.from_bytes(hash_bytes[:8], byteorder='big') % (2**31)
    return task_seed


def process_task_list(
    task_list: List[Dict[str, Any]],
    global_seed: int,
    is_train: bool,
) -> List[Dict[str, Any]]:
    """Process task configs into individual task dicts with deterministic seeds.

    Identical logic to Orbit's process_task_list for seed compatibility.
    """
    result = []
    for task_cfg in task_list:
        size_key = "train_size" if is_train else "test_size"
        size = task_cfg.get(size_key, 512 if is_train else 64)
        env_id = task_cfg["env_id"]

        task_specific_seed = get_task_specific_seed(task_cfg, global_seed)
        if is_train:
            task_specific_seed = task_specific_seed * 2

        task_rng = np.random.default_rng(task_specific_seed)
        seeds = task_rng.integers(0, 1_000_000, size=size).tolist()

        for i, s in enumerate(seeds):
            task_dict: Dict[str, Any] = {}
            for key, value in task_cfg.items():
                if key not in ["train_size", "test_size"]:
                    task_dict[key] = value
            task_dict["seed"] = int(s)
            task_dict["uid"] = f"{env_id}-{s}"
            task_dict["data_source"] = env_id
            result.append(task_dict)

    return result


def build_parquet_rows(task_dicts: List[Dict[str, Any]], split: str) -> List[Dict]:
    """Convert task dicts to parquet-compatible rows for LaMer's RLHFDataset."""
    rows = []
    for i, task_dict in enumerate(task_dicts):
        row = {
            "data_source": task_dict["data_source"],
            "prompt": [{"role": "user", "content": "placeholder"}],
            "extra_info": task_dict,
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare GEM multi-task data for LaMer")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to multi_task_multi_episode_config.yaml")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for deterministic task generation")
    parser.add_argument("--output_dir", type=str, default="~/data/gem-multi-task",
                        help="Output directory for parquet files")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_tasks = config.get("train_tasks", [])
    val_tasks = config.get("val_tasks", [])

    if not train_tasks:
        raise ValueError("train_tasks must be specified in config")
    if not val_tasks:
        raise ValueError("val_tasks must be specified in config")

    # Generate task dicts with Orbit-identical seeds
    train_data = process_task_list(train_tasks, args.seed, is_train=True)
    val_data = process_task_list(val_tasks, args.seed, is_train=False)

    print(f"Generated {len(train_data)} training tasks:")
    for task_cfg in train_tasks:
        env_id = task_cfg["env_id"]
        count = sum(1 for d in train_data if d["env_id"] == env_id)
        print(f"  {env_id}: {count} tasks")

    print(f"Generated {len(val_data)} validation tasks:")
    for task_cfg in val_tasks:
        env_id = task_cfg["env_id"]
        count = sum(1 for d in val_data if d["env_id"] == env_id)
        print(f"  {env_id}: {count} tasks")

    # Build parquet rows
    train_rows = build_parquet_rows(train_data, "train")
    val_rows = build_parquet_rows(val_data, "test")

    # Save to parquet
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"\nSaved train parquet ({len(train_df)} rows): {train_path}")
    print(f"Saved val parquet ({len(val_df)} rows): {val_path}")

    # Print sample for verification
    print(f"\nSample train task: {json.dumps(train_data[0], indent=2)}")
    print(f"Sample val task: {json.dumps(val_data[0], indent=2)}")


if __name__ == "__main__":
    main()
