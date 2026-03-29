"""
Training entry point.

Loads a YAML config, optionally overrides individual fields from the CLI,
then dispatches to the SB3 trainer.

Usage
─────
    # SB3-based SAC
    python scripts/train.py --config config/sac.yaml

    # SB3-based PPO with 8 parallel envs
    python scripts/train.py --config config/ppo_sb3.yaml

    # Override any field on the fly (dot notation for nested algo_kwargs)
    python scripts/train.py --config config/sac.yaml \\
        --set total_timesteps=500000 seed=1 algo_kwargs.batch_size=128
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# Make repo root importable regardless of working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deps", "robocasa"))
sys.path.insert(0, os.path.join(ROOT, "deps", "robosuite"))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _cast(value: str):
    """Best-effort cast of a CLI string to int / float / bool / None / str."""
    if value.lower() == "null" or value.lower() == "none":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply key=value overrides to a nested config dict.

    Supports dot notation for nested keys, e.g.:
        algo_kwargs.batch_size=128  →  cfg["algo_kwargs"]["batch_size"] = 128
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value format.")
        key, raw_value = item.split("=", 1)
        value = _cast(raw_value)
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def load_config(config_path: str, overrides: list[str]) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return _apply_overrides(cfg, overrides)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train an RL policy on RoboCasa.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config fields, e.g. --set total_timesteps=500000 seed=1",
    )
    parser.add_argument(
        "--reward", default=None,
        help=(
            "Reward function to use. Available: "
            "StagedPickPlaceReward, ReleasingPickPlaceReward, "
            "DensePickPlaceReward, BinaryMilestoneReward, ComposedPickPlaceReward. "
            "Defaults to StagedPickPlaceReward."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)

    print(f"Config  : {args.config}")
    if args.set:
        print(f"Overrides: {args.set}")

    from rl.trainer import TrainConfig, train
    from rl.reward import (
        BinaryMilestoneReward,
        ComposedPickPlaceReward,
        DensePickPlaceReward,
        ReleasingPickPlaceReward,
        StagedPickPlaceReward,
    )

    REWARD_REGISTRY = {
        "StagedPickPlaceReward":    StagedPickPlaceReward,
        "ReleasingPickPlaceReward": ReleasingPickPlaceReward,
        "DensePickPlaceReward":     DensePickPlaceReward,
        "BinaryMilestoneReward":    BinaryMilestoneReward,
        "ComposedPickPlaceReward":  ComposedPickPlaceReward,
    }

    reward_name = args.reward or cfg.pop("reward_fn", "StagedPickPlaceReward")
    if reward_name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward '{reward_name}'. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    reward_fn = REWARD_REGISTRY[reward_name]()
    print(f"Reward  : {reward_name}")

    cfg.pop("trainer", None)
    cfg.pop("reward_fn", None)
    seed = cfg.pop("seed", 42)

    train_cfg = TrainConfig(seed=seed, **{k: v for k, v in cfg.items()
                                          if k in TrainConfig.__dataclass_fields__})
    train(train_cfg, reward_fn=reward_fn)


if __name__ == "__main__":
    main()
