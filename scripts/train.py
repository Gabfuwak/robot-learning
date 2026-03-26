"""
Training entry point.

Loads a YAML config, optionally overrides individual fields from the CLI,
then dispatches to the appropriate trainer (SB3 or custom PPO).

Usage
─────
    # SB3-based SAC
    python scripts/train.py --config config/sac.yaml

    # SB3-based PPO with 8 parallel envs
    python scripts/train.py --config config/ppo_sb3.yaml

    # Custom from-scratch PPO
    python scripts/train.py --config config/ppo_custom.yaml

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
# Trainer dispatch
# ---------------------------------------------------------------------------

def run_sb3(cfg: dict):
    from rl.trainer import TrainConfig, train
    from rl.reward import StagedPickPlaceReward

    # Pop trainer key — not part of TrainConfig
    cfg.pop("trainer", None)
    seed = cfg.pop("seed", 42)

    train_cfg = TrainConfig(seed=seed, **{k: v for k, v in cfg.items()
                                          if k in TrainConfig.__dataclass_fields__})
    train(train_cfg, reward_fn=StagedPickPlaceReward())


def run_custom_ppo(cfg: dict):
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
    from robosuite.controllers import load_composite_controller_config

    from rl.custom_ppo import PPOConfig, PPOTrainer
    from rl.env_wrapper import RoboCasaWrapper
    from rl.reward import StagedPickPlaceReward
    from rl.trainer import ENV_REGISTRY

    cfg.pop("trainer", None)
    env_name     = cfg.pop("env_name", "PickPlaceCounterToCabinet")
    layout_ids   = cfg.pop("layout_ids", -2)
    style_ids    = cfg.pop("style_ids", -2)
    horizon      = cfg.pop("horizon", 500)
    control_freq = cfg.pop("control_freq", 20)
    image_size   = cfg.pop("image_size", 64)
    use_camera_obs = cfg.pop("use_camera_obs", False)
    camera_names = cfg.pop("camera_names", [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ])
    seed = cfg.pop("seed", 42)

    # Build raw env
    env_cls = ENV_REGISTRY[env_name]
    ctrl    = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw_env = env_cls(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=use_camera_obs,
        has_renderer=False,
        has_offscreen_renderer=use_camera_obs,
        use_object_obs=True,
        camera_names=camera_names if use_camera_obs else [],
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=control_freq,
        ignore_done=False,
        seed=seed,
        horizon=horizon,
        layout_ids=layout_ids,
        style_ids=style_ids,
    )
    raw_env.reset()

    env = RoboCasaWrapper(
        raw_env,
        reward_fn=StagedPickPlaceReward(),
        use_camera_obs=use_camera_obs,
        camera_names=camera_names,
        image_size=image_size,
    )

    # Build PPOConfig from remaining keys
    ppo_cfg = PPOConfig(**{k: v for k, v in cfg.items()
                           if k in PPOConfig.__dataclass_fields__})
    trainer = PPOTrainer(ppo_cfg, env)
    trainer.train()
    env.close()


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
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    trainer_type = cfg.get("trainer", "sb3")

    print(f"Trainer : {trainer_type}")
    print(f"Config  : {args.config}")
    if args.set:
        print(f"Overrides: {args.set}")

    if trainer_type == "sb3":
        run_sb3(cfg)
    elif trainer_type == "custom_ppo":
        run_custom_ppo(cfg)
    else:
        raise ValueError(f"Unknown trainer '{trainer_type}'. Choose: sb3, custom_ppo")


if __name__ == "__main__":
    main()
