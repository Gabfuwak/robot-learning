"""
Evaluation entry point.

Loads a trained checkpoint and runs N evaluation episodes, printing
per-episode and aggregate stats.

Works with both SB3 checkpoints (.zip) and custom PPO checkpoints (.pt).

Usage
─────
    # Evaluate an SB3 model
    python scripts/eval.py \\
        --config config/sac.yaml \\
        --checkpoint runs/my_run/sac_final.zip \\
        --n_episodes 20

    # Evaluate a custom PPO model
    python scripts/eval.py \\
        --config config/ppo_custom.yaml \\
        --checkpoint runs/my_run/ppo_final.pt \\
        --n_episodes 20

    # Render while evaluating
    python scripts/eval.py \\
        --config config/sac.yaml \\
        --checkpoint runs/my_run/sac_final.zip \\
        --render

    # Evaluate on training distribution instead of test
    python scripts/eval.py \\
        --config config/sac.yaml \\
        --checkpoint runs/my_run/sac_final.zip \\
        --split train
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deps", "robocasa"))
sys.path.insert(0, os.path.join(ROOT, "deps", "robosuite"))


# ---------------------------------------------------------------------------
# Environment builder (shared between SB3 and custom eval)
# ---------------------------------------------------------------------------

def build_env(cfg: dict, split: str, render: bool):
    """Build a single (unwrapped) RoboCasaWrapper for evaluation."""
    from robosuite.controllers import load_composite_controller_config
    from rl.env_wrapper import RoboCasaWrapper
    from rl.reward import StagedPickPlaceReward
    from rl.trainer import ENV_REGISTRY

    env_name     = cfg.get("env_name", "PickPlaceCounterToCabinet")
    horizon      = cfg.get("horizon", 500)
    control_freq = cfg.get("control_freq", 20)
    image_size   = cfg.get("image_size", 64)
    use_camera   = cfg.get("use_camera_obs", False)
    camera_names = cfg.get("camera_names", [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ])
    seed = cfg.get("seed", 42)

    # Split determines which layouts/styles are used
    if split == "test":
        layout_ids, style_ids = -1, -1   # layouts/styles 1-10
    elif split == "train":
        layout_ids, style_ids = -2, -2   # layouts/styles 11-60
    else:
        layout_ids, style_ids = -3, -3   # all

    env_cls = ENV_REGISTRY[env_name]
    ctrl    = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw_env = env_cls(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=use_camera,
        has_renderer=render,
        has_offscreen_renderer=use_camera and not render,
        use_object_obs=True,
        camera_names=camera_names if use_camera else [],
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=control_freq,
        ignore_done=False,
        seed=seed + 999,   # different seed from training
        horizon=horizon,
        layout_ids=layout_ids,
        style_ids=style_ids,
    )
    raw_env.reset()

    return RoboCasaWrapper(
        raw_env,
        reward_fn=StagedPickPlaceReward(),
        use_camera_obs=use_camera,
        camera_names=camera_names,
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# SB3 evaluation
# ---------------------------------------------------------------------------

def eval_sb3(checkpoint: str, cfg: dict, n_episodes: int, split: str, render: bool):
    from stable_baselines3.common.base_class import BaseAlgorithm
    from rl.trainer import ALGO_REGISTRY

    algo_name = cfg.get("algo", "SAC")
    algo_cls  = ALGO_REGISTRY[algo_name]

    env = build_env(cfg, split, render)

    print(f"Loading {algo_name} checkpoint: {checkpoint}")
    model: BaseAlgorithm = algo_cls.load(checkpoint, env=None)

    results = _run_episodes(
        n_episodes=n_episodes,
        env=env,
        predict_fn=lambda obs: model.predict(obs["state"], deterministic=True)[0],
        render=render,
    )

    env.close()
    return results


# ---------------------------------------------------------------------------
# Custom PPO evaluation
# ---------------------------------------------------------------------------

def eval_custom_ppo(checkpoint: str, cfg: dict, n_episodes: int, split: str, render: bool):
    import torch
    from rl.custom_ppo import PPOConfig, ActorCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env    = build_env(cfg, split, render)

    print(f"Loading custom PPO checkpoint: {checkpoint}")
    ckpt   = torch.load(checkpoint, map_location=device, weights_only=False)
    ppo_cfg: PPOConfig = ckpt["cfg"]
    policy = ActorCritic(env.observation_space, env.action_space.shape[0], ppo_cfg).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    def predict_fn(obs: dict) -> np.ndarray:
        obs_t = {
            k: torch.from_numpy(np.array(v, dtype=np.float32)).unsqueeze(0).to(device)
            for k, v in obs.items()
        }
        action, _, _ = policy.predict(obs_t, deterministic=True)
        return action[0]

    results = _run_episodes(
        n_episodes=n_episodes,
        env=env,
        predict_fn=predict_fn,
        render=render,
    )

    env.close()
    return results


# ---------------------------------------------------------------------------
# Shared episode runner
# ---------------------------------------------------------------------------

def _run_episodes(n_episodes: int, env, predict_fn, render: bool) -> dict:
    """
    Run n_episodes and return aggregate stats.

    Returns dict with keys:
        episode_rewards  list[float]
        episode_lengths  list[int]
        success_rate     float
        mean_reward      float
        std_reward       float
    """
    import robocasa.utils.object_utils as OU

    episode_rewards = []
    episode_lengths = []
    successes       = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        done = False

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            if render:
                env.render()

        # Check task success via environment internals
        raw_env = env.unwrapped_env
        success = bool(raw_env._check_success())

        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        successes.append(success)

        print(
            f"  Episode {ep + 1:>3}/{n_episodes} | "
            f"reward={total_reward:7.3f} | steps={step:>4} | "
            f"success={'YES' if success else 'NO '}"
        )

    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate":    float(np.mean(successes)),
        "mean_reward":     float(np.mean(episode_rewards)),
        "std_reward":      float(np.std(episode_rewards)),
        "mean_length":     float(np.mean(episode_lengths)),
    }
    return results


def print_summary(results: dict):
    print("\n── Evaluation Summary ─────────────────────────")
    print(f"  Success rate : {results['success_rate'] * 100:.1f}%")
    print(f"  Mean reward  : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean length  : {results['mean_length']:.1f} steps")
    print("────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy on RoboCasa.")
    parser.add_argument("--config",     required=True, help="Path to YAML config file used for training.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.zip for SB3, .pt for custom PPO).")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--split",      default="test", choices=["train", "test", "all"],
                        help="Which layout/style distribution to evaluate on (default: test).")
    parser.add_argument("--render",     action="store_true", help="Render the environment during eval.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer_type = cfg.get("trainer", "sb3")
    checkpoint   = args.checkpoint

    print(f"Trainer    : {trainer_type}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Episodes   : {args.n_episodes}")
    print(f"Split      : {args.split}")

    if trainer_type == "sb3":
        results = eval_sb3(checkpoint, cfg, args.n_episodes, args.split, args.render)
    elif trainer_type == "custom_ppo":
        results = eval_custom_ppo(checkpoint, cfg, args.n_episodes, args.split, args.render)
    else:
        raise ValueError(f"Unknown trainer '{trainer_type}'.")

    print_summary(results)


if __name__ == "__main__":
    main()
