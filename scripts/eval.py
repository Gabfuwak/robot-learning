"""
Evaluation entry point.

Loads a trained SB3 checkpoint and runs N evaluation episodes, printing
per-episode and aggregate stats. Optionally saves a tiled multi-camera video.

Usage
─────
    # Eval only (no video)
    python scripts/eval.py \\
        --config config/sac.yaml \\
        --checkpoint runs/my_run/sac_final.zip \\
        --n_episodes 20

    # Save tiled multi-camera video (4 views, 2x2 grid)
    python scripts/eval.py \\
        --config config/sac.yaml \\
        --checkpoint runs/my_run/sac_final.zip \\
        --save_video --video_dir eval_videos/my_run

    # Render live while evaluating
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


VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_tiled_frame(raw_env, width: int, height: int) -> np.ndarray:
    """Render all VIZ_CAMERAS and stitch into a 2-column tiled image."""
    cols = 2
    rows = (len(VIZ_CAMERAS) + cols - 1) // cols
    tile_rows = []
    for r in range(rows):
        row_frames = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(VIZ_CAMERAS):
                frame = raw_env.sim.render(
                    camera_name=VIZ_CAMERAS[idx],
                    width=width,
                    height=height,
                    depth=False,
                )
                frame = np.flipud(frame)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)


def build_env(cfg: dict, split: str, render: bool, save_video: bool = False, render_size: int = 256):
    """Build a single RoboCasaWrapper for evaluation."""
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

    if split == "test":
        layout_ids, style_ids = -1, -1
    elif split == "train":
        layout_ids, style_ids = -2, -2
    else:
        layout_ids, style_ids = -3, -3

    env_cls = ENV_REGISTRY[env_name]
    ctrl    = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw_env = env_cls(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=use_camera,
        has_renderer=render,
        has_offscreen_renderer=(use_camera and not render) or save_video,
        use_object_obs=True,
        camera_names=list(set((camera_names if use_camera else []) + (VIZ_CAMERAS if save_video else []))),
        camera_heights=render_size if save_video else image_size,
        camera_widths=render_size if save_video else image_size,
        control_freq=control_freq,
        ignore_done=False,
        seed=seed + 999,
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


def _run_episodes(n_episodes: int, env, predict_fn, render: bool,
                  save_video: bool = False, video_dir: str = "eval_videos",
                  render_size: int = 256) -> dict:
    if save_video:
        import imageio
        os.makedirs(video_dir, exist_ok=True)

    episode_rewards = []
    episode_lengths = []
    successes       = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        frames = []

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            if render:
                env.render()

            if save_video:
                frame = render_tiled_frame(env.unwrapped_env, render_size, render_size)
                frames.append(frame)

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

        if save_video and frames:
            vid_path = os.path.join(video_dir, f"ep_{ep:02d}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"    -> video saved to {vid_path}")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate":    float(np.mean(successes)),
        "mean_reward":     float(np.mean(episode_rewards)),
        "std_reward":      float(np.std(episode_rewards)),
        "mean_length":     float(np.mean(episode_lengths)),
    }


def print_summary(results: dict):
    print("\n── Evaluation Summary ─────────────────────────")
    print(f"  Success rate : {results['success_rate'] * 100:.1f}%")
    print(f"  Mean reward  : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean length  : {results['mean_length']:.1f} steps")
    print("────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy on RoboCasa.")
    parser.add_argument("--config",      required=True, help="Path to YAML config file used for training.")
    parser.add_argument("--checkpoint",  required=True, help="Path to SB3 checkpoint (.zip).")
    parser.add_argument("--n_episodes",  type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--split",       default="test", choices=["train", "test", "all"],
                        help="Which layout/style distribution to evaluate on (default: test).")
    parser.add_argument("--render",      action="store_true", help="Render the environment live during eval.")
    parser.add_argument("--save_video",  action="store_true", help="Save tiled multi-camera video per episode.")
    parser.add_argument("--video_dir",   default="eval_videos", help="Directory to save videos (default: eval_videos).")
    parser.add_argument("--render_size", type=int, default=256, help="Width/height of each camera tile (default: 256).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algo_name = cfg.get("algo", "SAC")
    print(f"Algo       : {algo_name}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Episodes   : {args.n_episodes}")
    print(f"Split      : {args.split}")

    from stable_baselines3.common.base_class import BaseAlgorithm
    from rl.trainer import ALGO_REGISTRY

    algo_cls = ALGO_REGISTRY[algo_name]
    env      = build_env(cfg, args.split, args.render, args.save_video, args.render_size)

    print(f"Loading {algo_name} checkpoint: {args.checkpoint}")
    model: BaseAlgorithm = algo_cls.load(args.checkpoint, env=None)

    results = _run_episodes(
        n_episodes=args.n_episodes,
        env=env,
        predict_fn=lambda obs: model.predict(obs, deterministic=True)[0],
        render=args.render,
        save_video=args.save_video,
        video_dir=args.video_dir,
        render_size=args.render_size,
    )

    env.close()
    print_summary(results)


if __name__ == "__main__":
    main()
