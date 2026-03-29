"""
Diffusion Policy training for RoboCasa (state-only, no camera obs).

Uses LeRobot's DiffusionPolicy with the 110-dim environment observation split
into two LeRobot feature keys:

    observation.state              (68-dim) — robot proprioception  [FeatureType.STATE]
    observation.environment_state  (42-dim) — object positions       [FeatureType.ENV]

This split is required by DiffusionPolicy's validate_features(), which demands
at least one image or an environment-state feature in addition to robot state.
Both keys together reconstruct the full 110-dim observation used at test time.

Architecture (memory-reduced defaults)
────────────────────────────────────────
  Vision encoder : disabled (state-only)
  UNet           : down_dims=(256, 512, 1024)  → ~20M params (vs 79M default)
  Noise schedule : DDIM, 10 inference steps   (vs DDPM 100)

Sequence structure per sample
──────────────────────────────
  observation.state              : (n_obs_steps, 68)
  observation.environment_state  : (n_obs_steps, 42)
  action                         : (horizon, 12)
  action_is_pad                  : (horizon,) bool  — True for positions past episode end

Entry points
────────────
    scripts/train_diffusion.py  — train from demonstrations
    scripts/eval_IL.py          — not applicable (use scripts/eval_diffusion.py)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from il.base import ILConfig, collect_episode_data

# LeRobot feature key constants
_OBS_STATE     = "observation.state"            # robot proprioception (68-dim)
_OBS_ENV_STATE = "observation.environment_state"  # object state (42-dim)
_ACTION        = "action"

ROBOT_DIM = 68
OBJ_DIM   = 42
ACTION_DIM = 12


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiffusionILConfig:
    # --- Dataset ---
    task:    str           = "PickPlaceCounterToCabinet"
    split:   str           = "target"
    source:  str           = "human"
    n_demos: Optional[int] = None

    # --- Sequence structure ---
    n_obs_steps:    int = 2    # observation history fed to the model
    horizon:        int = 16   # total action sequence predicted by UNet
    n_action_steps: int = 8    # actions executed per re-plan cycle

    # --- UNet architecture (reduced from LeRobot defaults for limited VRAM) ---
    down_dims:               tuple[int, ...] = field(default_factory=lambda: (256, 512, 1024))
    diffusion_step_embed_dim: int            = 128

    # --- Noise schedule ---
    noise_scheduler_type: str = "DDIM"   # DDPM or DDIM
    num_train_timesteps:  int = 100
    num_inference_steps:  int = 10       # DDIM steps at inference (≪ DDPM 100)
    prediction_type:      str = "epsilon"

    # --- Training ---
    n_steps:         int   = 100_000
    batch_size:      int   = 64
    lr:              float = 1e-4
    lr_warmup_steps: int   = 500
    weight_decay:    float = 1e-6
    grad_clip:       float = 10.0
    num_workers:     int   = 2
    seed:            int   = 42

    # --- Evaluation ---
    n_eval_episodes: int = 10
    eval_freq:       int = 10_000
    layout_ids:      int = -1       # -1 = test layouts
    style_ids:       int = -1
    horizon_env:     int = 500      # episode horizon for evaluation env

    # --- Persistence ---
    save_dir:        str = "runs_diffusion"
    cache_dir:       str = ".demo_cache"
    checkpoint_freq: int = 10_000


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DemoSequenceDataset(Dataset):
    """
    Sliding-window dataset over a list of demonstration episodes.

    Splits the 110-dim state into:
      - observation.state             (first 68 dims) — robot proprioception
      - observation.environment_state (last 42 dims)  — object state

    Each item:
      observation.state             : (n_obs_steps, 68)  left-padded at episode start
      observation.environment_state : (n_obs_steps, 42)  left-padded at episode start
      action                        : (horizon, 12)      right-padded at episode end
      action_is_pad                 : (horizon,) bool    True for padded positions
    """

    def __init__(
        self,
        episode_data:       list[tuple[np.ndarray, np.ndarray]],
        n_obs_steps:        int,
        horizon:            int,
        drop_n_last_frames: int = 0,
    ):
        self.episode_states  = [ep[0] for ep in episode_data]   # each (T, 110)
        self.episode_actions = [ep[1] for ep in episode_data]   # each (T, 12)
        self.n_obs_steps     = n_obs_steps
        self.horizon         = horizon

        self._index: list[tuple[int, int]] = []
        for ep_idx, (ep_s, ep_a) in enumerate(episode_data):
            T       = len(ep_a)
            valid_T = max(0, T - drop_n_last_frames)
            for t in range(valid_T):
                self._index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start_t = self._index[idx]
        ep_states  = self.episode_states[ep_idx]   # (T, 110)
        ep_actions = self.episode_actions[ep_idx]  # (T, 12)
        T = len(ep_actions)

        # Observation history: left-pad by repeating the first frame
        obs = np.stack([
            ep_states[max(0, start_t - self.n_obs_steps + 1 + i)]
            for i in range(self.n_obs_steps)
        ])  # (n_obs_steps, 110)

        # Action sequence: right-pad by repeating the last action
        actions, is_pad = [], []
        for i in range(self.horizon):
            t = start_t + i
            if t < T:
                actions.append(ep_actions[t])
                is_pad.append(False)
            else:
                actions.append(ep_actions[T - 1])
                is_pad.append(True)
        actions = np.stack(actions)            # (horizon, 12)
        is_pad  = np.array(is_pad, dtype=bool)

        return {
            _OBS_STATE:     torch.FloatTensor(obs[:, :ROBOT_DIM]),   # (n_obs, 68)
            _OBS_ENV_STATE: torch.FloatTensor(obs[:, ROBOT_DIM:]),   # (n_obs, 42)
            _ACTION:        torch.FloatTensor(actions),
            "action_is_pad": torch.BoolTensor(is_pad),
        }


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def _compute_stats(episode_data: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Compute min/max/mean/std for normalization (MIN_MAX used for STATE/ENV/ACTION)."""
    all_states  = np.concatenate([ep[0] for ep in episode_data], axis=0)  # (N, 110)
    all_actions = np.concatenate([ep[1] for ep in episode_data], axis=0)  # (N, 12)

    robot_states = all_states[:, :ROBOT_DIM]
    obj_states   = all_states[:, ROBOT_DIM:]

    def _s(arr: np.ndarray) -> dict:
        return {
            "min":  torch.FloatTensor(arr.min(axis=0)),
            "max":  torch.FloatTensor(arr.max(axis=0)),
            "mean": torch.FloatTensor(arr.mean(axis=0)),
            "std":  torch.FloatTensor(arr.std(axis=0).clip(min=1e-8)),
        }

    return {
        _OBS_STATE:     _s(robot_states),
        _OBS_ENV_STATE: _s(obj_states),
        _ACTION:        _s(all_actions),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_diffusion(
    policy,
    cfg:         DiffusionILConfig,
    save_video:  bool = False,
    video_dir:   str  = "eval_videos_diffusion",
    render_size: int  = 256,
) -> dict:
    """Roll out the Diffusion Policy for cfg.n_eval_episodes and return stats."""
    import imageio

    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
    from robosuite.controllers import load_composite_controller_config

    from rl.env_wrapper import RoboCasaWrapper
    from rl.reward import StagedPickPlaceReward

    _VIZ_CAMERAS = [
        "robot0_agentview_center", "robot0_agentview_left",
        "robot0_agentview_right",  "robot0_eye_in_hand",
    ]

    ctrl    = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw_env = PickPlaceCounterToCabinet(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=save_video,
        use_object_obs=True,
        camera_names=_VIZ_CAMERAS if save_video else [],
        camera_heights=render_size,
        camera_widths=render_size,
        control_freq=20,
        ignore_done=False,
        horizon=cfg.horizon_env,
        layout_ids=cfg.layout_ids,
        style_ids=cfg.style_ids,
    )
    env = RoboCasaWrapper(env=raw_env, reward_fn=StagedPickPlaceReward())

    if save_video:
        os.makedirs(video_dir, exist_ok=True)

    device = next(policy.parameters()).device
    policy.eval()

    successes, total_rewards, lengths = [], [], []

    for ep in range(cfg.n_eval_episodes):
        obs, _       = env.reset()
        policy.reset()                # clear the internal observation queue
        total_reward = 0.0
        ep_len       = 0
        done         = False
        frames: list[np.ndarray] = []

        while not done:
            state = obs["state"]                         # (110,)
            robot = torch.FloatTensor(state[:ROBOT_DIM]).unsqueeze(0).to(device)   # (1, 68)
            obj   = torch.FloatTensor(state[ROBOT_DIM:]).unsqueeze(0).to(device)   # (1, 42)

            with torch.no_grad():
                action_t = policy.select_action({
                    _OBS_STATE:     robot,
                    _OBS_ENV_STATE: obj,
                })                                       # (1, 12)
            action = action_t.squeeze(0).cpu().numpy()  # (12,)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            ep_len       += 1
            done          = terminated or truncated

            if save_video:
                cols, rows = 2, 2
                tile_rows = []
                for r in range(rows):
                    row_frames = []
                    for c in range(cols):
                        ic = r * cols + c
                        if ic < len(_VIZ_CAMERAS):
                            frame = raw_env.sim.render(
                                camera_name=_VIZ_CAMERAS[ic],
                                width=render_size, height=render_size, depth=False,
                            )
                            frame = np.flipud(frame)
                        else:
                            frame = np.zeros((render_size, render_size, 3), dtype=np.uint8)
                        row_frames.append(frame)
                    tile_rows.append(np.concatenate(row_frames, axis=1))
                frames.append(np.concatenate(tile_rows, axis=0))

        success = bool(raw_env._check_success())
        successes.append(success)
        total_rewards.append(total_reward)
        lengths.append(ep_len)
        print(
            f"  Episode {ep + 1:>3}/{cfg.n_eval_episodes}"
            f"  reward={total_reward:7.3f}  steps={ep_len:>4}"
            f"  success={'YES' if success else 'NO '}"
        )

        if save_video and frames:
            vid_path = os.path.join(video_dir, f"ep_{ep:02d}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"    -> saved {vid_path}")

    env.close()
    results = {
        "success_rate": float(np.mean(successes)),
        "mean_reward":  float(np.mean(total_rewards)),
        "std_reward":   float(np.std(total_rewards)),
        "mean_length":  float(np.mean(lengths)),
    }
    print(
        f"\n── Evaluation Summary ──────────────────────────\n"
        f"  Success rate : {results['success_rate'] * 100:.1f}%\n"
        f"  Mean reward  : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}\n"
        f"  Mean length  : {results['mean_length']:.1f} steps\n"
        f"────────────────────────────────────────────────"
    )
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_diffusion(cfg: DiffusionILConfig):
    """
    Train a Diffusion Policy on cached BC demonstrations and return the policy.
    """
    from lerobot.configs.policies import PolicyFeature, FeatureType, NormalizationMode
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diffusion] Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load per-episode data
    # ------------------------------------------------------------------
    il_cfg = ILConfig(
        task=cfg.task, split=cfg.split, source=cfg.source,
        n_demos=cfg.n_demos, cache_dir=cfg.cache_dir,
    )
    episode_data  = collect_episode_data(il_cfg)
    n_transitions = sum(len(ep[1]) for ep in episode_data)
    print(f"[diffusion] {len(episode_data)} episodes, {n_transitions:,} transitions")

    drop_n = max(0, cfg.horizon - cfg.n_action_steps)
    dataset = DemoSequenceDataset(
        episode_data=episode_data,
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.horizon,
        drop_n_last_frames=drop_n,
    )
    print(f"[diffusion] Dataset: {len(dataset):,} samples  "
          f"(n_obs={cfg.n_obs_steps}, horizon={cfg.horizon}, drop_last={drop_n})")

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # 2. Policy
    # ------------------------------------------------------------------
    dataset_stats = _compute_stats(episode_data)

    policy_cfg = DiffusionConfig(
        input_features={
            _OBS_STATE:     PolicyFeature(type=FeatureType.STATE, shape=(ROBOT_DIM,)),
            _OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV,   shape=(OBJ_DIM,)),
        },
        output_features={
            _ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        n_obs_steps    = cfg.n_obs_steps,
        horizon        = cfg.horizon,
        n_action_steps = cfg.n_action_steps,
        down_dims      = tuple(cfg.down_dims),
        diffusion_step_embed_dim = cfg.diffusion_step_embed_dim,
        noise_scheduler_type = cfg.noise_scheduler_type,
        num_train_timesteps  = cfg.num_train_timesteps,
        num_inference_steps  = cfg.num_inference_steps,
        prediction_type      = cfg.prediction_type,
        crop_shape           = None,    # no vision
        normalization_mapping = {
            "STATE":  NormalizationMode.MIN_MAX,
            "ENV":    NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
    )

    policy = DiffusionPolicy(policy_cfg, dataset_stats=dataset_stats).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[diffusion] Parameters: {n_params:,}")

    # Save dataset stats alongside checkpoints so eval_diffusion.py can load them
    import json as _json
    _stats_serialisable = {
        key: {k: v.tolist() for k, v in d.items()}
        for key, d in dataset_stats.items()
    }

    # ------------------------------------------------------------------
    # 3. Optimizer + LR schedule
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr,
        betas=(0.95, 0.999), eps=1e-8, weight_decay=cfg.weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=cfg.lr_warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.n_steps - cfg.lr_warmup_steps), eta_min=cfg.lr * 0.01,
    )
    lr_sched = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.lr_warmup_steps],
    )

    # ------------------------------------------------------------------
    # 4. Run
    # ------------------------------------------------------------------
    run_name  = f"{cfg.task}_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"[diffusion] Run:       {run_name}")
    print(f"[diffusion] Save path: {save_path}")

    with open(os.path.join(save_path, "dataset_stats.json"), "w") as _f:
        _json.dump(_stats_serialisable, _f)
    print(f"[diffusion] Steps: {cfg.n_steps:,}  batch={cfg.batch_size}  down_dims={cfg.down_dims}")

    best_loss   = float("inf")
    loader_iter = iter(loader)
    policy.train()

    for step in range(1, cfg.n_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()
        lr_sched.step()

        if step % 500 == 0 or step == 1:
            print(f"  step={step:>7,}/{cfg.n_steps:,}  "
                  f"loss={loss.item():.6f}  lr={lr_sched.get_last_lr()[0]:.2e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(policy.state_dict(), os.path.join(save_path, "diffusion_best.pt"))

        if step % cfg.checkpoint_freq == 0:
            ckpt = os.path.join(save_path, f"diffusion_step{step}.pt")
            torch.save(policy.state_dict(), ckpt)
            print(f"  [ckpt] → {ckpt}")

        if step % cfg.eval_freq == 0:
            print(f"\n[diffusion] Evaluating at step {step} ...")
            policy.eval()
            evaluate_diffusion(policy, cfg)
            policy.train()

    torch.save(policy.state_dict(), os.path.join(save_path, "diffusion_final.pt"))
    print(f"[diffusion] Final → {save_path}/diffusion_final.pt  (best loss: {best_loss:.6f})")
    return policy
