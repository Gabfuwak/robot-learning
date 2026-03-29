"""
Behaviour Cloning (BC) for RoboCasa.

Algorithm-specific components only. Shared config and data-collection
utilities live in il/base.py.

    BCPolicy              — state → action MLP policy
    train_bc()            — supervised training loop (MSE, Adam, cosine LR)
    evaluate_bc()         — environment rollout with optional video saving
    warm_start_from_bc()  — copy BC weights into a freshly built SB3 model so
                            that RL fine-tuning starts from the BC solution
                            rather than random initialisation (BC → RL warm-start)

Entry points
────────────
    scripts/train_IL.py    — train a BC policy from demonstrations
    scripts/eval_IL.py     — evaluate a BC checkpoint
    scripts/train.py       — RL training; pass bc_checkpoint in config/CLI to warm-start
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from il.base import ILConfig


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """
    Behaviour-cloning policy: flat state → continuous action.

    Architecture
    ────────────
      StateEncoder  (reused from rl/architecture.py — 2-layer MLP with LayerNorm)
        → net_arch hidden layers (Linear + ReLU)
          → Linear(hidden[-1], action_dim) → Tanh   (output in [-1, 1])
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        embed_dim:  int,
        net_arch:   list[int],
    ):
        super().__init__()
        from rl.architecture import StateEncoder
        self.encoder = StateEncoder(state_dim, embed_dim)

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for hidden in net_arch:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.action_head = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.encoder(state))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bc(
    policy:    BCPolicy,
    states:    np.ndarray,
    actions:   np.ndarray,
    cfg:       ILConfig,
    save_path: str,
) -> tuple[BCPolicy, np.ndarray, np.ndarray]:
    """
    Supervised behaviour cloning with MSE loss, Adam optimiser, and cosine LR decay.

    Saves bc_best.pt (lowest training loss) and bc_final.pt, plus normalisation
    stats (state_mean.npy, state_std.npy) needed at inference time.

    Returns
    -------
    policy     : trained BCPolicy (on device)
    state_mean : (1, STATE_DIM) float32
    state_std  : (1, STATE_DIM) float32
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    policy = policy.to(device)

    # Z-score normalisation
    state_mean = states.mean(axis=0, keepdims=True).astype(np.float32)
    state_std  = (states.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    np.save(os.path.join(save_path, "state_mean.npy"), state_mean)
    np.save(os.path.join(save_path, "state_std.npy"),  state_std)

    states_norm = ((states - state_mean) / state_std).astype(np.float32)

    dataset = TensorDataset(
        torch.FloatTensor(states_norm),
        torch.FloatTensor(actions),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
    )
    loss_fn   = nn.MSELoss()
    best_loss = float("inf")
    n_samples = len(dataset)

    print(f"[train] {n_samples:,} transitions | {cfg.n_epochs} epochs | batch={cfg.batch_size} | lr={cfg.lr}")

    for epoch in range(1, cfg.n_epochs + 1):
        policy.train()
        running_loss = 0.0

        for batch_states, batch_actions in loader:
            batch_states  = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            pred = policy(batch_states)
            loss = loss_fn(pred, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += loss.item() * len(batch_states)

        scheduler.step()
        epoch_loss = running_loss / n_samples

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{cfg.n_epochs}"
                f"  loss={epoch_loss:.6f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(policy.state_dict(), os.path.join(save_path, "bc_best.pt"))

    torch.save(policy.state_dict(), os.path.join(save_path, "bc_final.pt"))
    print(f"[train] Saved  → {save_path}/bc_final.pt  (best loss: {best_loss:.6f})")

    return policy, state_mean, state_std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def _make_eval_env(cfg: ILConfig, save_video: bool = False, render_size: int = 256):
    """Build a RoboCasaWrapper on the test distribution for evaluation."""
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
    from robosuite.controllers import load_composite_controller_config

    from rl.env_wrapper import RoboCasaWrapper
    from rl.reward import StagedPickPlaceReward

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
        control_freq=cfg.control_freq,
        ignore_done=False,
        horizon=cfg.horizon,
        layout_ids=cfg.layout_ids,
        style_ids=cfg.style_ids,
    )
    return RoboCasaWrapper(env=raw_env, reward_fn=StagedPickPlaceReward())


def _render_tiled_frame(raw_env, render_size: int) -> np.ndarray:
    """Render all four viz cameras and stitch into a 2×2 grid."""
    cols = 2
    rows = (len(_VIZ_CAMERAS) + cols - 1) // cols
    tile_rows = []
    for r in range(rows):
        row_frames = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(_VIZ_CAMERAS):
                frame = raw_env.sim.render(
                    camera_name=_VIZ_CAMERAS[idx],
                    width=render_size,
                    height=render_size,
                    depth=False,
                )
                frame = np.flipud(frame)
            else:
                frame = np.zeros((render_size, render_size, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)


def evaluate_bc(
    policy:      BCPolicy,
    state_mean:  np.ndarray,
    state_std:   np.ndarray,
    cfg:         ILConfig,
    save_video:  bool = False,
    video_dir:   str  = "eval_videos_il",
    render_size: int  = 256,
) -> dict:
    """
    Roll out the BC policy in the environment for cfg.n_eval_episodes episodes.

    If save_video=True, a 2×2 tiled video (4 cameras) is saved for each episode
    under <video_dir>/ep_00.mp4, ep_01.mp4, ...

    Returns a dict with success_rate, mean_reward, std_reward, mean_length.
    """
    import imageio

    device = next(policy.parameters()).device
    policy.eval()

    env = _make_eval_env(cfg, save_video=save_video, render_size=render_size)
    if save_video:
        os.makedirs(video_dir, exist_ok=True)

    successes:     list[bool]  = []
    total_rewards: list[float] = []
    lengths:       list[int]   = []

    for ep in range(cfg.n_eval_episodes):
        obs, _       = env.reset()
        total_reward = 0.0
        ep_len       = 0
        done         = False
        frames: list[np.ndarray] = []

        while not done:
            state      = obs["state"]
            state_norm = (state - state_mean[0]) / state_std[0]
            with torch.no_grad():
                state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
                action  = policy(state_t).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            ep_len       += 1
            done          = terminated or truncated

            if save_video:
                frames.append(_render_tiled_frame(env.unwrapped_env, render_size))

        success = bool(env.unwrapped_env._check_success())
        successes.append(success)
        total_rewards.append(total_reward)
        lengths.append(ep_len)

        print(
            f"  Episode {ep + 1:>3}/{cfg.n_eval_episodes}"
            f"  reward={total_reward:7.3f}"
            f"  steps={ep_len:>4}"
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
# BC → RL warm-start
# ---------------------------------------------------------------------------

def warm_start_from_bc(
    sb3_model,
    bc_checkpoint: str,
    state_dim:  int       = 110,
    action_dim: int       = 12,
    embed_dim:  int       = 256,
    net_arch:   list[int] = None,
) -> None:
    """
    Copy BC policy weights into a freshly built SB3 model so that RL fine-tuning
    starts from the BC solution rather than random initialisation.

    Verified weight mapping (net_arch=[256,256], state-only obs):

      BCPolicy                         SB3 SAC actor
      ─────────────────────────────    ──────────────────────────────────────
      encoder                      →  actor.features_extractor.state_encoder
      action_head[0] Linear(256→256)→  actor.latent_pi[0]
      action_head[2] Linear(256→256)→  actor.latent_pi[2]
      action_head[4] Linear(256→12) →  actor.mu
      action_head[5] Tanh           →  (dropped — SB3 SAC applies its own squashing)

    The encoder is also copied to the critic's feature extractor so the shared
    state representation starts warm for both actor and critic.

    Args:
        sb3_model:      A freshly instantiated SB3 model (SAC, TD3, PPO, …).
        bc_checkpoint:  Path to bc_best.pt / bc_final.pt.
        state_dim:      Must match the checkpoint (default 110).
        action_dim:     Must match the checkpoint (default 12).
        embed_dim:      state_embed_dim used when training BC (default 256).
        net_arch:       net_arch used when training BC (default [256, 256]).
    """
    if net_arch is None:
        net_arch = [256, 256]

    # Reconstruct BCPolicy with the same architecture and load saved weights
    bc = BCPolicy(state_dim, action_dim, embed_dim, net_arch)
    bc.load_state_dict(torch.load(bc_checkpoint, map_location="cpu"))
    bc.eval()

    policy = sb3_model.policy
    actor  = policy.actor

    # 1. Encoder → actor feature extractor
    actor.features_extractor.state_encoder.load_state_dict(bc.encoder.state_dict())

    # 2. action_head hidden layers → actor.latent_pi
    #    BC action_head indices: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear, 5=Tanh
    #    latent_pi indices:      0=Linear, 1=ReLU, 2=Linear, 3=ReLU
    for latent_idx, bc_idx in enumerate([0, 2]):          # only Linear layers have params
        actor.latent_pi[latent_idx * 2].load_state_dict(
            bc.action_head[bc_idx].state_dict()
        )

    # 3. Final BC linear → actor.mu  (Tanh is dropped; SAC squashes internally)
    actor.mu.load_state_dict(bc.action_head[4].state_dict())

    # 4. Copy encoder to critic feature extractor(s) as well
    #    SB3 SAC uses share_features_extractor=False by default, so the critic
    #    has its own separate extractor.
    for critic in [getattr(policy, "critic", None), getattr(policy, "critic_target", None)]:
        if critic is not None and hasattr(critic, "features_extractor"):
            fe = critic.features_extractor
            if hasattr(fe, "state_encoder"):
                fe.state_encoder.load_state_dict(bc.encoder.state_dict())

    n_copied = (
        sum(p.numel() for p in bc.encoder.parameters())
        + sum(p.numel() for p in bc.action_head[:5].parameters())
    )
    print(f"[warm-start] Copied {n_copied:,} parameters from {bc_checkpoint}")
