"""
Generative Adversarial Imitation Learning (GAIL) for RoboCasa.

GAIL replaces the hand-crafted reward function with a learned discriminator.
The discriminator D(s, a) is trained to distinguish expert (s, a) pairs from
agent (s, a) pairs.  The RL agent (SAC) maximises log D(s, a), i.e. it tries
to produce trajectories that the discriminator cannot distinguish from the
expert demonstrations.

Architecture
────────────
  Discriminator : [state ‖ action] → Linear → LayerNorm → Tanh
                                    → Linear → LayerNorm → Tanh
                                    → Linear(1)    (raw logit, no activation)

  Policy        : SAC (stable-baselines3), optionally warm-started from BC

GAIL reward
───────────
  r(s, a) = log σ(clamp(D_logit, -10, 10))   ≡  -softplus(-clamp(logit))

  Clamped to avoid -inf rewards when the discriminator is very confident.
  Normalised to [-1, 0] by dividing by |log σ(-10)|.

Discriminator training
──────────────────────
  Every disc_update_freq environment steps the GAILCallback:
    1. Samples a fresh batch of (state, action) pairs from the expert buffer.
    2. Samples a fresh batch from an agent ring-buffer (numpy-backed, O(1) sample).
    3. Runs n_disc_updates gradient steps, resampling each step.
    4. Uses label smoothing (0.9 / 0.1) to prevent overconfident collapse.
    5. Clips discriminator gradients to max_norm=10.

Usage
─────
    python scripts/train_GAIL.py

    # Warm-start actor from a BC checkpoint
    python scripts/train_GAIL.py --bc_checkpoint runs_il/<run>/bc_best.pt

    # Fewer expert demos
    python scripts/train_GAIL.py --n_demos 50

    # All options
    python scripts/train_GAIL.py --help
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.reward import RewardFn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GAILConfig:
    # --- Expert data ---
    task:             str            = "PickPlaceCounterToCabinet"
    split:            str            = "target"
    source:           str            = "human"
    n_demos:          int | None     = None

    # --- Environment ---
    env_name:         str            = "PickPlaceCounterToCabinet"
    layout_ids:       int | list     = -2
    style_ids:        int | list     = -2
    horizon:          int            = 500
    control_freq:     int            = 20

    # --- Policy architecture ---
    state_embed_dim:  int            = 256
    image_embed_dim:  int            = 128
    net_arch:         list[int]      = field(default_factory=lambda: [256, 256])

    # --- Discriminator architecture ---
    disc_hidden:      list[int]      = field(default_factory=lambda: [256, 256])

    # --- SAC hyper-parameters ---
    total_timesteps:  int            = 1_000_000
    learning_rate:    float          = 3e-4
    gamma:            float          = 0.99
    buffer_size:      int            = 300_000
    batch_size:       int            = 256
    tau:              float          = 0.005
    ent_coef:         str | float    = "auto"
    learning_starts:  int            = 10_000
    train_freq:       int            = 1
    gradient_steps:   int            = 1
    n_envs:           int            = 1

    # --- Discriminator training ---
    disc_lr:          float          = 3e-4
    disc_update_freq: int            = 1_000   # env steps between discriminator updates
    disc_batch_size:  int            = 256
    n_disc_updates:   int            = 5       # gradient steps per discriminator update (resamples each step)
    disc_grad_clip:   float          = 10.0    # max gradient norm for discriminator
    label_smoothing:  float          = 0.1     # expert labels = 1-ε, agent labels = ε
    agent_buf_size:   int            = 50_000  # agent ring-buffer capacity

    # --- BC warm-start (optional) ---
    bc_checkpoint:    str | None     = None

    # --- Logging / saving ---
    seed:             int            = 42
    run_name:         str            = ""
    save_dir:         str            = "runs_gail"
    log_dir:          str            = "/tmp/rl_logs"
    checkpoint_freq:  int            = 50_000
    n_eval_episodes:  int            = 5
    cache_dir:        str            = ".demo_cache"


# ---------------------------------------------------------------------------
# Numpy ring buffer — O(1) push and O(batch) sample
# ---------------------------------------------------------------------------

class _RingBuffer:
    """
    Fixed-capacity circular buffer backed by a pre-allocated numpy array.

    Unlike collections.deque, random-index access is O(1) because the backing
    store is a contiguous numpy array. This makes batch sampling fast even
    when the buffer is large (50k+ entries).
    """

    def __init__(self, capacity: int, dim: int):
        self._buf  = np.zeros((capacity, dim), dtype=np.float32)
        self._ptr  = 0
        self._full = False
        self._cap  = capacity

    def push(self, x: np.ndarray) -> None:
        self._buf[self._ptr] = x
        self._ptr = (self._ptr + 1) % self._cap
        if self._ptr == 0:
            self._full = True

    def sample(self, n: int) -> np.ndarray:
        """Return up to n rows sampled uniformly without replacement."""
        size = len(self)
        n    = min(n, size)
        idx  = np.random.choice(size, size=n, replace=False)
        return self._buf[idx]

    def __len__(self) -> int:
        return self._cap if self._full else self._ptr


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Binary classifier: (state, action) → logit ∈ ℝ.

    Positive class (label ≈ 1): expert demonstrations.
    Negative class (label ≈ 0): agent rollouts.

    Raw logit output — apply sigmoid externally (or use BCEWithLogitsLoss).
    """

    def __init__(
        self,
        state_dim:  int       = 110,
        action_dim: int       = 12,
        hidden:     list[int] = None,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns raw logit of shape (batch, 1)."""
        return self.net(torch.cat([state, action], dim=-1))


# ---------------------------------------------------------------------------
# GAIL reward function
# ---------------------------------------------------------------------------

# Normalisation constant: |log σ(-10)| ≈ 10.000045
_LOGIT_CLAMP  = 10.0
_REWARD_SCALE = abs(float(F.logsigmoid(torch.tensor(-_LOGIT_CLAMP))))


class GAILRewardFn(RewardFn):
    """
    Reward derived from the GAIL discriminator.

    r(s, a) = logsigmoid(clamp(D_logit, -10, 10)) / |logsigmoid(-10)|  ∈ [-1, 0]

    The logit clamp prevents -inf rewards when the discriminator is very
    confident (which would inject NaN into the SAC replay buffer).

    Also maintains a numpy ring-buffer of recent agent (state, action) pairs
    for the GAILCallback to sample during discriminator updates.
    """

    def __init__(
        self,
        discriminator: Discriminator,
        state_dim:     int = 110,
        action_dim:    int = 12,
        agent_buf_size: int = 50_000,
    ):
        self.discriminator   = discriminator
        self._state_buf  = _RingBuffer(agent_buf_size, state_dim)
        self._action_buf = _RingBuffer(agent_buf_size, action_dim)

    def __call__(self, env, obs: dict, action=None) -> float:
        if action is None:
            return 0.0

        state = obs.get("robot0_proprio-state")
        obj   = obs.get("object-state")
        if state is None or obj is None:
            return 0.0

        state_vec = np.concatenate([state, obj]).astype(np.float32)
        action_vec = np.asarray(action, dtype=np.float32).ravel()

        # Push into ring-buffer for discriminator training
        self._state_buf.push(state_vec)
        self._action_buf.push(action_vec)

        # Compute GAIL reward
        device = next(self.discriminator.parameters()).device
        with torch.no_grad():
            s_t    = torch.from_numpy(state_vec).unsqueeze(0).to(device)
            a_t    = torch.from_numpy(action_vec).unsqueeze(0).to(device)
            logit  = self.discriminator(s_t, a_t).squeeze()          # scalar
            logit  = logit.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
            reward = F.logsigmoid(logit).item()                       # ≤ 0

        return reward / _REWARD_SCALE   # normalise to [-1, 0]

    def sample_agent_batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample up to n (state, action) pairs from the agent ring-buffer."""
        n = min(n, len(self._state_buf))
        if n == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        # Both buffers advance in sync — same indices give matched pairs
        size = len(self._state_buf)
        idx  = np.random.choice(size, size=n, replace=False)
        return self._state_buf._buf[idx], self._action_buf._buf[idx]


# ---------------------------------------------------------------------------
# GAIL Callback — updates the discriminator during SAC training
# ---------------------------------------------------------------------------

class GAILCallback(BaseCallback):
    """
    SB3 callback that trains the GAIL discriminator.

    Every disc_update_freq environment steps:
      1. Run n_disc_updates gradient steps.
      2. Each step re-samples fresh batches from the expert buffer and the
         agent ring-buffer (avoids overfitting to a single batch).
      3. Label smoothing: expert labels = 1 - ε, agent labels = ε.
      4. Gradient clipping to prevent discriminator explosion.
    """

    def __init__(
        self,
        expert_states:    np.ndarray,
        expert_actions:   np.ndarray,
        discriminator:    Discriminator,
        gail_reward_fn:   GAILRewardFn,
        disc_lr:          float = 3e-4,
        disc_update_freq: int   = 1_000,
        disc_batch_size:  int   = 256,
        n_disc_updates:   int   = 5,
        disc_grad_clip:   float = 10.0,
        label_smoothing:  float = 0.1,
        verbose:          int   = 0,
    ):
        super().__init__(verbose=verbose)
        self.expert_states    = expert_states.astype(np.float32)
        self.expert_actions   = expert_actions.astype(np.float32)
        self.discriminator    = discriminator
        self.gail_reward_fn   = gail_reward_fn
        self.disc_update_freq = disc_update_freq
        self.disc_batch_size  = disc_batch_size
        self.n_disc_updates   = n_disc_updates
        self.disc_grad_clip   = disc_grad_clip
        self.label_smoothing  = label_smoothing

        self._optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)
        self._loss_fn   = nn.BCEWithLogitsLoss()
        self._n_updates = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.disc_update_freq == 0:
            self._update_discriminator()
        return True

    def _update_discriminator(self) -> None:
        n_agent = len(self.gail_reward_fn._state_buf)
        if n_agent < self.disc_batch_size // 4:
            return   # too few agent samples yet

        device      = next(self.discriminator.parameters()).device
        batch_size  = min(self.disc_batch_size, n_agent)
        n_expert    = len(self.expert_states)

        eps         = self.label_smoothing
        last_loss   = 0.0
        last_e_acc  = 0.0
        last_a_acc  = 0.0

        for _ in range(self.n_disc_updates):
            # --- Resample every step to avoid batch overfitting ---
            e_idx = np.random.choice(n_expert, size=batch_size,
                                     replace=(n_expert < batch_size))
            e_states  = torch.from_numpy(self.expert_states[e_idx]).to(device)
            e_actions = torch.from_numpy(self.expert_actions[e_idx]).to(device)

            a_s, a_a  = self.gail_reward_fn.sample_agent_batch(batch_size)
            a_states  = torch.from_numpy(a_s).to(device)
            a_actions = torch.from_numpy(a_a).to(device)

            # --- Forward ---
            e_logits = self.discriminator(e_states,  e_actions)   # (B, 1)
            a_logits = self.discriminator(a_states,  a_actions)   # (B, 1)

            # Label smoothing: expert → 1-ε, agent → ε
            e_labels = torch.full_like(e_logits, 1.0 - eps)
            a_labels = torch.full_like(a_logits, eps)

            loss = self._loss_fn(
                torch.cat([e_logits, a_logits]),
                torch.cat([e_labels, a_labels]),
            )

            # --- Backward with gradient clipping ---
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_grad_clip)
            self._optimizer.step()

            last_loss  = loss.item()
            last_e_acc = (torch.sigmoid(e_logits.detach()) > 0.5).float().mean().item()
            last_a_acc = (torch.sigmoid(a_logits.detach()) < 0.5).float().mean().item()

        self._n_updates += 1
        if self.verbose >= 1 and self._n_updates % 10 == 0:
            print(
                f"[disc] step={self.num_timesteps:>8,}  "
                f"loss={last_loss:.4f}  "
                f"expert_acc={last_e_acc:.2f}  agent_acc={last_a_acc:.2f}  "
                f"agent_buf={n_agent:,}  updates={self._n_updates}"
            )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_gail(cfg: GAILConfig) -> Any:
    """
    Build environments, train a GAIL discriminator alongside SAC, and return
    the trained SB3 model.

    Steps
    ─────
      1. Collect expert (state, action) pairs from demonstrations.
      2. Build Discriminator and GAILRewardFn.
      3. Build training env (GAIL reward) and eval env (task reward).
      4. Instantiate SAC (optionally warm-started from BC).
      5. Run model.learn() with GAILCallback updating the discriminator.
    """
    from stable_baselines3 import SAC

    from il.base import ILConfig, collect_demo_data
    from il.bc import warm_start_from_bc
    from rl.architecture import RoboCasaFeaturesExtractor
    from rl.reward import StagedPickPlaceReward
    from rl.trainer import TrainConfig, _make_env

    # ------------------------------------------------------------------
    # 1. Expert data
    # ------------------------------------------------------------------
    il_cfg = ILConfig(
        task      = cfg.task,
        split     = cfg.split,
        source    = cfg.source,
        n_demos   = cfg.n_demos,
        cache_dir = cfg.cache_dir,
    )
    expert_states, expert_actions = collect_demo_data(il_cfg)
    state_dim  = expert_states.shape[1]
    action_dim = expert_actions.shape[1]
    print(f"[gail] Expert data: {len(expert_states):,} transitions  "
          f"state_dim={state_dim}  action_dim={action_dim}")

    # ------------------------------------------------------------------
    # 2. Discriminator + GAIL reward
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc   = Discriminator(
        state_dim  = state_dim,
        action_dim = action_dim,
        hidden     = cfg.disc_hidden,
    ).to(device)

    gail_reward = GAILRewardFn(
        discriminator  = disc,
        state_dim      = state_dim,
        action_dim     = action_dim,
        agent_buf_size = cfg.agent_buf_size,
    )

    # ------------------------------------------------------------------
    # 3. Environments
    # ------------------------------------------------------------------
    env_cfg = TrainConfig(
        env_name        = cfg.env_name,
        layout_ids      = cfg.layout_ids,
        style_ids       = cfg.style_ids,
        horizon         = cfg.horizon,
        control_freq    = cfg.control_freq,
        use_camera_obs  = False,
        state_embed_dim = cfg.state_embed_dim,
        image_embed_dim = cfg.image_embed_dim,
        n_envs          = cfg.n_envs,
        log_dir         = cfg.log_dir,
        seed            = cfg.seed,
    )

    env_fns   = [_make_env(env_cfg, gail_reward, rank=i) for i in range(cfg.n_envs)]
    train_env = SubprocVecEnv(env_fns) if cfg.n_envs > 1 else DummyVecEnv(env_fns)

    # Eval uses the task reward so EvalCallback measures real success, not adversarial reward
    eval_env  = DummyVecEnv([_make_env(env_cfg, StagedPickPlaceReward(), rank=99, eval_mode=True)])

    # ------------------------------------------------------------------
    # 4. SAC model
    # ------------------------------------------------------------------
    run_name  = cfg.run_name or f"{cfg.env_name}_gail_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class  = RoboCasaFeaturesExtractor,
        features_extractor_kwargs = dict(
            state_embed_dim = cfg.state_embed_dim,
            image_embed_dim = cfg.image_embed_dim,
        ),
        net_arch = cfg.net_arch,
    )

    model = SAC(
        policy          = "MultiInputPolicy",
        env             = train_env,
        learning_rate   = cfg.learning_rate,
        gamma           = cfg.gamma,
        buffer_size     = cfg.buffer_size,
        batch_size      = cfg.batch_size,
        tau             = cfg.tau,
        ent_coef        = cfg.ent_coef,
        learning_starts = cfg.learning_starts,
        train_freq      = cfg.train_freq,
        gradient_steps  = cfg.gradient_steps,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        seed            = cfg.seed,
        tensorboard_log = os.path.join(cfg.log_dir, "tb"),
    )

    if cfg.bc_checkpoint is not None:
        warm_start_from_bc(
            model,
            bc_checkpoint = cfg.bc_checkpoint,
            embed_dim     = cfg.state_embed_dim,
            net_arch      = cfg.net_arch,
        )

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    save_freq = max(cfg.checkpoint_freq // cfg.n_envs, 1)
    gail_cb   = GAILCallback(
        expert_states    = expert_states,
        expert_actions   = expert_actions,
        discriminator    = disc,
        gail_reward_fn   = gail_reward,
        disc_lr          = cfg.disc_lr,
        disc_update_freq = cfg.disc_update_freq,
        disc_batch_size  = cfg.disc_batch_size,
        n_disc_updates   = cfg.n_disc_updates,
        disc_grad_clip   = cfg.disc_grad_clip,
        label_smoothing  = cfg.label_smoothing,
        verbose          = 1,
    )
    callbacks = CallbackList([
        gail_cb,
        CheckpointCallback(
            save_freq   = save_freq,
            save_path   = os.path.join(save_path, "checkpoints"),
            name_prefix = "gail_sac",
            verbose     = 1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path = os.path.join(save_path, "best"),
            log_path             = os.path.join(cfg.log_dir, "eval"),
            eval_freq            = save_freq,
            n_eval_episodes      = cfg.n_eval_episodes,
            deterministic        = True,
            verbose              = 1,
        ),
    ])

    print(f"[gail] Run:             {run_name}")
    print(f"[gail] Save path:       {save_path}")
    print(f"[gail] TensorBoard:     tensorboard --logdir {os.path.join(cfg.log_dir, 'tb')}")
    print(f"[gail] Expert size:     {len(expert_states):,} transitions")
    print(f"[gail] Total timesteps: {cfg.total_timesteps:,}")
    print(f"[gail] BC warm-start:   {cfg.bc_checkpoint or 'none'}")
    print(f"[gail] Device:          {device}")

    model.learn(
        total_timesteps = cfg.total_timesteps,
        callback        = callbacks,
        progress_bar    = True,
    )

    final_path = os.path.join(save_path, "gail_sac_final")
    model.save(final_path)
    torch.save(disc.state_dict(), os.path.join(save_path, "discriminator_final.pt"))
    print(f"[gail] Model saved → {final_path}.zip")
    print(f"[gail] Disc  saved → {save_path}/discriminator_final.pt")

    train_env.close()
    eval_env.close()
    return model
