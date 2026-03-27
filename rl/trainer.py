"""
Generalised RL trainer for RoboCasa environments.

Supports any SB3 algorithm (SAC, PPO, TD3, DDPG, A2C, …) via an algorithm
registry.  Algorithm-specific hyperparameters are passed through algo_kwargs
so the config stays clean regardless of which algorithm is chosen.

Entry point
───────────
    from rl import TrainConfig, train
    from rl.reward import StagedPickPlaceReward

    # SAC (default)
    train(TrainConfig(), reward_fn=StagedPickPlaceReward())

    # PPO — override only the params that differ
    train(
        TrainConfig(
            algo="PPO",
            n_envs=8,              # PPO benefits from parallel envs
            algo_kwargs=dict(
                n_steps=2048,
                clip_range=0.2,
                ent_coef=0.01,
            ),
        ),
        reward_fn=StagedPickPlaceReward(),
    )

    # TD3
    train(
        TrainConfig(
            algo="TD3",
            algo_kwargs=dict(
                buffer_size=300_000,
                learning_starts=10_000,
                action_noise=None,  # or OrnsteinUhlenbeckActionNoise(...)
            ),
        ),
    )

Algorithm registry
──────────────────
    Add a new algorithm with:
        from rl.trainer import ALGO_REGISTRY
        from stable_baselines3 import MyAlgo
        ALGO_REGISTRY["MyAlgo"] = MyAlgo
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Type

from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .architecture import RoboCasaFeaturesExtractor
from .env_wrapper import RoboCasaWrapper
from .reward import RewardFn, StagedPickPlaceReward


# ---------------------------------------------------------------------------
# Registries — extend these to add new envs or algorithms
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, Type] = {
    "PickPlaceCounterToCabinet": PickPlaceCounterToCabinet,
}

ALGO_REGISTRY: dict[str, Type[BaseAlgorithm]] = {
    "SAC":  SAC,
    "TD3":  TD3,
    "DDPG": DDPG,
    "PPO":  PPO,
    "A2C":  A2C,
}

# MultiInputPolicy lives under each algorithm's own sub-module in SB3.
_MULTI_INPUT_POLICY: dict[str, str] = {
    "SAC":  "MultiInputPolicy",
    "TD3":  "MultiInputPolicy",
    "DDPG": "MultiInputPolicy",
    "PPO":  "MultiInputPolicy",
    "A2C":  "MultiInputPolicy",
}

# Default algo_kwargs for each algorithm.
# Users override these via TrainConfig.algo_kwargs.
_ALGO_DEFAULTS: dict[str, dict[str, Any]] = {
    "SAC": dict(
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        ent_coef="auto",
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
    ),
    "TD3": dict(
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
    ),
    "DDPG": dict(
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
    ),
    "PPO": dict(
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.0,
        gae_lambda=0.95,
    ),
    "A2C": dict(
        n_steps=5,
        ent_coef=0.0,
        gae_lambda=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # --- Algorithm ---
    algo:             str             = "SAC"
    algo_kwargs:      dict[str, Any]  = field(default_factory=dict)
    # algo_kwargs are merged on top of _ALGO_DEFAULTS[algo], so you only need
    # to specify values that differ from the defaults shown above.

    # --- Environment ---
    env_name:         str             = "PickPlaceCounterToCabinet"
    layout_ids:       int | list      = -2    # -2 = all train layouts
    style_ids:        int | list      = -2    # -2 = all train styles
    horizon:          int             = 500
    control_freq:     int             = 20

    # --- Observation ---
    use_camera_obs:   bool            = False
    camera_names:     list[str]       = field(default_factory=lambda: [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ])
    image_size:       int             = 64

    # --- Architecture ---
    state_embed_dim:  int             = 256
    image_embed_dim:  int             = 128
    net_arch:         list[int]       = field(default_factory=lambda: [256, 256])

    # --- Common hyperparameters (apply to all algorithms) ---
    seed:             int             = 42
    total_timesteps:  int             = 1_000_000
    learning_rate:    float           = 3e-4
    gamma:            float           = 0.99

    # --- Parallelism ---
    # PPO/A2C benefit from n_envs > 1; off-policy algos typically use 1.
    n_envs:           int             = 1

    # --- Logging / saving ---
    run_name:         str             = ""        # auto-generated if empty
    save_dir:         str             = "runs"
    log_dir:          str             = "/tmp/rl_logs"
    checkpoint_freq:  int             = 50_000
    n_eval_episodes:  int             = 5


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def _make_raw_env(cfg: TrainConfig, rank: int, eval_mode: bool = False):
    env_cls = ENV_REGISTRY[cfg.env_name]
    ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")

    layout_ids = cfg.layout_ids if not eval_mode else -1
    style_ids  = cfg.style_ids  if not eval_mode else -1

    env = env_cls(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=cfg.use_camera_obs,
        has_renderer=False,
        has_offscreen_renderer=cfg.use_camera_obs,
        use_object_obs=True,
        camera_names=cfg.camera_names if cfg.use_camera_obs else [],
        camera_heights=cfg.image_size,
        camera_widths=cfg.image_size,
        control_freq=cfg.control_freq,
        ignore_done=False,
        seed=cfg.seed + rank,
        horizon=cfg.horizon,
        layout_ids=layout_ids,
        style_ids=style_ids,
    )
    env.reset()
    return env


def _make_env(cfg: TrainConfig, reward_fn: RewardFn, rank: int, eval_mode: bool = False):
    def _init():
        raw = _make_raw_env(cfg, rank, eval_mode=eval_mode)
        wrapped = RoboCasaWrapper(
            env=raw,
            reward_fn=reward_fn,
            use_camera_obs=cfg.use_camera_obs,
            camera_names=cfg.camera_names,
            image_size=cfg.image_size,
        )
        log_dir = os.path.join(cfg.log_dir, "envs", str(rank))
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(wrapped, log_dir)
    return _init


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, reward_fn: RewardFn | None = None) -> BaseAlgorithm:
    """
    Build environments, instantiate the chosen algorithm, and run training.

    Args:
        cfg:       Training configuration.
        reward_fn: Reward function. Defaults to StagedPickPlaceReward.

    Returns:
        The trained SB3 model.
    """
    if cfg.algo not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo '{cfg.algo}'. Available: {list(ALGO_REGISTRY.keys())}")

    if reward_fn is None:
        reward_fn = StagedPickPlaceReward()

    algo_cls  = ALGO_REGISTRY[cfg.algo]
    run_name  = cfg.run_name or f"{cfg.env_name}_{cfg.algo.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    # --- Vectorised training env ---
    env_fns   = [_make_env(cfg, reward_fn, rank=i) for i in range(cfg.n_envs)]
    train_env = SubprocVecEnv(env_fns) if cfg.n_envs > 1 else DummyVecEnv(env_fns)

    # --- Eval env (single, test distribution) ---
    eval_env  = DummyVecEnv([_make_env(cfg, reward_fn, rank=99, eval_mode=True)])

    # --- Policy kwargs ---
    policy_kwargs = dict(
        features_extractor_class=RoboCasaFeaturesExtractor,
        features_extractor_kwargs=dict(
            state_embed_dim=cfg.state_embed_dim,
            image_embed_dim=cfg.image_embed_dim,
        ),
        net_arch=cfg.net_arch,
    )

    # --- Merge algo-specific kwargs: defaults ← user overrides ---
    merged_algo_kwargs = {**_ALGO_DEFAULTS.get(cfg.algo, {}), **cfg.algo_kwargs}

    # --- Instantiate model ---
    model = algo_cls(
        policy=_MULTI_INPUT_POLICY[cfg.algo],
        env=train_env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=os.path.join(cfg.log_dir, "tb"),
        **merged_algo_kwargs,
    )

    # --- Callbacks ---
    save_freq = max(cfg.checkpoint_freq // cfg.n_envs, 1)
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(save_path, "checkpoints"),
            name_prefix=cfg.algo.lower(),
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_path, "best"),
            log_path=os.path.join(cfg.log_dir, "eval"),
            eval_freq=save_freq,
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ),
    ])

    print(f"Algorithm:       {cfg.algo}")
    print(f"Run:             {run_name}")
    print(f"Save path:       {save_path}")
    print(f"TensorBoard:     tensorboard --logdir {os.path.join(cfg.log_dir, 'tb')}")
    print(f"Camera obs:      {cfg.use_camera_obs}")
    print(f"Reward fn:       {type(reward_fn).__name__}")
    print(f"n_envs:          {cfg.n_envs}")
    print(f"Total timesteps: {cfg.total_timesteps:,}")
    print(f"Algo kwargs:     {merged_algo_kwargs}")

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(save_path, f"{cfg.algo.lower()}_final")
    model.save(final_path)
    print(f"Model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()
    return model
