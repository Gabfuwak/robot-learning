"""
PPO trainer for RoboCasa environments using Stable Baselines 3.

Entry point
───────────
    from rl import TrainConfig, train
    from rl.reward import StagedPickPlaceReward

    train(TrainConfig(), reward_fn=StagedPickPlaceReward())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Type

from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .architecture import RoboCasaFeaturesExtractor
from .callbacks import CurriculumCallback, StageSuccessCallback
from .env_wrapper import RoboCasaWrapper
from .reward import RewardFn, StagedPickPlaceReward


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, Type] = {
    "PickPlaceCounterToCabinet": PickPlaceCounterToCabinet,
}

_PPO_DEFAULTS: dict[str, Any] = dict(
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    gae_lambda=0.95,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # --- Algorithm ---
    algo_kwargs:      dict[str, Any]  = field(default_factory=dict)
    # Merged on top of _PPO_DEFAULTS — only specify values that differ.

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
    include_cab_obs:  bool            = False   # append env.cab.pos (3 floats) to state for Stage 1

    # --- Curriculum ---
    # Set curriculum_init_dist > 0 to enable spawn-distance curriculum.
    # max_spawn_dist grows by curriculum_epsilon for every training episode
    # where the object is successfully grasped, up to curriculum_max_dist.
    curriculum_init_dist: float       = 0.0     # 0 = disabled (no curriculum)
    curriculum_epsilon:   float       = 0.05    # scaling factor: delta = epsilon * current_dist / n_envs
    curriculum_max_dist:  float       = 0.50    # cap (metres); ~full counter range

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
    n_envs:           int             = 16   # PPO benefits from parallel collection

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
            include_cab_obs=cfg.include_cab_obs,
            max_spawn_dist=cfg.curriculum_init_dist if cfg.curriculum_init_dist > 0 else None,
        )
        log_dir = os.path.join(cfg.log_dir, "envs", str(rank))
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(wrapped, log_dir)
    return _init


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, reward_fn: RewardFn | None = None) -> PPO:
    """
    Build environments, instantiate PPO, and run training.

    Args:
        cfg:       Training configuration.
        reward_fn: Reward function. Defaults to StagedPickPlaceReward.

    Returns:
        The trained SB3 PPO model.
    """
    if reward_fn is None:
        reward_fn = StagedPickPlaceReward()

    run_name  = cfg.run_name or f"{cfg.env_name}_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # --- Merge PPO kwargs: defaults ← user overrides ---
    merged_algo_kwargs = {**_PPO_DEFAULTS, **cfg.algo_kwargs}

    # --- Instantiate model ---
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=os.path.join(cfg.log_dir, "tb"),
        **merged_algo_kwargs,
    )

    # --- Eval env for stage-success callback (separate from EvalCallback's env) ---
    stage_eval_env = DummyVecEnv([_make_env(cfg, reward_fn, rank=98, eval_mode=True)])

    # --- Curriculum callback + curriculum-difficulty eval env (optional) ---
    curriculum_cb       = None
    curriculum_eval_env = None
    if cfg.curriculum_init_dist > 0:
        curriculum_cb = CurriculumCallback(
            init_dist=cfg.curriculum_init_dist,
            epsilon=cfg.curriculum_epsilon,
            max_dist=cfg.curriculum_max_dist,
        )
        curriculum_eval_env = DummyVecEnv([_make_env(cfg, reward_fn, rank=97, eval_mode=True)])

    # --- Callbacks ---
    save_freq = max(cfg.checkpoint_freq // cfg.n_envs, 1)
    cb_list = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(save_path, "checkpoints"),
            name_prefix="ppo",
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
        StageSuccessCallback(
            eval_env=stage_eval_env,
            eval_freq=save_freq,
            n_eval_episodes=cfg.n_eval_episodes,
            curriculum_callback=curriculum_cb,
            curriculum_eval_env=curriculum_eval_env,
            verbose=1,
        ),
    ]
    if curriculum_cb is not None:
        cb_list.append(curriculum_cb)
    callbacks = CallbackList(cb_list)

    print(f"Run:             {run_name}")
    print(f"Save path:       {save_path}")
    print(f"TensorBoard:     tensorboard --logdir {os.path.join(cfg.log_dir, 'tb')}")
    print(f"Camera obs:      {cfg.use_camera_obs}")
    print(f"Reward fn:       {type(reward_fn).__name__}")
    print(f"n_envs:          {cfg.n_envs}")
    if cfg.curriculum_init_dist > 0:
        print(f"Curriculum:      init={cfg.curriculum_init_dist}m  eps={cfg.curriculum_epsilon}m  cap={cfg.curriculum_max_dist}m")
    print(f"Total timesteps: {cfg.total_timesteps:,}")
    print(f"PPO kwargs:      {merged_algo_kwargs}")

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(save_path, "ppo_final")
    model.save(final_path)
    print(f"Model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()
    stage_eval_env.close()
    if curriculum_eval_env is not None:
        curriculum_eval_env.close()
    return model
