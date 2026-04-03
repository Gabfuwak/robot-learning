"""
PPO trainer for RoboCasa environments using Stable Baselines 3.

Entry point
───────────
    from rl import TrainConfig, train
    from rl.reward import StagedPickPlaceReward

    train(TrainConfig(), reward_fn=StagedPickPlaceReward())
"""

from __future__ import annotations

import dataclasses
import json
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
from .lewm_extractor import LeWMFeaturesExtractor
from .callbacks import CurriculumCallback, RollingSuccessCallback, StageSuccessCallback
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

    # --- LeWM encoder (optional) ---
    # If set, uses a frozen LeWM encoder instead of the CNN+state extractor.
    # Requires use_camera_obs=True and camera_names to include
    # "robot0_agentview_left" and "robot0_eye_in_hand".
    lewm_checkpoint:    str           = ""       # e.g. "lewm_epoch_84" (no suffix)
    lewm_embed_dim:     int           = 256
    lewm_proprio_slice: tuple         = (7, 16)  # indices into state vector

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

    # --- Weights & Biases ---
    wandb_enabled:    bool            = False
    wandb_entity:     str             = "gbwk-proj"
    wandb_project:    str             = "robocasa-pickplace"


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

def train(cfg: TrainConfig, reward_fn: RewardFn | None = None, resume_from: str | None = None) -> PPO:
    """
    Build environments, instantiate PPO, and run training.

    Args:
        cfg:         Training configuration.
        reward_fn:   Reward function. Defaults to StagedPickPlaceReward.
        resume_from: Path to a checkpoint .zip to resume from (e.g.
                     "runs/.../checkpoints/ppo_750000_steps.zip").
                     Timestep count and training state are preserved.

    Returns:
        The trained SB3 PPO model.
    """
    if reward_fn is None:
        reward_fn = StagedPickPlaceReward()

    run_name  = cfg.run_name or f"{cfg.env_name}_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    # --- Weights & Biases ---
    _wandb_run = None
    if cfg.wandb_enabled:
        import wandb

        # Re-attach to existing run if resuming
        _wandb_run_id = None
        if resume_from:
            _orig_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(resume_from)))
            _rid_file = os.path.join(_orig_run_dir, "wandb_run_id.txt")
            if os.path.exists(_rid_file):
                with open(_rid_file) as _f:
                    _wandb_run_id = _f.read().strip()

        _cfg_dict = {
            k: v for k, v in dataclasses.asdict(cfg).items()
            if not k.startswith("wandb_")
        }
        _cfg_dict["reward_fn"] = type(reward_fn).__name__

        _wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_name,
            config=_cfg_dict,
            id=_wandb_run_id,
            resume="must" if _wandb_run_id else None,
            sync_tensorboard=True,
        )
        with open(os.path.join(save_path, "wandb_run_id.txt"), "w") as _f:
            _f.write(_wandb_run.id)
        print(f"W&B run:         {_wandb_run.url}")

    # --- Vectorised training env ---
    env_fns   = [_make_env(cfg, reward_fn, rank=i) for i in range(cfg.n_envs)]
    train_env = SubprocVecEnv(env_fns) if cfg.n_envs > 1 else DummyVecEnv(env_fns)

    # --- Eval env (single, test distribution) ---
    eval_env  = DummyVecEnv([_make_env(cfg, reward_fn, rank=99, eval_mode=True)])

    # --- Policy kwargs ---
    if cfg.lewm_checkpoint:
        assert cfg.use_camera_obs, "lewm_checkpoint requires use_camera_obs=True"
        assert "robot0_agentview_left" in cfg.camera_names, \
            "lewm_checkpoint requires 'robot0_agentview_left' in camera_names"
        assert "robot0_eye_in_hand" in cfg.camera_names, \
            "lewm_checkpoint requires 'robot0_eye_in_hand' in camera_names"
        policy_kwargs = dict(
            features_extractor_class=LeWMFeaturesExtractor,
            features_extractor_kwargs=dict(
                checkpoint=cfg.lewm_checkpoint,
                embed_dim=cfg.lewm_embed_dim,
                proprio_slice=cfg.lewm_proprio_slice,
            ),
            net_arch=cfg.net_arch,
        )
    else:
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
    if resume_from:
        print(f"Resuming from:   {resume_from}")
        model = PPO.load(
            resume_from,
            env=train_env,
            tensorboard_log=os.path.join(cfg.log_dir, "tb"),
        )
    else:
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
        # Determine starting dist: resume from saved state, or max_dist if none found
        if resume_from:
            run_dir    = os.path.dirname(os.path.dirname(os.path.abspath(resume_from)))
            state_file = os.path.join(run_dir, CurriculumCallback.STATE_FILE)
            if os.path.exists(state_file):
                with open(state_file) as _f:
                    _resume_dist = json.load(_f)["current_dist"]
                print(f"Curriculum state: resuming at current_dist={_resume_dist:.3f}m  (from {state_file})")
            else:
                _resume_dist = cfg.curriculum_max_dist
                print(f"Curriculum state: no file found — starting at max_dist={_resume_dist:.3f}m")
            curriculum_init = _resume_dist
        else:
            curriculum_init = cfg.curriculum_init_dist

        curriculum_cb = CurriculumCallback(
            init_dist=curriculum_init,
            epsilon=cfg.curriculum_epsilon,
            max_dist=cfg.curriculum_max_dist,
            save_path=save_path,
        )
        curriculum_eval_env = DummyVecEnv([_make_env(cfg, reward_fn, rank=97, eval_mode=True)])

    # --- Callbacks ---
    save_freq = max(cfg.checkpoint_freq // cfg.n_envs, 1)
    cb_list = [
        RollingSuccessCallback(window=100),
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

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=resume_from is None,
    )

    final_path = os.path.join(save_path, "ppo_final")
    model.save(final_path)
    print(f"Model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()
    stage_eval_env.close()
    if curriculum_eval_env is not None:
        curriculum_eval_env.close()
    if _wandb_run is not None:
        _wandb_run.finish()
    return model
