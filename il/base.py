"""
Shared base for all Imitation Learning algorithms.

This module contains the pieces that are algorithm-agnostic and reused across
BC, GAIL, Diffusion Policy, and any future IL method:

    ILConfig              — common hyperparameters (dataset, env, persistence)
    collect_demo_data()   — replay demos through sim → flat (states, actions) arrays
    collect_episode_data()— same, but returns per-episode pairs for sequence models

Internal helpers (not part of the public API):
    _make_replay_env()    — rebuild RoboCasa env from dataset metadata
    _reset_to()           — restore a recorded MuJoCo sim state
    _extract_state_vec()  — concatenate proprio + object obs → 110-dim float32
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

@dataclass
class ILConfig:
    """
    Hyperparameters shared across all IL algorithms.

    Fields used by data collection, evaluation, and persistence are defined
    here. Algorithm-specific fields (architecture, training schedule, etc.)
    live in the per-algorithm config classes (e.g. BCConfig, GAILConfig).
    """

    # --- Dataset ---
    task:    str           = "PickPlaceCounterToCabinet"
    split:   str           = "target"   # "pretrain" | "target"
    source:  str           = "human"    # "human" | "mg"
    n_demos: Optional[int] = None       # None = use all available demos

    # --- Environment (evaluation only) ---
    layout_ids:   int = -1   # -1 = test layouts (1–10)
    style_ids:    int = -1   # -1 = test styles  (1–10)
    horizon:      int = 500
    control_freq: int = 20

    # --- Architecture (used by BCPolicy and warm_start_from_bc) ---
    state_embed_dim: int       = 256
    net_arch:        list[int] = None   # default set in __post_init__

    # --- Training ---
    n_epochs:     int   = 100
    batch_size:   int   = 256
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    grad_clip:    float = 1.0
    seed:         int   = 42

    # --- Evaluation ---
    n_eval_episodes: int = 10

    # --- Persistence ---
    save_dir:  str = "runs_il"
    cache_dir: str = ".demo_cache"

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [256, 256]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_replay_env(env_meta: dict):
    """
    Rebuild the RoboCasa environment from dataset metadata for sim-state replay.
    Offscreen renderer and camera obs are disabled to keep data collection fast.
    """
    import robosuite

    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"]               = env_meta["env_name"]
    env_kwargs["has_renderer"]           = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"]         = False
    env_kwargs["use_object_obs"]         = True
    return robosuite.make(**env_kwargs)


def _reset_to(env, state: dict) -> None:
    """
    Load a recorded sim state (and optionally model XML + ep_meta) into the env.
    Mirrors robocasa.scripts.dataset_scripts.playback_dataset.reset_to.
    """
    import robosuite

    if "model" in state:
        ep_meta = json.loads(state["ep_meta"]) if state.get("ep_meta") else {}
        if hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        elif hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        env.reset()
        robosuite_minor = int(robosuite.__version__.split(".")[1])
        if robosuite_minor <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()

    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()


def _extract_state_vec(raw_obs: dict) -> np.ndarray:
    """
    Concatenate robot0_proprio-state (68) + object-state (42) → 110-dim float32.
    Matches exactly what RoboCasaWrapper returns as obs["state"].
    """
    keys  = ["robot0_proprio-state", "object-state"]
    parts = [raw_obs[k].astype(np.float32) for k in keys if k in raw_obs]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Public data collection
# ---------------------------------------------------------------------------

def collect_demo_data(cfg: ILConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Replay every demonstration through the simulator using recorded MuJoCo states
    to collect full (110-dim state, 12-dim action) pairs.

    The parquet files only store a 16-dim low-level state (no object positions),
    so replay is necessary to obtain the full observation available at test time.

    Results are cached on disk so subsequent runs skip the expensive replay step.

    Returns
    -------
    states  : (N, STATE_DIM) float32
    actions : (N, 12)        float32
    """
    import robocasa.utils.lerobot_utils as LU
    from robocasa.utils.dataset_registry_utils import get_ds_path

    dataset_path = get_ds_path(cfg.task, source=cfg.source, split=cfg.split)
    if dataset_path is None or not Path(dataset_path).exists():
        raise RuntimeError(
            f"Dataset not found at: {dataset_path}\n"
            f"Download it with:\n"
            f"  python -m robocasa.scripts.download_datasets "
            f"--tasks {cfg.task} --split {cfg.split} --source {cfg.source}"
        )
    dataset_dir = Path(dataset_path)

    # ---- Cache check --------------------------------------------------------
    cache_key     = f"{cfg.task}_{cfg.split}_{cfg.source}_{cfg.n_demos}"
    cache_states  = Path(cfg.cache_dir) / f"{cache_key}_states.npy"
    cache_actions = Path(cfg.cache_dir) / f"{cache_key}_actions.npy"
    if cache_states.exists() and cache_actions.exists():
        print(f"[data] Loading cached demo data from {cfg.cache_dir}/")
        return np.load(cache_states), np.load(cache_actions)

    # ---- Build env and iterate episodes -------------------------------------
    env_meta = LU.get_env_metadata(dataset_dir)
    env      = _make_replay_env(env_meta)

    episodes = LU.get_episodes(dataset_dir)
    if cfg.n_demos is not None:
        episodes = episodes[: cfg.n_demos]
    n_ep = len(episodes)

    all_states:  list[np.ndarray] = []
    all_actions: list[np.ndarray] = []

    print(f"[data] Replaying {n_ep} demonstrations to collect observations ...")
    for ind in range(n_ep):
        if (ind + 1) % max(1, n_ep // 10) == 0:
            print(f"  Episode {ind + 1}/{n_ep}")

        sim_states = LU.get_episode_states(dataset_dir, ind)   # (T, mujoco_dim)
        actions    = LU.get_episode_actions(dataset_dir, ind)  # (T, 12)
        xml        = LU.get_episode_model_xml(dataset_dir, ind)
        ep_meta    = LU.get_episode_meta(dataset_dir, ind)
        T          = len(actions)

        # Restore this episode's kitchen layout and object placement
        _reset_to(env, {
            "model":   xml,
            "states":  sim_states[0],
            "ep_meta": json.dumps(ep_meta),
        })

        ep_states: list[np.ndarray] = []
        for t in range(T):
            # force_update=True: manually setting sim state does not automatically
            # refresh the observable cache — without this every timestep returns
            # the same frozen observation from the initial reset.
            _reset_to(env, {"states": sim_states[t]})
            raw_obs = env._get_observations(force_update=True)
            ep_states.append(_extract_state_vec(raw_obs))

        all_states.extend(ep_states)
        all_actions.append(actions)

    env.close()

    states_arr  = np.array(all_states,                         dtype=np.float32)
    actions_arr = np.concatenate(all_actions, axis=0).astype(np.float32)

    os.makedirs(cfg.cache_dir, exist_ok=True)
    np.save(cache_states,  states_arr)
    np.save(cache_actions, actions_arr)
    print(f"[data] Cached {len(states_arr):,} transitions → {cfg.cache_dir}/")

    return states_arr, actions_arr


def collect_episode_data(cfg: ILConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Like collect_demo_data but returns per-episode (states, actions) pairs
    instead of flat concatenated arrays. Required by sequence models such as
    Diffusion Policy that need temporal structure within each episode.

    Reuses the flat cache from collect_demo_data — episode lengths are read
    from the parquet action files (fast, no sim replay needed) and used to
    split the cached flat arrays.

    Returns
    -------
    list of (states, actions) per episode:
        states  : (T, STATE_DIM) float32
        actions : (T, 12)        float32
    """
    import robocasa.utils.lerobot_utils as LU
    from robocasa.utils.dataset_registry_utils import get_ds_path

    flat_states, flat_actions = collect_demo_data(cfg)

    dataset_path = get_ds_path(cfg.task, source=cfg.source, split=cfg.split)
    dataset_dir  = Path(dataset_path)

    episodes = LU.get_episodes(dataset_dir)
    if cfg.n_demos is not None:
        episodes = episodes[: cfg.n_demos]

    ep_lengths = [len(LU.get_episode_actions(dataset_dir, i)) for i in range(len(episodes))]

    result = []
    offset = 0
    for T in ep_lengths:
        result.append((flat_states[offset: offset + T], flat_actions[offset: offset + T]))
        offset += T
    return result
