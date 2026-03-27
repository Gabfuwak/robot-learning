"""
Gymnasium wrapper around a raw RoboCasa environment.

Responsibilities:
  - Exposes a Dict observation space containing a flat state vector and,
    optionally, one CHW uint8 image tensor per camera.
  - Replaces the environment's built-in reward with any RewardFn.
  - Handles the gymnasium (reset → (obs, info), step → (obs, r, term, trunc, info))
    API on top of robosuite's older (reset → obs, step → (obs, r, done, info)) API.

Observation space
─────────────────
  "state"               float32  (STATE_DIM,)          always present
  "image_<camera_name>" uint8    (3, H, W)  per camera  only if use_camera_obs=True

The state vector concatenates robot0_proprio-state (68) and object-state (42) by
default, giving STATE_DIM=110 for PandaOmron on PickPlaceCounterToCabinet.

Action space
────────────
  Box(-1, 1, shape=(12,), dtype=float32)  — full 12-dim action as described in
  obs_action_space.md.
"""

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import RewardFn


class RoboCasaWrapper(gym.Env):
    """
    Args:
        env:             Raw RoboCasa environment (not yet wrapped by GymWrapper).
        reward_fn:       A RewardFn instance called each step to produce the reward.
        use_camera_obs:  Whether to include camera images in the observation.
        camera_names:    List of camera names to include when use_camera_obs=True.
                         Defaults to the three standard training cameras.
        image_size:      Height and width to resize each image to (square).
        state_keys:      Observation keys to concatenate into the "state" vector.
                         Defaults to the two aggregated keys.
    """

    DEFAULT_CAMERAS = [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ]
    DEFAULT_STATE_KEYS = ["robot0_proprio-state", "object-state"]

    def __init__(
        self,
        env,
        reward_fn: RewardFn,
        use_camera_obs: bool = False,
        camera_names: list[str] | None = None,
        image_size: int = 64,
        state_keys: list[str] | None = None,
    ):
        super().__init__()
        self._env = env
        self._reward_fn = reward_fn
        self._use_camera_obs = use_camera_obs
        self._camera_names = camera_names if camera_names is not None else self.DEFAULT_CAMERAS
        self._image_size = image_size
        self._state_keys = state_keys if state_keys is not None else self.DEFAULT_STATE_KEYS

        # Derive state dim from a throw-away reset so we don't hard-code 110.
        raw = self._env.reset()
        state_dim = self._extract_state(raw).shape[0]

        obs_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
        }
        if use_camera_obs:
            for cam in self._camera_names:
                obs_spaces[f"image_{cam}"] = spaces.Box(
                    0, 255, shape=(3, image_size, image_size), dtype=np.uint8
                )

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._last_raw_obs = raw

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_state(self, raw_obs: dict) -> np.ndarray:
        parts = [raw_obs[k].astype(np.float32) for k in self._state_keys if k in raw_obs]
        return np.concatenate(parts)

    def _extract_image(self, raw_obs: dict, cam: str) -> np.ndarray:
        """Returns a (3, H, W) uint8 array, right-side up."""
        img = raw_obs[f"{cam}_image"]   # (H, W, 3) uint8, MuJoCo upside-down
        img = img[::-1]                 # flip vertically
        if img.shape[0] != self._image_size or img.shape[1] != self._image_size:
            img = cv2.resize(img, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
        return img.transpose(2, 0, 1).copy()  # HWC → CHW

    def _build_obs(self, raw_obs: dict) -> dict:
        obs: dict[str, np.ndarray] = {"state": self._extract_state(raw_obs)}
        if self._use_camera_obs:
            for cam in self._camera_names:
                obs[f"image_{cam}"] = self._extract_image(raw_obs, cam)
        return obs

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        raw_obs = self._env.reset()
        self._last_raw_obs = raw_obs
        # Allow reward functions to reset per-episode state (e.g. milestone flags)
        if hasattr(self._reward_fn, "on_episode_reset"):
            self._reward_fn.on_episode_reset()
        return self._build_obs(raw_obs), {}

    def step(self, action: np.ndarray):
        raw_obs, _builtin_reward, done, info = self._env.step(action)
        self._last_raw_obs = raw_obs

        reward = self._reward_fn(self._env, raw_obs)

        # robosuite sets done=True both on termination and horizon truncation
        truncated = bool(info.get("is_horizon_reached", False))
        terminated = done and not truncated

        return self._build_obs(raw_obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Pass-through for reward-fn helpers that need the live env
    # ------------------------------------------------------------------

    @property
    def unwrapped_env(self):
        """Direct access to the raw RoboCasa environment."""
        return self._env
