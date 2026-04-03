"""
Custom SB3 callbacks for RoboCasa training.

RollingSuccessCallback
──────────────────────
Tracks a rolling window of completed training episodes and logs live grasp/success
rates without running a separate eval loop.

Logged metrics
──────────────
  train/rolling_grasp_rate   — grasp rate over last `window` training episodes
  train/rolling_success_rate — success rate over last `window` training episodes

StageSuccessCallback
────────────────────
Evaluates the policy every `eval_freq` steps and logs per-stage success rates
to TensorBoard (and later W&B). Runs its own eval loop over `n_eval_episodes`
using a dedicated DummyVecEnv so it doesn't interfere with EvalCallback.

Logged metrics
──────────────
  eval/stage_grasp_rate          — fraction of episodes where object was grasped ≥ once
  eval/success_rate              — fraction of episodes where object entered the cabinet
  eval/curriculum_grasp_rate    — grasp rate at current curriculum difficulty
  eval/curriculum_success_rate  — success rate at current curriculum difficulty

CurriculumCallback
──────────────────
Implements a grasp-triggered spawn-distance curriculum. Every training episode
that ends with a successful grasp increments the allowed spawn distance by
`epsilon * (current_dist / max_dist) / n_envs` (up to `max_dist`). The
normalized factor is 0 at init_dist and 1 at max_dist, so harder grasps earn
more progress; dividing by `n_envs` keeps progression rate independent of
parallelism.

Logged metrics
──────────────
  curriculum/max_spawn_dist — current maximum allowed spawn distance (metres)
"""

from __future__ import annotations

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class RollingSuccessCallback(BaseCallback):
    """
    Logs rolling grasp and success rates over training episodes (no eval loop).

    Args:
        window: Number of completed episodes to average over.
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._grasps   = deque(maxlen=window)
        self._successes = deque(maxlen=window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self._grasps.append(int(info.get("ever_grasped", False)))
                self._successes.append(int(info.get("ever_inside",  False)))

        if self._grasps:
            self.logger.record("train/rolling_grasp_rate",   sum(self._grasps)    / len(self._grasps))
            self.logger.record("train/rolling_success_rate", sum(self._successes) / len(self._successes))

        return True


class StageSuccessCallback(BaseCallback):
    """
    Args:
        eval_env:             A DummyVecEnv wrapping a single RoboCasaWrapper eval env.
        eval_freq:            Number of *environment steps* between evaluations.
                              Pass cfg.checkpoint_freq to align with checkpoint saves.
        n_eval_episodes:      Number of episodes to roll out per evaluation.
        curriculum_callback:  Optional CurriculumCallback — if provided together with
                              curriculum_eval_env, also logs eval/curriculum_success_rate
                              on an env synced to the current curriculum difficulty.
        curriculum_eval_env:  A DummyVecEnv with max_spawn_dist set, used for the
                              curriculum-difficulty evaluation.
        verbose:              0 = silent, 1 = print results.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int = 10,
        curriculum_callback: "CurriculumCallback | None" = None,
        curriculum_eval_env: "VecEnv | None" = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env             = eval_env
        self.eval_freq            = eval_freq
        self.n_eval_episodes      = n_eval_episodes
        self.curriculum_callback  = curriculum_callback
        self.curriculum_eval_env  = curriculum_eval_env

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        grasp_count  = 0
        inside_count = 0

        obs = self.eval_env.reset()
        ep_grasped = ep_inside = False
        episodes_done = 0

        while episodes_done < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.eval_env.step(action)

            ep_grasped = ep_grasped or infos[0].get("ever_grasped", False)
            ep_inside  = ep_inside  or infos[0].get("ever_inside",  False)

            if dones[0]:
                grasp_count  += int(ep_grasped)
                inside_count += int(ep_inside)
                ep_grasped = ep_inside = False
                episodes_done += 1

        grasp_rate  = grasp_count  / self.n_eval_episodes
        success_rate = inside_count / self.n_eval_episodes

        self.logger.record("eval/stage_grasp_rate", grasp_rate)
        self.logger.record("eval/success_rate",     success_rate)

        if self.curriculum_callback is not None and self.curriculum_eval_env is not None:
            curr_dist = self.curriculum_callback.current_dist
            self.curriculum_eval_env.env_method("set_max_spawn_dist", curr_dist)

            curr_grasp_count = curr_inside_count = 0
            obs = self.curriculum_eval_env.reset()
            ep_grasped = ep_inside = False
            episodes_done = 0

            while episodes_done < self.n_eval_episodes:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.curriculum_eval_env.step(action)

                ep_grasped = ep_grasped or infos[0].get("ever_grasped", False)
                ep_inside  = ep_inside  or infos[0].get("ever_inside",  False)

                if dones[0]:
                    curr_grasp_count  += int(ep_grasped)
                    curr_inside_count += int(ep_inside)
                    ep_grasped = ep_inside = False
                    episodes_done += 1

            self.logger.record("eval/curriculum_grasp_rate",   curr_grasp_count  / self.n_eval_episodes)
            self.logger.record("eval/curriculum_success_rate", curr_inside_count / self.n_eval_episodes)

        if self.verbose:
            print(
                f"[StageSuccessCallback] step={self.num_timesteps:,}  "
                f"grasp_rate={grasp_rate:.2f}  success_rate={success_rate:.2f}"
                f"  ({self.n_eval_episodes} eps)"
            )

        self.logger.dump(step=self.num_timesteps)
        return True


class CurriculumCallback(BaseCallback):
    """
    Increments max_spawn_dist on all training envs by
    `epsilon * (current_dist / max_dist) / n_envs` for every training episode
    that ends with a successful grasp.

    Args:
        init_dist:  Starting max spawn distance (metres).
        epsilon:    Max increment per grasped episode at full difficulty, single env
                    (dimensionless; typical range 0.01–0.2).
        max_dist:   Hard cap on max spawn distance (metres).
        save_path:  Directory where ``curriculum_state.json`` is written on every
                    update so that training can be resumed at the correct difficulty.
    """

    STATE_FILE = "curriculum_state.json"

    def __init__(self, init_dist: float, epsilon: float, max_dist: float,
                 save_path: str | None = None):
        super().__init__()
        self.current_dist = init_dist
        self.epsilon      = epsilon
        self.max_dist     = max_dist
        self.save_path    = save_path

    def _save_state(self) -> None:
        if self.save_path is None:
            return
        import json
        import os
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, self.STATE_FILE), "w") as f:
            json.dump({"current_dist": self.current_dist}, f)

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_max_spawn_dist", self.current_dist)
        self._save_state()

    def _on_step(self) -> bool:
        changed = False
        n_envs = self.training_env.num_envs
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and info.get("ever_grasped", False):
                delta = self.epsilon * (self.current_dist / self.max_dist) / n_envs
                self.current_dist = min(self.current_dist + delta, self.max_dist)
                changed = True

        if changed:
            self.training_env.env_method("set_max_spawn_dist", self.current_dist)
            self._save_state()

        self.logger.record("curriculum/max_spawn_dist", self.current_dist)
        return True
