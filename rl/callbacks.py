"""
Custom SB3 callbacks for RoboCasa training.

StageSuccessCallback
────────────────────
Evaluates the policy every `eval_freq` steps and logs per-stage success rates
to TensorBoard (and later W&B). Runs its own eval loop over `n_eval_episodes`
using a dedicated DummyVecEnv so it doesn't interfere with EvalCallback.

Logged metrics
──────────────
  eval/stage_grasp_rate   — fraction of episodes where object was grasped ≥ once
  eval/success_rate       — fraction of episodes where object entered the cabinet
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class StageSuccessCallback(BaseCallback):
    """
    Args:
        eval_env:        A DummyVecEnv wrapping a single RoboCasaWrapper eval env.
        eval_freq:       Number of *environment steps* between evaluations.
                         Pass cfg.checkpoint_freq to align with checkpoint saves.
        n_eval_episodes: Number of episodes to roll out per evaluation.
        verbose:         0 = silent, 1 = print results.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes

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

        if self.verbose:
            print(
                f"[StageSuccessCallback] step={self.num_timesteps:,}  "
                f"grasp_rate={grasp_rate:.2f}  success_rate={success_rate:.2f}"
                f"  ({self.n_eval_episodes} eps)"
            )

        return True
