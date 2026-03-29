"""
Imitation Learning (IL) package for RoboCasa.

Modules
───────
    base       — Shared config (ILConfig) and data-collection pipeline
    bc         — Behaviour Cloning: policy, training, evaluation, BC→RL warm-start
    gail       — GAIL: discriminator-based imitation via adversarial training
    diffusion  — Diffusion Policy: generative imitation via denoising
"""

from .base import (
    ILConfig,
    collect_demo_data,
    collect_episode_data,
)

from .bc import (
    BCPolicy,
    evaluate_bc,
    train_bc,
    warm_start_from_bc,
)

__all__ = [
    # base
    "ILConfig",
    "collect_demo_data",
    "collect_episode_data",
    # bc
    "BCPolicy",
    "evaluate_bc",
    "train_bc",
    "warm_start_from_bc",
]
