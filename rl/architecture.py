"""
Neural network architecture for RoboCasa RL policies.

RoboCasaFeaturesExtractor is a SB3-compatible BaseFeaturesExtractor that handles
the Dict observation space produced by RoboCasaWrapper:

  "state"               float32 (STATE_DIM,)   → StateEncoder  → (state_embed_dim,)
  "image_<cam>"  uint8  (3, H, W)              → ImageEncoder  → (image_embed_dim,)
                                                                      ↓ concat
                                                              (features_dim,)

The extracted feature vector is then consumed by SB3's standard SAC actor/critic
heads (two separate MLPs), so you get a full SAC agent by passing this extractor
via policy_kwargs.

Usage with SB3 SAC
──────────────────
    from stable_baselines3 import SAC
    from stable_baselines3.common.policies import MultiInputPolicy
    from rl.architecture import RoboCasaFeaturesExtractor

    model = SAC(
        policy=MultiInputPolicy,
        env=wrapped_env,
        policy_kwargs=dict(
            features_extractor_class=RoboCasaFeaturesExtractor,
            features_extractor_kwargs=dict(state_embed_dim=256, image_embed_dim=128),
            net_arch=[256, 256],   # actor/critic MLP layers after the extractor
        ),
    )
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """Two-layer MLP with LayerNorm for the flat state vector."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """
    Small CNN for a single (3, H, W) uint8 image.

    Pixel values are normalised to [0, 1] inside the forward pass.
    The final spatial feature map is average-pooled to a vector and projected
    to out_dim with a LayerNorm+Tanh to keep activations bounded.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            # 3 × H × W
            nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.ReLU(),   # 32 × (H-2)/2
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),  # 64 × …
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(), # 128 × …
            nn.AdaptiveAvgPool2d(1),                                 # 128 × 1 × 1
            nn.Flatten(),                                            # (128,)
        )
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) uint8 or float
        return self.proj(self.conv(x.float() / 255.0))


# ---------------------------------------------------------------------------
# SB3 feature extractor
# ---------------------------------------------------------------------------

class RoboCasaFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 BaseFeaturesExtractor for RoboCasaWrapper's Dict observation space.

    Args:
        observation_space: The Dict gym space produced by RoboCasaWrapper.
        state_embed_dim:   Output dimension of the state encoder.
        image_embed_dim:   Output dimension of each image encoder.
                           If no camera images are in the obs space, this is unused.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        state_embed_dim: int = 256,
        image_embed_dim: int = 128,
    ):
        spaces = observation_space.spaces
        image_keys = sorted(k for k in spaces if k.startswith("image_"))
        features_dim = state_embed_dim + image_embed_dim * len(image_keys)

        super().__init__(observation_space, features_dim=features_dim)

        state_in = spaces["state"].shape[0]
        self.state_encoder = StateEncoder(state_in, state_embed_dim)

        self.image_encoders = nn.ModuleDict(
            {k: ImageEncoder(image_embed_dim) for k in image_keys}
        )
        self._image_keys = image_keys

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.state_encoder(obs["state"])]
        for k in self._image_keys:
            parts.append(self.image_encoders[k](obs[k]))
        return torch.cat(parts, dim=-1)
