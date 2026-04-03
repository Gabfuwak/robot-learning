"""
LeWM-based features extractor for SB3 PPO.

Loads a frozen LeWM checkpoint directly (no stable_worldmodel dependency)
and uses its encoder to map (pixels, pixels_eih, proprio) → 256-dim latent z.

Usage in TrainConfig:
    cfg = TrainConfig(
        lewm_checkpoint="/path/to/lewm_epoch_84_object.ckpt",
        use_camera_obs=True,
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        image_size=96,
    )
"""

import os
import sys
import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Make the le-wm-robocasa submodule importable so torch.load can unpickle JEPA
_SUBMODULE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deps", "le-wm-robocasa")
if _SUBMODULE not in sys.path:
    sys.path.insert(0, _SUBMODULE)


class LeWMFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 BaseFeaturesExtractor backed by a frozen LeWM encoder.

    Expects RoboCasaWrapper's Dict obs space with:
      "state"                          float32 (STATE_DIM,)
      "image_robot0_agentview_left"    uint8   (3, H, W)
      "image_robot0_eye_in_hand"       uint8   (3, H, W)

    Args:
        observation_space:  Dict gym space from RoboCasaWrapper.
        checkpoint:         Absolute path to the _object.ckpt file.
        proprio_slice:      (start, end) into "state" giving the 9-dim LeWM proprio
                            (eef_pos_rel×3 + eef_rot_rel×4 + gripper×2).
        embed_dim:          LeWM latent dimension (must match checkpoint, default 256).
    """

    LEFT_CAM = "image_robot0_agentview_left"
    EIH_CAM  = "image_robot0_eye_in_hand"

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        checkpoint: str,
        proprio_slice: tuple[int, int] = (7, 16),
        embed_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=embed_dim)

        self._proprio_slice = proprio_slice

        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.eval()
        model.requires_grad_(False)
        self.lewm = model

        # ImageNet normalisation — registered as buffers so they follow .to(device)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """uint8 (B, 3, H, W) → float32 ImageNet-normalised."""
        return (img.float() / 255.0 - self._mean) / self._std

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        pixels     = self._normalize(obs[self.LEFT_CAM])   # (B, 3, H, W)
        pixels_eih = self._normalize(obs[self.EIH_CAM])    # (B, 3, H, W)
        s, e       = self._proprio_slice
        proprio    = obs["state"][:, s:e].float()           # (B, 9)

        # LeWM encode() expects (B, T, ...) — T=1 (single frame, no history)
        info = {
            "pixels":     pixels.unsqueeze(1),      # (B, 1, 3, H, W)
            "pixels_eih": pixels_eih.unsqueeze(1),  # (B, 1, 3, H, W)
            "proprio":    proprio.unsqueeze(1),      # (B, 1, 9)
        }

        with torch.no_grad():
            info = self.lewm.encode(info)

        return info["emb"][:, 0, :]  # (B, embed_dim)
