"""
Inspect the observation dictionary from PickPlaceCounterToCabinet.

Resets the environment once, then saves the full observation as:
  obs_output/observation.json   — all keys; image keys store a relative path instead of pixel data
  obs_output/images/            — one PNG per camera
"""

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deps", "robocasa"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deps", "robosuite"))

from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
from robosuite.controllers import load_composite_controller_config

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "obs_output")
IMAGE_DIR    = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON  = os.path.join(OUTPUT_DIR, "observation.json")

CAMERA_NAMES   = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
CAMERA_HEIGHT  = 256
CAMERA_WIDTH   = 256


def build_env():
    ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = PickPlaceCounterToCabinet(
        robots="PandaOmron",
        controller_configs=ctrl,
        layout_ids=[1],
        style_ids=[1],
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=True,
        camera_names=CAMERA_NAMES,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
        control_freq=20,
    )
    return env


def serialize_obs(obs: dict) -> dict:
    """Convert obs dict to a JSON-serialisable dict.

    - numpy arrays  → python list  (or scalar)
    - image arrays  → relative path string; the PNG is saved to IMAGE_DIR
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    result = {}

    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3:
            # Likely an RGB image (H, W, 3) uint8 — MuJoCo renders upside-down, flip it
            img_array = value[::-1].copy()
            rel_path  = os.path.join("images", f"{key}.png")
            abs_path  = os.path.join(OUTPUT_DIR, rel_path)
            Image.fromarray(img_array).save(abs_path)
            result[key] = {
                "__type__": "image",
                "path": rel_path,
                "shape": list(value.shape),   # original (H, W, 3)
                "dtype": str(value.dtype),
            }
        elif isinstance(value, np.ndarray):
            result[key] = {
                "__type__": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "data": value.tolist(),
            }
        else:
            result[key] = value

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building environment …")
    env = build_env()

    print("Resetting environment …")
    obs = env.reset()
    env.close()

    print(f"Observation has {len(obs)} keys:")
    for k, v in obs.items():
        shape = v.shape if isinstance(v, np.ndarray) else type(v).__name__
        print(f"  {k:45s}  {shape}")

    print("\nSerialising …")
    serialised = serialize_obs(obs)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(serialised, f, indent=2)

    print(f"\nSaved observation to  {OUTPUT_JSON}")
    print(f"Saved images to       {IMAGE_DIR}/")


if __name__ == "__main__":
    main()
