# Using RoboCasa Datasets for Training

## Overview

RoboCasa provides datasets in the **LeRobot format** across three primary categories:
- Pretraining (human)
- Pretraining (MimicGen)
- Target (human)

---

## Downloading Datasets

Datasets are stored by default in `datasets/` within the RoboCasa root directory. The location can be customized via `DATASET_BASE_PATH` in `robocasa/macros_private.py`.

```bash
# Download all datasets
python -m robocasa.scripts.download_datasets --all

# Pretraining human data
python -m robocasa.scripts.download_datasets --split pretrain --source human

# Pretraining synthetic (MimicGen)
python -m robocasa.scripts.download_datasets --split pretrain --source mimicgen

# Target human data
python -m robocasa.scripts.download_datasets --split target --source human

# Specific tasks only
python -m robocasa.scripts.download_datasets --tasks PickPlaceCounterToCabinet

# Overwrite existing datasets
python -m robocasa.scripts.download_datasets --split pretrain --source human --overwrite
```

---

## Dataset Structure

Each dataset in LeRobot format contains:

| Folder / File | Description |
|---|---|
| `meta/` | Metadata files: `info.json`, `tasks.jsonl`, `episodes.jsonl`, `stats.json`, `modality.json`, `embodiment.json` |
| `data/` | Low-dimensional trajectory data in Parquet format, organized by chunks and episodes |
| `videos/` | MP4 files for three camera views: left third-person, right third-person, eye-in-hand |
| `extras/` | MuJoCo/RoboCasa-specific metadata including environment args and compressed model XMLs |

---

## Retrieving Dataset Metadata

Use `get_ds_meta()` to look up path, horizon, filter key, and other metadata:

```python
from robocasa.utils.dataset_registry_utils import get_ds_meta

ds_meta = get_ds_meta(
    task="PickPlaceCounterToCabinet",
    split="target",       # "pretrain", "target", or "real"
    source="human",       # "human" or "mg" (mimicgen)
    demo_fraction=1.0     # fraction of demos to use (0.0 - 1.0)
)

print(ds_meta["path"])        # resolved dataset path
print(ds_meta["horizon"])     # episode horizon
print(ds_meta["filter_key"])  # e.g. "500_demos"
```

Or use the convenience wrapper to get just the path:

```python
from robocasa.utils.dataset_registry_utils import get_ds_path

path = get_ds_path(
    task="PickPlaceCounterToCabinet",
    source="human",
    split="pretrain"
)
```

---

## Loading a Dataset for Training

Access individual samples using `LeRobotDataset`:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="robocasa365", root=dataset_path)
sample = ds[start + timestep_idx]
```

Each sample contains:
- **Observation images** — from each camera view
- **Actions** — robot control actions at that timestep
- **Task instruction** — language description of the task

---

## Multi-Dataset (Soup) Training

To train across multiple tasks at once, use `get_ds_soup()`:

```python
from robocasa.utils.dataset_registry import get_ds_soup

ds_soup = get_ds_soup(
    task_soup="atomic_seen",  # predefined task group
    split="target",
    source="human"
)
```

Then combine with `LeRobotMixtureDataset`, providing a modality config that specifies:
- `video` — camera observations
- `state` — proprioceptive state
- `action` — robot actions
- `language` — task instruction

---

## Dataset Inspection

```bash
# Print dataset statistics
python robocasa/scripts/get_dataset_info.py --dataset <path>

# Visualize demonstrations
python robocasa/scripts/playback_dataset.py --n 10 --dataset <path>
```

---

## Sampling an Environment (Random Layout & Style)

### Layout and Style IDs

Kitchens are parameterized by two independent axes:

- **Layout** (`layout_ids`): controls the floor plan (counter arrangement, island presence, etc.) — integers 1–60
- **Style** (`style_ids`): controls the visual theme (cabinet color, countertop material, appliance style, etc.) — integers 1–60

Negative IDs select predefined groups:

| ID | Name | Layouts | Styles |
|---|---|---|---|
| `-1` | TEST | 1–10 | 1–10 |
| `-2` | TRAIN | 11–60 | 11–60 |
| `-3` | ALL | 1–60 | 1–60 |
| `-4` | NO_ISLAND | [1,3,5,6,8] | — |
| `-5` | ISLAND | [2,4,7,9,10] | — |

At each `env.reset()` the environment samples one `(layout_id, style_id)` pair uniformly from the Cartesian product of the provided lists, rebuilds the kitchen arena, and re-randomizes object placements.

### Creating an Environment

```python
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
from robosuite.controllers import load_composite_controller_config

ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")

# --- Fixed layout and style (deterministic kitchen) ---
env = PickPlaceCounterToCabinet(
    robots="PandaOmron",
    controller_configs=ctrl,
    layout_ids=[1],    # single layout
    style_ids=[1],     # single style
    seed=42,
    use_camera_obs=True,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_object_obs=True,
    camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
    control_freq=20,
)

# --- Fully random across all training layouts and styles ---
env = PickPlaceCounterToCabinet(
    robots="PandaOmron",
    controller_configs=ctrl,
    layout_ids=-2,     # all training layouts (11–60)
    style_ids=-2,      # all training styles (11–60)
    use_camera_obs=True,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_object_obs=True,
    camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()   # new layout+style sampled here
```

The helper `create_env` wraps this with split-aware defaults:

```python
from robocasa.utils.env_utils import create_env

# split="pretrain"  →  layout_ids=-2, style_ids=-2, obj_instance_split="pretrain"
# split="target"    →  layout_ids=-1, style_ids=-1  (test distribution)
# split="all"       →  layout_ids=-3, style_ids=-3
env = create_env(env_name="PickPlaceCounterToCabinet", split="pretrain")
```

---

## Observation Dictionary

`env.reset()` and `env.step()` both return a dictionary with the following keys.

### Robot Proprioception

All keys are prefixed with `robot0_` for the first robot (PandaOmron).

| Key | Shape | dtype | Description |
|---|---|---|---|
| `robot0_joint_pos` | `(7,)` | float64 | Arm joint positions in radians |
| `robot0_joint_pos_cos` | `(7,)` | float64 | Cosine encoding of joint positions |
| `robot0_joint_pos_sin` | `(7,)` | float64 | Sine encoding of joint positions |
| `robot0_joint_vel` | `(7,)` | float64 | Arm joint velocities (rad/s) |
| `robot0_joint_acc` | `(7,)` | float64 | Arm joint accelerations |
| `robot0_eef_pos` | `(3,)` | float64 | End-effector world position (meters) |
| `robot0_eef_quat` | `(4,)` | float64 | End-effector world orientation, quaternion xyzw (body-queried) |
| `robot0_eef_quat_site` | `(4,)` | float64 | End-effector world orientation, quaternion xyzw (site-queried, preferred) |
| `robot0_gripper_qpos` | `(2,)` | float64 | Gripper finger joint positions |
| `robot0_gripper_qvel` | `(2,)` | float64 | Gripper finger joint velocities |
| `robot0_base_pos` | `(3,)` | float64 | Mobile base position in world frame (meters) |
| `robot0_base_quat` | `(4,)` | float64 | Mobile base orientation, quaternion xyzw |
| `robot0_base_to_eef_pos` | `(3,)` | float64 | End-effector position relative to base frame |
| `robot0_base_to_eef_quat` | `(4,)` | float64 | End-effector orientation relative to base frame (body-queried) |
| `robot0_base_to_eef_quat_site` | `(4,)` | float64 | End-effector orientation relative to base frame (site-queried, preferred) |

These are also concatenated into:

| Key | Shape | Description |
|---|---|---|
| `robot0_proprio-state` | `(68,)` | All proprioception keys above concatenated |

### Object State

Each object in the scene contributes 4 keys. For `PickPlaceCounterToCabinet` there are 3 objects: the target (`obj`) and two distractors (`distr_counter`, `distr_cab`).

| Key pattern | Shape | dtype | Description |
|---|---|---|---|
| `{name}_pos` | `(3,)` | float64 | Object world position (meters) |
| `{name}_quat` | `(4,)` | float64 | Object world orientation, quaternion xyzw |
| `{name}_to_robot0_eef_pos` | `(3,)` | float64 | Object position relative to end-effector |
| `{name}_to_robot0_eef_quat` | `(4,)` | float64 | Object orientation relative to end-effector |

Substituting the 3 object names gives 12 individual keys (3 × 4), also concatenated into:

| Key | Shape | Description |
|---|---|---|
| `object-state` | `(42,)` | All object keys concatenated (3 objects × 14 dims) |

> The number of objects (and hence the shape of `object-state`) varies per task — it depends on how many objects `_get_obj_cfgs()` defines. The proprio-state shape (68) is fixed across all PandaOmron tasks.

### Camera Images

Returned when `use_camera_obs=True`. Each camera produces one key per enabled modality.

| Key pattern | Shape | dtype | Description |
|---|---|---|---|
| `{camera_name}_image` | `(H, W, 3)` | uint8 | RGB image. H and W are set by `camera_heights`/`camera_widths`. |
| `{camera_name}_depth` | `(H, W)` | float32 | Depth image (only when `camera_depths=True`). |

Available camera names:

| Camera name | Position |
|---|---|
| `robot0_agentview_left` | Left stereo — default for training |
| `robot0_agentview_right` | Right stereo — default for training |
| `robot0_eye_in_hand` | Wrist-mounted — default for training |
| `robot0_agentview_center` | Front-facing center view |
| `robot0_frontview` | Fixed front view |

> MuJoCo renders images upside-down. Flip with `img[::-1]` before use.

### Aggregated State Vector (LeRobot dataset format)

When loading from a LeRobot parquet file, the low-dimensional state is stored as a flat `(16,)` float64 array (`observation.state`):

| Indices | Source key | Description |
|---|---|---|
| `[0:3]` | `robot0_base_pos` | Mobile base world position |
| `[3:7]` | `robot0_base_quat` | Mobile base world orientation |
| `[7:10]` | `robot0_base_to_eef_pos` | EEF position relative to base |
| `[10:14]` | `robot0_base_to_eef_quat` | EEF orientation relative to base |
| `[14:16]` | `robot0_gripper_qpos` | Gripper finger positions |

### Full Example

```python
import numpy as np
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
from robosuite.controllers import load_composite_controller_config

ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
env = PickPlaceCounterToCabinet(
    robots="PandaOmron",
    controller_configs=ctrl,
    layout_ids=-2,          # random training layout each reset
    style_ids=-2,           # random training style each reset
    use_camera_obs=True,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_object_obs=True,
    camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()

# -- Robot state --
eef_pos  = obs["robot0_eef_pos"]             # (3,)  world-frame EEF position
eef_quat = obs["robot0_eef_quat_site"]       # (4,)  world-frame EEF orientation (xyzw)
gripper  = obs["robot0_gripper_qpos"]        # (2,)  finger joint positions
base_pos = obs["robot0_base_pos"]            # (3,)  mobile base world position
base_quat= obs["robot0_base_quat"]           # (4,)  mobile base world orientation
rel_pos  = obs["robot0_base_to_eef_pos"]     # (3,)  EEF position relative to base
rel_quat = obs["robot0_base_to_eef_quat_site"] # (4,) EEF orientation relative to base

# -- Object state --
obj_pos      = obs["obj_pos"]                    # (3,)  target object world position
obj_quat     = obs["obj_quat"]                   # (4,)  target object world orientation
obj_rel_pos  = obs["obj_to_robot0_eef_pos"]      # (3,)  object position relative to EEF
obj_rel_quat = obs["obj_to_robot0_eef_quat"]     # (4,)  object orientation relative to EEF

# Heuristic: object is likely grasped if it stays close to EEF as arm moves
is_grasped = np.linalg.norm(obj_rel_pos) < 0.05

# -- Camera images (MuJoCo renders upside-down — flip before use) --
left_img  = obs["robot0_agentview_left_image"][::-1]   # (256, 256, 3) uint8 RGB
wrist_img = obs["robot0_eye_in_hand_image"][::-1]      # (256, 256, 3) uint8 RGB

# -- Flat vectors (as seen by GymWrapper / RL policy) --
from robosuite.wrappers.gym_wrapper import GymWrapper
gym_env  = GymWrapper(env, keys=None)          # keys=None → proprio + object state only
obs_flat = gym_env.reset()                     # (110,) float64
#   [0:68]   robot0_proprio-state
#   [68:110] object-state
```
