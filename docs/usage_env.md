# Observation & Action Space — PickPlaceCounterToCabinet

Robot: `PandaOmron` | Controller: default composite OSC

---

## Observation Space

When using `GymWrapper(env, keys=None)` the observation is flattened into a **110-dim vector** (`robot0_proprio-state` + `object-state`).

### Robot proprio-state (68 dims)

| Key | Shape | Description |
|---|---|---|
| `robot0_joint_pos` | (7,) | Arm joint positions |
| `robot0_joint_pos_cos` | (7,) | Cosine encoding of joint positions |
| `robot0_joint_pos_sin` | (7,) | Sine encoding of joint positions |
| `robot0_joint_vel` | (7,) | Arm joint velocities |
| `robot0_joint_acc` | (7,) | Arm joint accelerations |
| `robot0_eef_pos` | (3,) | EEF world position |
| `robot0_eef_quat` | (4,) | EEF world orientation |
| `robot0_eef_quat_site` | (4,) | EEF site orientation |
| `robot0_gripper_qpos` | (2,) | Gripper joint positions |
| `robot0_gripper_qvel` | (2,) | Gripper joint velocities |
| `robot0_base_pos` | (3,) | Mobile base world position |
| `robot0_base_quat` | (4,) | Mobile base world orientation |
| `robot0_base_to_eef_pos` | (3,) | EEF position relative to base |
| `robot0_base_to_eef_quat` | (4,) | EEF orientation relative to base |
| `robot0_base_to_eef_quat_site` | (4,) | EEF site orientation relative to base |

### Object-state (42 dims)

Each object contributes 14 dims: `pos (3) + quat (4) + to_eef_pos (3) + to_eef_quat (4)`

| Object | Keys | Description |
|---|---|---|
| `obj` | `obj_pos`, `obj_quat`, `obj_to_robot0_eef_pos`, `obj_to_robot0_eef_quat` | Target object (to be picked and placed) |
| `distr_counter` | `distr_counter_pos`, `distr_counter_quat`, ... | Distractor on the counter |
| `distr_cab` | `distr_cab_pos`, `distr_cab_quat`, ... | Distractor inside the cabinet |

> **Note:** The object-state keys and total dimension vary per task depending on how many objects `_get_obj_cfgs()` spawns. The proprio-state (68 dims) is consistent across all tasks with `PandaOmron`.

### Camera observations (optional)

Enabled by passing `use_camera_obs=True` and `has_offscreen_renderer=True` at env creation. Each camera adds a `(H, W, 3)` uint8 RGB image to the obs dict as `{camera_name}_image`.

| Camera name | Description |
|---|---|
| `robot0_agentview_center` | Front-facing third-person view |
| `robot0_agentview_left` | Left third-person view |
| `robot0_agentview_right` | Right third-person view |
| `robot0_eye_in_hand` | Wrist-mounted camera |
| `robot0_frontview` | Fixed front view |

Resolution is set via `camera_heights` and `camera_widths` at env creation. MuJoCo renders images upside-down — flip with `img[::-1]` before use.

> **Note:** No force/torque sensor is included in the observation. The robot has no explicit grasp feedback. Successful grasping can only be inferred indirectly — if the object is held, `obj_to_robot0_eef_pos` stays near zero as the arm moves; if the grasp fails, it grows as the EEF moves away from the object.

---

## Example Usage

```python
import robosuite
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
from robosuite.controllers import load_composite_controller_config

ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
env = PickPlaceCounterToCabinet(
    robots="PandaOmron",
    controller_configs=ctrl,
    use_camera_obs=True,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_object_obs=True,
    camera_names=["robot0_agentview_center", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    layout_ids=[1],
    style_ids=[1],
)

obs = env.reset()

# --- Robot state ---
joint_pos   = obs["robot0_joint_pos"]          # (7,) arm joint angles in radians
joint_vel   = obs["robot0_joint_vel"]          # (7,) arm joint velocities
eef_pos     = obs["robot0_eef_pos"]            # (3,) gripper tip position in world frame
eef_quat    = obs["robot0_eef_quat"]           # (4,) gripper tip orientation (xyzw)
gripper     = obs["robot0_gripper_qpos"]       # (2,) gripper finger positions
base_pos    = obs["robot0_base_pos"]           # (3,) mobile base position in world frame
base_quat   = obs["robot0_base_quat"]          # (4,) mobile base orientation

# --- Object state ---
obj_pos     = obs["obj_pos"]                   # (3,) target object world position
obj_quat    = obs["obj_quat"]                  # (4,) target object world orientation
obj_rel_pos = obs["obj_to_robot0_eef_pos"]     # (3,) object position relative to EEF
obj_rel_quat= obs["obj_to_robot0_eef_quat"]    # (4,) object orientation relative to EEF

# Distractor objects (same pattern)
distr_pos   = obs["distr_counter_pos"]         # (3,) counter distractor world position
cab_pos     = obs["distr_cab_pos"]             # (3,) cabinet distractor world position

# Infer whether the object is being held:
# if grasped, obj_rel_pos stays near zero as the arm moves
import numpy as np
is_likely_grasped = np.linalg.norm(obj_rel_pos) < 0.05

# --- Camera images ---
import numpy as np
agentview   = obs["robot0_agentview_center_image"]  # (128, 128, 3) uint8 RGB, upside-down
eye_in_hand = obs["robot0_eye_in_hand_image"]        # (128, 128, 3) uint8 RGB, upside-down

# Flip right-side up
agentview   = agentview[::-1]
eye_in_hand = eye_in_hand[::-1]

# --- Flattened vector (as seen by GymWrapper / RL policy) ---
from robosuite.wrappers.gym_wrapper import GymWrapper
gym_env = GymWrapper(env, keys=None)
obs_flat = gym_env.reset()   # (110,) float64 — proprio (68) + object (42)
                              # images are NOT included when keys=None and use_camera_obs=False
```

---

## Action Space

**12-dim continuous**, clipped to `[-1, 1]`.

| Indices | Description |
|---|---|
| `[0:3]` | EEF position delta (x, y, z) |
| `[3:6]` | EEF orientation delta (rotation vector) |
| `[6]` | Gripper: `-1` = open, `+1` = close |
| `[7:10]` | Mobile base velocity (vx, vy, vθ) |
| `[10:12]` | Padding (torso / unused) |

In absolute mode (set `ctrl_cfg['body_parts']['right']['input_type'] = 'absolute'`), indices `[0:3]` become a target EEF position in the robot base frame instead of a delta.
