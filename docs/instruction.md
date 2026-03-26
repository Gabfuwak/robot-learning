# Using the RoboCasa Dataset / Environment

This guide explains how to set up and use a RoboCasa kitchen manipulation environment, using `PnPCounterToCab` (pick-and-place from counter to cabinet) as a worked example. The same pattern applies to other tasks.

---

## Project Structure

```
helpers/
├── env/
│   └── custom_pnp_counter_to_cab.py   # Custom env subclass (reward, layout, objects)
├── rl_scripts/
│   ├── train_robocasa.py               # PPO training entry point
│   └── eval_robocasa.py                # Evaluation + video rendering
└── grasp_apple_absolute_gripper.py     # Scripted demo (OSC absolute controller)
```

---

## 1. Custom Environment

RoboCasa environments are subclassed from robosuite/robocasa base classes. The pattern is:

```python
# helpers/env/custom_pnp_counter_to_cab.py
from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab

class MyPnPCounterToCab(PnPCounterToCab):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('layout_ids', [1])   # fix kitchen layout
        kwargs.setdefault('style_ids', [1])    # fix kitchen style
        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        # Override to specify exact objects (apple + bowl distractor)
        ...

    def reward(self, action=None):
        # Override with your custom reward
        return 0
```

**Key overrides:**
- `_get_obj_cfgs()` — controls which objects are spawned and their placement regions.
- `reward()` — implement your own reward shaping here.
- `_get_placement_initializer()` — optionally fix fixture placement with a seeded RNG for reproducibility.

### Register with robosuite

```python
from robosuite.environments.base import register_env
register_env(MyPnPCounterToCab)

# Now usable via:
import robosuite
env = robosuite.make("MyPnPCounterToCab", robots="PandaOmron", ...)
```

---

## 2. Creating the Environment

```python
import robosuite
from robosuite.controllers import load_composite_controller_config

ctrl_cfg = load_composite_controller_config(controller=None, robot="PandaOmron")

env = robosuite.make(
    env_name="MyPnPCounterToCab",
    robots="PandaOmron",
    controller_configs=ctrl_cfg,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_object_obs=True,
    use_camera_obs=True,
    camera_names=["robot0_agentview_center", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
    control_freq=20,
    reward_shaping=True,
    ignore_done=False,
    horizon=500,
    seed=42,
)

obs = env.reset()
```

### Camera names available
- `robot0_agentview_center`
- `robot0_agentview_left`
- `robot0_agentview_right`
- `robot0_eye_in_hand`
- `robot0_frontview`

---

## 3. Observation Space

After `env.reset()`, `obs` is a dict. Key entries:

| Key | Description |
|---|---|
| `{cam}_image` | RGB image from camera, shape `(H, W, 3)`, rendered upside-down (flip with `img[::-1]`) |
| `robot0_eef_pos` | End-effector position in world frame |
| `robot0_eef_quat` | End-effector orientation (quaternion) |
| `robot0_gripper_qpos` | Gripper joint positions |
| `obj_pos` | Target object position |

Access object position directly from sim:
```python
body_id = env.obj_body_id["obj"]
apple_pos = env.sim.data.body_xpos[body_id].copy()  # world frame
```

---

## 4. Action Space

For `PandaOmron` with the default composite OSC controller, the action vector is **12-dimensional**:

| Indices | Description |
|---|---|
| `[0:3]` | EEF position delta (or absolute target in base frame if `input_type='absolute'`) |
| `[3:6]` | EEF orientation as rotation vector |
| `[6]` | Gripper: `-1` = open, `+1` = close |
| `[7:10]` | Mobile base velocity `[vx, vy, vrz]` |
| `[10:12]` | Padding (torso / mode flag) |

### Delta mode (default)
Actions are deltas clipped to `[-1, 1]` and scaled internally by the controller.

### Absolute mode
Set `ctrl_cfg['body_parts']['right']['input_type'] = 'absolute'` to send target poses directly in the robot base frame instead of deltas.

---

## 5. Stepping the Environment

```python
action = env.action_space.sample()   # or compute your own
obs, reward, done, info = env.step(action)

# Check task success
success = env.env._check_success()   # or env._check_success() depending on wrappers
```

---

## 6. Gymnasium Wrapper (for RL)

Wrap with `GymWrapper` for compatibility with SB3 / other RL libraries:

```python
from robosuite.wrappers.gym_wrapper import GymWrapper

env = GymWrapper(env, keys=None)
# keys=None is required — explicit key lists cause observation space mismatches (RoboCasa bug)
```

Then wrap with `Monitor` and vectorize:

```python
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

env = Monitor(env, log_dir="/tmp/gym/0")
vec_env = DummyVecEnv([lambda: env])        # single env
# or
vec_env = SubprocVecEnv([make_env_fn] * n)  # n parallel envs
```

---

## 7. RL Training (PPO example)

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_pnp")
```

Run the training script:
```bash
python helpers/rl_scripts/train_robocasa.py --n_envs 4 --headless
```

---

## 8. Evaluation

```bash
python helpers/rl_scripts/eval_robocasa.py \
    --model_path runs/ppo_pnp.zip \
    --episodes 10 \
    --save_video \
    --video_path eval_videos/
```

Video frames are rendered from all four cameras and tiled into a 2×2 grid.

```python
# Manual multi-camera render (inside eval loop)
frame = raw_env.sim.render(camera_name="robot0_agentview_center",
                            width=256, height=256, depth=False)
frame = np.flipud(frame)   # MuJoCo renders upside-down
```

Access the unwrapped env through wrapper chain:
```python
raw_env = env.env.env   # GymWrapper -> Monitor -> raw env
```

---

## 9. Scripted Demo (OSC Absolute)

`grasp_apple_absolute_gripper.py` implements a full pick-and-place pipeline without any learning, using a state machine:

```
HOME → HOVER → DESCEND → GRASP → LIFT → MOVE_TO_CAB → PLACE → RELEASE → RETURN_HOME → DONE
```

Run it:
```bash
python helpers/grasp_apple_absolute_gripper.py --seed 2 --save-video
```

This is useful for:
- Verifying the environment works before training
- Collecting scripted demonstrations for imitation learning
- Debugging reward functions

---

## 10. Applying to Other Tasks

The same pattern generalizes to any RoboCasa task:

1. Find the base class in `robocasa/environments/kitchen/`
2. Subclass it, override `_get_obj_cfgs()` and `reward()`
3. Register with `register_env(MyCustomEnv)`
4. Use `robosuite.make("MyCustomEnv", ...)` as above

Other available single-stage tasks include `PnPCounterToSink`, `TurnOnMicrowave`, `OpenDoor`, etc.
