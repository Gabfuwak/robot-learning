# Demo Dataset — IL Pipeline Reference

This document describes the demo dataset as seen by the imitation learning
pipeline: what is stored on disk, why sim replay is required to extract usable
training observations, what each data dimension means, and how the public API
in `il/base.py` exposes it.

For dataset download instructions and environment/observation details see
`docs/usage_dataset.md` and `docs/usage_env.md`.

---

## 1. Two representations of the same data

The dataset lives on disk in **LeRobot format**. It stores two kinds of data
for every timestep:

| Representation | Dims | Where | Used by |
|---|---|---|---|
| **Parquet state** | 16 | `data/**/*.parquet` | nothing directly in our pipeline |
| **Replayed state** | 110 | built by `collect_demo_data()` | all IL algorithms |

The parquet state is a minimal snapshot of the robot (base + EEF + gripper) —
it is enough to reconstruct the simulator but not enough to serve as policy
input, because it lacks joint angles, joint velocities, accelerations, and
all object positions.

To obtain the full 110-dim observation that the policy sees at test time,
`collect_demo_data()` re-plays every recorded MuJoCo state through the
simulator and calls `env._get_observations()` at each timestep. The result is
cached to disk so subsequent runs skip the replay.

---

## 2. On-disk layout

```
<dataset_root>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # per-episode low-dim data
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.robot0_agentview_left/
│       ├── observation.images.robot0_agentview_right/
│       └── observation.images.robot0_eye_in_hand/
├── extras/                          # RoboCasa-specific, not part of LeRobot spec
│   ├── dataset_meta.json            # env_kwargs used to recreate the environment
│   └── episode_000000/
│       ├── states.npz               # MuJoCo sim states (T, sim_dim) compressed
│       ├── model.xml.gz             # full MJCF XML for this episode's kitchen
│       └── ep_meta.json             # layout_id, style_id, object placement, …
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── stats.json
    ├── modality.json                # action / state key → slice mapping
    └── embodiment.json
```

The `extras/` folder is the key for replay. It stores the full MuJoCo state
vector at every timestep (not just the 16-dim parquet subset) plus the exact
MJCF XML that produced the kitchen layout so the simulator can be restored
exactly.

---

## 3. Parquet contents

Each `episode_*.parquet` has one row per timestep. Relevant columns:

### action (12-dim)

Stored in **LeRobot ordering** (defined in `lerobot_utils.ACTION_KEY_ORDERING_HDF5`):

| Indices | Semantic | Always zero? |
|---|---|---|
| `[0:3]` | End-effector position delta (x, y, z) | No |
| `[3:6]` | End-effector rotation delta (rotation vector) | No |
| `[6]` | Gripper: −1 = open, +1 = close | No |
| `[7:11]` | Mobile base velocity (vx, vy, vθ, padding) | **Yes — base never moves in demos** |
| `[11]` | Control mode flag | **Yes — always 0** |

> **Important**: 5 of the 12 action dimensions (`[7:11]` base motion and `[11]`
> control mode) are **always zero** across all demonstrations. This matters for
> diffusion policy: MIN_MAX normalization maps these to a constant −1 (with the
> 1e-8 epsilon denominator), and the UNet wastes capacity predicting a trivial
> constant. See the gotchas section.

`get_episode_actions()` in `lerobot_utils.py` reads the parquet and
**re-orders** columns from LeRobot ordering back to the HDF5 ordering that the
environment expects. The final 12-dim array returned by `collect_demo_data()`
is in **environment / HDF5 ordering**:

| Indices | Semantic |
|---|---|
| `[0:3]` | End-effector position delta |
| `[3:6]` | End-effector rotation delta |
| `[6]` | Gripper |
| `[7:10]` | Mobile base velocity (vx, vy, vθ) |
| `[10:12]` | Padding / control mode — always 0 |

### observation.state (16-dim, parquet only)

Not used by our IL pipeline directly. Included here for reference.

| Indices | Source key | Description |
|---|---|---|
| `[0:3]` | `robot0_base_pos` | Mobile base world position |
| `[3:7]` | `robot0_base_quat` | Mobile base world orientation (quaternion) |
| `[7:10]` | `robot0_base_to_eef_pos` | EEF position relative to base |
| `[10:14]` | `robot0_base_to_eef_quat` | EEF orientation relative to base |
| `[14:16]` | `robot0_gripper_qpos` | Gripper finger joint positions |

---

## 4. Replayed state (110-dim)

This is what `collect_demo_data()` produces and what every policy actually
receives as input. It concatenates two aggregated observation keys:

```
robot0_proprio-state  (68-dim)  +  object-state  (42-dim)  =  110-dim
```

See `docs/usage_env.md` for the full per-field breakdown. Brief summary:

| Slice | Source | Description |
|---|---|---|
| `[0:7]` | `robot0_joint_pos` | Arm joint positions (radians) |
| `[7:14]` | `robot0_joint_pos_cos` | Cosine encoding |
| `[14:21]` | `robot0_joint_pos_sin` | Sine encoding |
| `[21:28]` | `robot0_joint_vel` | Arm joint velocities |
| `[28:35]` | `robot0_joint_acc` | Arm joint accelerations |
| `[35:38]` | `robot0_eef_pos` | EEF world position |
| `[38:42]` | `robot0_eef_quat` | EEF world orientation |
| `[42:46]` | `robot0_eef_quat_site` | EEF world orientation (site) |
| `[46:48]` | `robot0_gripper_qpos` | Gripper finger positions |
| `[48:50]` | `robot0_gripper_qvel` | Gripper finger velocities |
| `[50:53]` | `robot0_base_pos` | Mobile base world position |
| `[53:57]` | `robot0_base_quat` | Mobile base world orientation |
| `[57:60]` | `robot0_base_to_eef_pos` | EEF relative to base |
| `[60:64]` | `robot0_base_to_eef_quat` | EEF orientation relative to base |
| `[64:68]` | `robot0_base_to_eef_quat_site` | EEF orientation relative to base (site) |
| `[68:82]` | `obj_*` | Target object: pos(3) + quat(4) + to_eef_pos(3) + to_eef_quat(4) |
| `[82:96]` | `distr_counter_*` | Counter distractor: same 14-dim structure |
| `[96:110]` | `distr_cab_*` | Cabinet distractor: same 14-dim structure |

> `robot0_base_quat` (dims `[53:57]`) and one additional state dim are
> **constant** across all demos because the mobile base does not rotate
> during pick-and-place. This mirrors the 5 constant action dims.

---

## 5. Splits and sources

| `split` | `source` | Description | Typical size |
|---|---|---|---|
| `"target"` | `"human"` | Human tele-op demos on test layouts/styles | ~500 episodes |
| `"pretrain"` | `"human"` | Human tele-op demos on training layouts/styles | ~thousands |
| `"pretrain"` | `"mg"` | MimicGen synthetic augmentation of human demos | ~tens of thousands |

**Layout/style split for evaluation**:

| `layout_ids` / `style_ids` | Name | IDs |
|---|---|---|
| `-1` | TEST  | 1–10 |
| `-2` | TRAIN | 11–60 |
| `-3` | ALL   | 1–60 |

The `"target"` split was collected on TEST layouts/styles (`-1`). The
`"pretrain"` split was collected on TRAIN layouts/styles (`-2`).

---

## 6. IL pipeline API (`il/base.py`)

### `collect_demo_data(cfg: ILConfig) → (states, actions)`

Returns flat numpy arrays across all demos concatenated:

```python
states  : np.ndarray  shape (N, 110)  float32   # replayed full observations
actions : np.ndarray  shape (N, 12)   float32   # environment-ordered actions
```

`N` = total number of timesteps across all episodes (typically ~100k for the
500-demo target split).

**Replay process** (runs once, then cached):
1. Reads `extras/dataset_meta.json` to rebuild the environment.
2. For each episode, loads `extras/episode_*/model.xml.gz` (kitchen layout)
   and `extras/episode_*/states.npz` (per-timestep MuJoCo state).
3. Restores each timestep's MuJoCo state with `env.sim.set_state_from_flattened()`
   and calls `env._get_observations(force_update=True)` to get the full obs dict.
4. Concatenates `robot0_proprio-state` (68) + `object-state` (42).

> `force_update=True` is required. Setting sim state directly does not
> invalidate the observation cache — without it every timestep returns the
> same frozen observation from the initial reset.

### `collect_episode_data(cfg: ILConfig) → list[(states, actions)]`

Same as `collect_demo_data` but preserves episode boundaries. Returns a list
of `(states, actions)` tuples, one per episode:

```python
episode_data[i]  →  (states_i, actions_i)
# states_i  : (T_i, 110)  float32
# actions_i : (T_i, 12)   float32
```

Used by Diffusion Policy (`DemoSequenceDataset`), which needs temporal
structure within episodes to build sliding-window samples with observation
history and action chunks.

Internally it just calls `collect_demo_data()` (using the cache) and then
re-splits by reading episode lengths from the parquet action files — no extra
sim replay needed.

---

## 7. Disk cache

Replay is expensive (~minutes for 500 demos). Results are cached under
`cfg.cache_dir` (default `.demo_cache/`) as two `.npy` files:

```
.demo_cache/
└── {task}_{split}_{source}_{n_demos}_states.npy
└── {task}_{split}_{source}_{n_demos}_actions.npy
```

The cache key encodes `task`, `split`, `source`, and `n_demos`. If any of
these change, new files are written alongside the old ones (old files are not
deleted automatically). If you change the observation keys or state extraction
logic in `_extract_state_vec()`, **delete the cache manually** — the key does
not include a content hash.

```bash
rm -rf .demo_cache/
```

---

## 8. Gotchas

**5/12 action dimensions are always zero.**
`[7:10]` (base velocity) and `[10:12]` (padding/control mode) are constant 0
across all human demonstrations — the base was never moved. Policies that use
MIN_MAX normalization (e.g. Diffusion Policy) map these to a constant −1 after
normalization and back to 0 after unnormalization. Training is not corrupted
(LeRobot uses a 1e-8 epsilon), but 42% of the action space is trivially
predictable, diluting the denoising gradient.

**2/68 state dimensions are constant.**
`robot0_base_quat` dims 53–54 (the z-axis rotation components) are constant
across all demos for the same reason. Same normalization concern applies.

**Replay must match the test-time observation pipeline.**
`_extract_state_vec()` concatenates `robot0_proprio-state` and `object-state`
in that order, which exactly matches what `RoboCasaWrapper._extract_state()`
does at test time. If you change the keys or their order in either place,
update both.

**Episode lengths vary.**
Human demos for `PickPlaceCounterToCabinet` range roughly from 80 to 500
steps. `collect_episode_data()` preserves this variance. Algorithms that
require fixed-length sequences (e.g. a vanilla transformer) need to handle
padding externally.
