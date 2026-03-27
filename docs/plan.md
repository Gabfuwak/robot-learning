# Training & Evaluation Pipeline Plan — PnP Counter To Cabinet

---

## Goals

- Generalized training pipeline configurable entirely via a YAML file
- Custom PyTorch actor/critic architectures (pluggable)
- Proper reward shaping
- Curriculum learning support
- Clean separation between: environment, reward, policy, training algo, evaluation

---

## Project Structure

```
robot-learning/
├── configs/
│   └── sac_pnp_baseline.yaml       # example config (see below)
├── envs/
│   └── pnp_counter_to_cab.py       # env factory + registration
├── rewards/
│   └── pnp_reward.py               # reward shaping functions
├── models/
│   ├── mlp.py                      # simple MLP actor/critic
│   └── cnn.py                      # CNN encoder for image obs (future)
├── curriculum/
│   └── scheduler.py                # curriculum stage logic
├── algos/
│   └── sac.py                      # SAC trainer wrapping SB3 or custom
├── train.py                        # main entry point — loads config, runs training
└── eval.py                         # evaluation + video rendering
```

---

## Config File Design

Everything lives in a single YAML. Example:

```yaml
# configs/sac_pnp_baseline.yaml

seed: 42
total_timesteps: 2_000_000

env:
  task: PickPlaceCounterToCabinet
  robot: PandaOmron
  horizon: 500
  control_freq: 20
  layout_ids: [1]
  style_ids: [1]
  use_camera_obs: false
  camera_names: []
  camera_height: 128
  camera_width: 128

reward:
  shaper: PnPRewardShaper        # class name in rewards/pnp_reward.py
  weights:
    reach: 1.0
    grasp: 2.0
    lift: 2.0
    place: 5.0
    success: 10.0

algo:
  name: SAC                      # SAC | PPO | TD3
  policy: MlpPolicy              # MlpPolicy | CustomMlpPolicy | CnnPolicy
  learning_rate: 3e-4
  buffer_size: 300_000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  ent_coef: auto
  learning_starts: 10_000
  train_freq: 1
  gradient_steps: 1

policy_arch:                     # passed to policy_kwargs
  actor:
    type: MLP                    # MLP | CNN | Transformer
    hidden_dims: [256, 256]
    activation: ReLU
  critic:
    type: MLP
    hidden_dims: [256, 256]
    activation: ReLU

curriculum:
  enabled: false
  stages:
    - name: reach_only
      until_timestep: 500_000
      reward_overrides:
        weights:
          reach: 1.0
          grasp: 0.0
          lift: 0.0
          place: 0.0
          success: 0.0
    - name: reach_and_grasp
      until_timestep: 1_000_000
      reward_overrides:
        weights:
          reach: 1.0
          grasp: 2.0
          lift: 0.0
          place: 0.0
          success: 0.0
    - name: full_task
      until_timestep: 2_000_000
      reward_overrides:
        weights:
          reach: 1.0
          grasp: 2.0
          lift: 2.0
          place: 5.0
          success: 10.0

eval:
  freq: 50_000                   # evaluate every N timesteps
  n_episodes: 10
  deterministic: true
  save_video: false
  video_path: eval_videos/

logging:
  checkpoint_freq: 50_000
  save_path: runs/
  tensorboard: true
  wandb: false
  wandb_project: robot-learning
```

---

## Component Design

### 1. Environment (`envs/pnp_counter_to_cab.py`)

- `make_env(cfg)` factory reads `env` block from config
- Wraps with `GymWrapper` + `Monitor`
- Injects the reward shaper so `env.reward()` calls it

### 2. Reward Shaping (`rewards/pnp_reward.py`)

Dense reward decomposed into stages:

```
r_total = w_reach   * r_reach        # EEF close to object
        + w_grasp   * r_grasp        # object lifted off surface
        + w_lift    * r_lift         # object above a height threshold
        + w_place   * r_place        # object close to cabinet target
        + w_success * r_success      # task success (sparse bonus)
```

- Each term is normalized to `[0, 1]`
- Weights configured in YAML
- Implement as a class with a `__call__(env) -> float` interface so it is easy to swap

### 3. Custom Policy Architectures (`models/`)

- Implement as PyTorch `nn.Module`
- Register with a string name so the config can refer to them by name
- Pass to SB3 via `policy_kwargs={"net_arch": ..., "features_extractor_class": ...}` or via a fully custom `ActorCriticPolicy` subclass for more control
- Start with MLP; add CNN encoder later when camera obs are needed
- For image-based policies: the model itself is responsible for how cameras are combined (concatenated embeddings, shared CNN, separate CNNs, etc.) — the pipeline just passes all image observations in as-is

### 4. Training Algo (`algos/sac.py`)

- Higher-level wrapper around SB3 `SAC` (or `PPO`, `TD3`) — custom training loop can be added later if needed
- Reads `algo` + `policy_arch` blocks from config
- Builds `policy_kwargs` from `policy_arch` config and passes to SB3
- Attaches callbacks (checkpoint, eval, curriculum)
- Interface: `Trainer(cfg).learn()` — consistent regardless of which SB3 algo is underneath

### 5. Curriculum (`curriculum/scheduler.py`)

- `CurriculumCallback(BaseCallback)` checks current timestep against stage thresholds
- On stage transition: updates the reward shaper weights in-place — no env rebuild needed
- Stages defined entirely in YAML as `reward_overrides` — no code changes needed to add a new stage

### 6. Train Entry Point (`train.py`)

```
load YAML config
→ build reward shaper
→ build env (injecting reward shaper)
→ build model (algo + policy arch from config)
→ attach callbacks (checkpoint, eval, curriculum)
→ model.learn(total_timesteps)
→ save final model
```

### 7. Eval Entry Point (`eval.py`)

- Load model from `.zip`
- Load matching env config from the same YAML
- Run N episodes, record: success rate, mean reward, mean episode length
- Optionally render and save video (tiled multi-camera)

---

## Implementation Order

1. **Reward shaping** — most impactful, unblocks everything else
2. **Config loader + `train.py`** — wire up YAML → SB3 with existing MlpPolicy first
3. **Custom policy arch** — plug in PyTorch MLP actor/critic via `policy_kwargs`
4. **Eval script** — success rate + optional video
5. **Curriculum** — add `CurriculumCallback` last once base pipeline is stable

---

## Resolved Design Decisions

- **Curriculum** only changes reward weights between stages — no env rebuild needed
- **Image-based policies** handle camera fusion internally (shared CNN, separate CNNs, concat, etc.) — the pipeline is agnostic
- **Training loop** uses a higher-level SB3 wrapper for now; custom loop can be added later if needed (e.g. auxiliary losses)
