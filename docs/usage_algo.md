# Algorithm Architecture — Encoders, Actor, and Critic

## Overview

The full policy is composed of three stacked pieces:

```
observation (Dict)
      │
      ▼
RoboCasaFeaturesExtractor          ← rl/architecture.py
  ├── StateEncoder  (MLP)
  └── ImageEncoder  (CNN)  × N cameras
      │  concat
      ▼
features vector  (state_embed_dim + image_embed_dim × N)
      │
      ├──▶ actor MLP  ──▶ action mean / log_std     (SAC)
      │                   action logits             (PPO)
      │
      └──▶ critic MLP ──▶ Q-value(s) / V-value
```

`net_arch` in `TrainConfig` controls the hidden layers of **both** the actor and critic MLPs. The feature extractor sits upstream and is shared.

---

## Are the Encoders Learned?

**Yes.** `RoboCasaFeaturesExtractor` is a standard `nn.Module` that is part of the policy network. Its weights are initialised randomly and updated by gradient descent along with every other parameter (actor head, critic head) during training. There is no frozen pre-trained backbone by default.

Concretely, during a SAC update:

1. Critic loss is backpropagated through the critic MLP **and** through the feature extractor.
2. Actor loss is backpropagated through the actor MLP **and** through the feature extractor (shared weights, so the encoder receives gradients from both losses).

The encoder therefore learns to produce representations that are jointly useful for predicting Q-values and for selecting actions.

---

## Extracting Sub-components After Training

SB3 stores everything inside `model.policy`. The exact attribute names differ slightly between off-policy (SAC, TD3, DDPG) and on-policy (PPO, A2C) algorithms.

### Loading a saved model

```python
from stable_baselines3 import SAC, PPO

# Off-policy
model = SAC.load("runs/my_run/sac_final")

# On-policy
model = PPO.load("runs/my_run/ppo_final")
```

### Feature extractor (encoder)

The extractor is the same attribute for all algorithms:

```python
extractor = model.policy.features_extractor   # RoboCasaFeaturesExtractor

state_enc = extractor.state_encoder           # StateEncoder  (nn.Module)
img_encs  = extractor.image_encoders          # nn.ModuleDict, one ImageEncoder per camera
```

Running the encoder standalone:

```python
import torch

# obs_dict mirrors what RoboCasaWrapper produces
obs_dict = {
    "state":            torch.zeros(1, 110),
    "image_robot0_eye_in_hand": torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
}
extractor.eval()
with torch.no_grad():
    features = extractor(obs_dict)   # (1, features_dim)
```

### SAC — actor and critic

```python
actor  = model.policy.actor          # sac/policies.py Actor (MLP + Gaussian head)
critic = model.policy.critic         # sac/policies.py ContinuousCritic (twin Q-nets)
# critic_target is a frozen copy kept for stable bootstrapping:
critic_target = model.policy.critic_target
```

Running them standalone (they call the extractor internally):

```python
actor.eval()
with torch.no_grad():
    # returns (mean, log_std) — or sample an action:
    action, log_prob, mean = actor.get_action(obs_dict)
    # Q-values from both heads:
    q1, q2 = critic(obs_dict, action)
```

### PPO — actor and critic

PPO uses a shared MLP trunk (`mlp_extractor`) on top of the feature extractor, then separate linear heads:

```python
mlp_extractor = model.policy.mlp_extractor   # shared latent MLP (policy + value streams)
action_net    = model.policy.action_net       # linear head → action logits / mean
value_net     = model.policy.value_net        # linear head → scalar V
```

Running them standalone:

```python
model.policy.eval()
with torch.no_grad():
    features    = model.policy.features_extractor(obs_dict)          # encoder
    latent_pi, latent_vf = model.policy.mlp_extractor(features)      # shared MLP
    action_logits = model.policy.action_net(latent_pi)
    value         = model.policy.value_net(latent_vf)
```

Or use the high-level policy API directly:

```python
with torch.no_grad():
    actions, values, log_probs = model.policy(obs_dict)
```

### Exporting an encoder to a separate file

```python
import torch

encoder = model.policy.features_extractor
torch.save(encoder.state_dict(), "encoder_weights.pt")

# Reload later (observation_space must match):
from rl.architecture import RoboCasaFeaturesExtractor
import gymnasium as gym

obs_space = ...  # same Dict space used during training
new_enc = RoboCasaFeaturesExtractor(obs_space, state_embed_dim=256, image_embed_dim=128)
new_enc.load_state_dict(torch.load("encoder_weights.pt"))
new_enc.eval()
```

---

## Using a Pretrained (Frozen) Image Encoder

You may want to load weights from a pretrained CNN (e.g. from a prior run or a separate visual pretraining stage) and keep them frozen so only the state encoder and actor/critic heads are trained.

### PPO / A2C — single extractor

```python
import torch

model = PPO(...)   # construct model first; do NOT call learn() yet

extractor = model.policy.features_extractor

# Load pretrained weights
pretrained = torch.load("pretrained_image_encoder.pt")

# Single camera:
extractor.image_encoders["image_robot0_eye_in_hand"].load_state_dict(pretrained)
# All cameras at once (if you saved the whole ModuleDict):
extractor.image_encoders.load_state_dict(pretrained)

# Freeze
for param in extractor.image_encoders.parameters():
    param.requires_grad = False
```

### SAC / TD3 — actor and critic each own a separate extractor

In off-policy algorithms the actor and critic hold independent copies of the feature extractor, so both must be updated:

```python
model = SAC(...)

for extractor in [
    model.policy.actor.features_extractor,
    model.policy.critic.features_extractor,
]:
    extractor.image_encoders.load_state_dict(torch.load("pretrained_image_encoder.pt"))
    for param in extractor.image_encoders.parameters():
        param.requires_grad = False
```

### Rebuilding the optimizer after freezing

SB3 constructs the optimizer at `__init__` time, collecting all `requires_grad=True` parameters. If you freeze after construction, the frozen params are already inside the optimizer's parameter groups — they receive no gradients so the update is a no-op, but they waste memory and compute. To exclude them cleanly, rebuild the optimizer before calling `learn()`:

```python
model.policy.optimizer = model.policy.optimizer_class(
    filter(lambda p: p.requires_grad, model.policy.parameters()),
    lr=cfg.learning_rate,
    **model.policy.optimizer_kwargs,
)

model.learn(total_timesteps=cfg.total_timesteps)
```

---

## Quick reference — attribute map

| Component | SAC / TD3 / DDPG | PPO / A2C |
|---|---|---|
| Feature extractor | `model.policy.features_extractor` | `model.policy.features_extractor` |
| State encoder | `.features_extractor.state_encoder` | `.features_extractor.state_encoder` |
| Image encoder(s) | `.features_extractor.image_encoders[key]` | `.features_extractor.image_encoders[key]` |
| Shared MLP trunk | — | `model.policy.mlp_extractor` |
| Actor head | `model.policy.actor` | `model.policy.action_net` |
| Critic / value head | `model.policy.critic` | `model.policy.value_net` |
| Target critic | `model.policy.critic_target` | — |
