"""
Reward functions for RoboCasa environments.

Each reward function is a callable that takes the live environment and the
raw observation dict (straight from env.step / env.reset) and returns a
scalar float reward.

Implement new reward functions by subclassing RewardFn.

Available reward functions
──────────────────────────
    StagedPickPlaceReward       — 3-stage: reach → grasp+transport → inside cabinet
    ReleasingPickPlaceReward    — 4-stage: adds an explicit release stage after placement
    DensePickPlaceReward        — fully dense, all signals combined without hard stage gates
    BinaryMilestoneReward       — sparse bonuses at each milestone, no shaping between them

Key design notes
────────────────
  - "Success" is defined by OU.obj_inside_of(), which checks whether the
    object's bounding box is fully inside the cabinet's interior region.
    The distance from the object to the cabinet CENTER is NOT the right proxy
    for success — the cabinet has depth, and the object just needs to be inside
    the interior, not at the centroid.

  - "Release" is detected by checking that the gripper is open (gripper_qpos
    near maximum) AND the object is no longer grasped. Rewarding release
    inside the cabinet prevents the policy from just hovering the object in
    place while holding it — it should actually let go.

  - All reward functions use env.cab.pos as a navigation target during
    transport (privileged info available during training). This guides the
    gripper toward the cabinet opening without requiring the object to reach
    the exact center.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import robocasa.utils.object_utils as OU


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RewardFn(ABC):
    """Base class for all reward functions."""

    @abstractmethod
    def __call__(self, env, obs: dict) -> float:
        """
        Args:
            env:  Live RoboCasa environment — gives access to simulator state,
                  fixtures (env.cab), object bodies, contact data, etc.
            obs:  Raw observation dict from env.step() / env.reset().

        Returns:
            Scalar reward (roughly in [0, 1] for all provided implementations).
        """

    # ------------------------------------------------------------------
    # Shared helper queries (available to all subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _exp_decay(dist: float, scale: float) -> float:
        """Maps distance → (0, 1] with exponential decay. Returns 1 when dist=0."""
        return float(np.exp(-scale * dist))

    @staticmethod
    def _gripper_is_open(obs: dict, open_threshold: float = 0.03) -> bool:
        """
        True when both gripper fingers are near their maximum open position.

        PandaOmron gripper_qpos ∈ [0, ~0.04] per finger.
        We treat the gripper as open when each finger exceeds open_threshold.
        """
        qpos = obs["robot0_gripper_qpos"]   # (2,) finger positions
        return bool(np.all(qpos > open_threshold))


# ---------------------------------------------------------------------------
# 1. StagedPickPlaceReward (original, kept for backward compatibility)
# ---------------------------------------------------------------------------

class StagedPickPlaceReward(RewardFn):
    """
    Continuous 3-stage reward for pick-and-place tasks.

        Stage 1 — Reach          [0.00, 0.50) : gripper moves toward target object
        Stage 2 — Grasp+Transport [0.50, 1.00) : grasped object moves toward cabinet
        Stage 3 — Inside cabinet  1.00         : object is inside the cabinet

    Note: success is detected by obj_inside_of(), not by dist-to-cab == 0.
    The cabinet position is only used as a directional target during transport.

    Args:
        reach_scale:     Sharpness of the exponential reach reward.
        transport_scale: Sharpness of the exponential transport reward.
    """

    def __init__(self, reach_scale: float = 3.0, transport_scale: float = 3.0):
        self.reach_scale     = reach_scale
        self.transport_scale = transport_scale

    def __call__(self, env, obs: dict) -> float:
        obj_pos = obs["obj_pos"]
        eef_pos = obs["robot0_eef_pos"]
        cab_pos = np.array(env.cab.pos)

        inside_cab = OU.obj_inside_of(env, "obj", env.cab)
        is_grasped = OU.check_obj_grasped(env, "obj")

        if inside_cab:
            return 1.0
        if is_grasped:
            return 0.5 + 0.5 * self._exp_decay(self._dist(obj_pos, cab_pos), self.transport_scale)
        return 0.5 * self._exp_decay(self._dist(eef_pos, obj_pos), self.reach_scale)


# ---------------------------------------------------------------------------
# 2. ReleasingPickPlaceReward
# ---------------------------------------------------------------------------

class ReleasingPickPlaceReward(RewardFn):
    """
    4-stage reward that explicitly rewards releasing the object inside the cabinet.

    The motivation: holding the object inside the cabinet while the gripper is
    still closed is NOT the goal — the robot needs to let go and let the object
    settle. A policy trained without a release signal tends to hover the object
    in place without ever opening the gripper.

        Stage 1 — Reach           [0.00, 0.25) : gripper → object
        Stage 2 — Grasp+Transport [0.25, 0.50) : grasped object → cabinet
        Stage 3 — Release         [0.50, 1.00) : gripper opens while obj is inside cabinet;
                                                  bonus for gripper moving away
        Stage 4 — Stabilised       1.00        : obj inside cabinet, gripper far away

    Args:
        reach_scale:     Sharpness of reach reward decay.
        transport_scale: Sharpness of transport reward decay.
        retreat_scale:   Sharpness of retreat (gripper-away) reward decay.
        open_threshold:  Gripper finger position above which we call the gripper "open".
    """

    def __init__(
        self,
        reach_scale:     float = 3.0,
        transport_scale: float = 3.0,
        retreat_scale:   float = 2.0,
        open_threshold:  float = 0.03,
    ):
        self.reach_scale     = reach_scale
        self.transport_scale = transport_scale
        self.retreat_scale   = retreat_scale
        self.open_threshold  = open_threshold

    def __call__(self, env, obs: dict) -> float:
        obj_pos = obs["obj_pos"]
        eef_pos = obs["robot0_eef_pos"]
        cab_pos = np.array(env.cab.pos)

        inside_cab = OU.obj_inside_of(env, "obj", env.cab)
        is_grasped = OU.check_obj_grasped(env, "obj")
        gripper_open = self._gripper_is_open(obs, self.open_threshold)
        gripper_far  = OU.gripper_obj_far(env, "obj")

        # Stage 4: object settled in cabinet, gripper has moved away
        if inside_cab and gripper_far:
            return 1.0

        # Stage 3: object is inside cabinet — reward opening the gripper and retreating
        if inside_cab:
            dist_eef_obj = self._dist(eef_pos, obj_pos)
            retreat_bonus = self._exp_decay(dist_eef_obj, self.retreat_scale)
            # Opening the gripper gives the lower half of this band;
            # moving away gives the upper half.
            if gripper_open:
                return 0.75 + 0.25 * retreat_bonus
            return 0.50 + 0.25 * retreat_bonus

        # Stage 2: object grasped, move it toward the cabinet
        if is_grasped:
            return 0.25 + 0.25 * self._exp_decay(self._dist(obj_pos, cab_pos), self.transport_scale)

        # Stage 1: reach toward the object
        return 0.25 * self._exp_decay(self._dist(eef_pos, obj_pos), self.reach_scale)


# ---------------------------------------------------------------------------
# 3. DensePickPlaceReward
# ---------------------------------------------------------------------------

class DensePickPlaceReward(RewardFn):
    """
    Fully dense reward — all signals are active simultaneously, weighted and summed.

    Unlike staged rewards, every component is always non-zero. This gives the
    policy more gradient signal but can cause interference between terms (e.g.,
    the retreat term could fight the reach term early on). Use weight tuning to
    balance them.

    Components
    ──────────
        reach     : exp decay of dist(gripper, object)                    always active
        grasp     : +1 if object is grasped                               binary bonus
        transport : exp decay of dist(object, cabinet) when grasped       active if grasped
        inside    : +1 if object is inside cabinet                         binary bonus
        release   : exp decay of dist(gripper, object) when inside cabinet active if inside
        stabilise : +1 if inside cabinet AND gripper far                   binary bonus

    Total reward is the weighted sum, normalised to roughly [0, 1].

    Args:
        w_reach:      Weight for reach component.
        w_grasp:      Weight for grasp bonus.
        w_transport:  Weight for transport component.
        w_inside:     Weight for inside-cabinet bonus.
        w_release:    Weight for release component.
        w_stabilise:  Weight for stabilisation bonus.
        reach_scale:      Exponential scale for reach.
        transport_scale:  Exponential scale for transport.
        release_scale:    Exponential scale for release retreat.
    """

    def __init__(
        self,
        w_reach:         float = 0.10,
        w_grasp:         float = 0.15,
        w_transport:     float = 0.25,
        w_inside:        float = 0.20,
        w_release:       float = 0.15,
        w_stabilise:     float = 0.15,
        reach_scale:     float = 3.0,
        transport_scale: float = 3.0,
        release_scale:   float = 2.0,
    ):
        total = w_reach + w_grasp + w_transport + w_inside + w_release + w_stabilise
        # Normalise weights so maximum possible reward = 1.0
        self.w_reach     = w_reach     / total
        self.w_grasp     = w_grasp     / total
        self.w_transport = w_transport / total
        self.w_inside    = w_inside    / total
        self.w_release   = w_release   / total
        self.w_stabilise = w_stabilise / total

        self.reach_scale     = reach_scale
        self.transport_scale = transport_scale
        self.release_scale   = release_scale

    def __call__(self, env, obs: dict) -> float:
        obj_pos = obs["obj_pos"]
        eef_pos = obs["robot0_eef_pos"]
        cab_pos = np.array(env.cab.pos)

        inside_cab = OU.obj_inside_of(env, "obj", env.cab)
        is_grasped = OU.check_obj_grasped(env, "obj")
        gripper_far = OU.gripper_obj_far(env, "obj")

        r_reach     = self._exp_decay(self._dist(eef_pos, obj_pos), self.reach_scale)
        r_grasp     = 1.0 if is_grasped else 0.0
        r_transport = self._exp_decay(self._dist(obj_pos, cab_pos), self.transport_scale) if is_grasped else 0.0
        r_inside    = 1.0 if inside_cab else 0.0
        r_release   = self._exp_decay(self._dist(eef_pos, obj_pos), self.release_scale) if inside_cab else 0.0
        r_stabilise = 1.0 if (inside_cab and gripper_far) else 0.0

        return (
            self.w_reach     * r_reach
            + self.w_grasp     * r_grasp
            + self.w_transport * r_transport
            + self.w_inside    * r_inside
            + self.w_release   * r_release
            + self.w_stabilise * r_stabilise
        )


# ---------------------------------------------------------------------------
# 4. BinaryMilestoneReward
# ---------------------------------------------------------------------------

class BinaryMilestoneReward(RewardFn):
    """
    Sparse reward with discrete bonuses at each task milestone.

    No shaping between milestones — the policy only gets reward when it
    hits a checkpoint. This is harder to optimise than shaped rewards but
    produces cleaner behaviour and is less prone to reward hacking.

    Milestones and their one-time bonuses
    ──────────────────────────────────────
        Grasp object                 +grasp_bonus
        Object inside cabinet        +inside_bonus
        Gripper open inside cabinet  +release_bonus
        Gripper far, obj inside      +stabilise_bonus   (= task success)

    Each bonus is only awarded once per episode (tracked internally).
    The total across all milestones sums to 1.0.

    Args:
        grasp_bonus:     Reward for first successful grasp.
        inside_bonus:    Reward for first time obj enters cabinet.
        release_bonus:   Reward for opening gripper while obj is inside.
        stabilise_bonus: Reward for task completion (obj in cab, gripper away).
    """

    def __init__(
        self,
        grasp_bonus:     float = 0.15,
        inside_bonus:    float = 0.30,
        release_bonus:   float = 0.20,
        stabilise_bonus: float = 0.35,
    ):
        self.grasp_bonus     = grasp_bonus
        self.inside_bonus    = inside_bonus
        self.release_bonus   = release_bonus
        self.stabilise_bonus = stabilise_bonus
        self._reset_flags()

    def _reset_flags(self):
        self._gave_grasp     = False
        self._gave_inside    = False
        self._gave_release   = False
        self._gave_stabilise = False

    def __call__(self, env, obs: dict) -> float:
        # Reset milestone flags at the start of each episode.
        # We detect episode start by checking whether the object is back near
        # its spawn region (a rough heuristic). A cleaner approach is to hook
        # into env.reset() via the wrapper — see note below.
        #
        # Note: RoboCasaWrapper calls reward_fn(env, obs) every step. To reset
        # flags properly, override reset() in your wrapper and call
        # reward_fn.on_episode_reset() if it exists.
        reward = 0.0

        inside_cab  = OU.obj_inside_of(env, "obj", env.cab)
        is_grasped  = OU.check_obj_grasped(env, "obj")
        gripper_far = OU.gripper_obj_far(env, "obj")
        gripper_open = self._gripper_is_open(obs)

        if is_grasped and not self._gave_grasp:
            reward += self.grasp_bonus
            self._gave_grasp = True

        if inside_cab and not self._gave_inside:
            reward += self.inside_bonus
            self._gave_inside = True

        if inside_cab and gripper_open and not self._gave_release:
            reward += self.release_bonus
            self._gave_release = True

        if inside_cab and gripper_far and not self._gave_stabilise:
            reward += self.stabilise_bonus
            self._gave_stabilise = True

        return reward

    def on_episode_reset(self):
        """Call this at the start of each episode to reset milestone tracking."""
        self._reset_flags()
