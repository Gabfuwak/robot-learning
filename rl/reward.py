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
    ComposedPickPlaceReward     — additive components: r_reach + r_lift + r_transport + success

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

  - Reach reward is delta-based: positive when the gripper moves closer to
    the target object, negative when it moves away, and exactly 0 once the
    gripper successfully grasps the object. Each subclass stores
    _prev_reach_dist and resets it on episode start via on_episode_reset().
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
    def __call__(self, env, obs: dict, action=None) -> float:
        """
        Args:
            env:    Live RoboCasa environment — gives access to simulator state,
                    fixtures (env.cab), object bodies, contact data, etc.
            obs:    Raw observation dict from env.step() / env.reset().
            action: The action taken (np.ndarray). Optional; used by GAIL reward.

        Returns:
            Scalar reward. Reach component is delta-based (see module docstring).
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

    def _delta_reach(self, curr_dist: float, is_grasped: bool) -> float:
        """
        Delta-based reach reward:
          - Returns 0 when grasped (reach phase is over); also resets stored dist
            so the next reach phase (if the object is dropped) starts fresh.
          - Returns 0 on the very first call of an episode (no previous distance).
          - Returns prev_dist - curr_dist otherwise:
              > 0  when gripper is approaching the object
              < 0  when gripper is moving away
        Subclasses must initialise  self._prev_reach_dist = None  in __init__
        and call  self._prev_reach_dist = None  in on_episode_reset().
        """
        if is_grasped:
            self._prev_reach_dist = None
            return 0.0
        if self._prev_reach_dist is None:
            self._prev_reach_dist = curr_dist
            return 0.0
        delta = self._prev_reach_dist - curr_dist   # positive = closer
        self._prev_reach_dist = curr_dist
        return delta

    def _delta_transport(self, curr_dist: float, is_grasped: bool, inside_cab: bool) -> float:
        """
        Delta-based transport reward:
          - Returns 0 when not grasped (transport phase not active); resets stored
            dist so the next transport phase starts fresh if grasped again.
          - Returns 0 when the object is already inside the cabinet.
          - Returns 0 on the very first step the object is grasped (no previous dist).
          - Returns prev_dist - curr_dist otherwise:
              > 0  when the held object is moving closer to the cabinet
              < 0  when the held object is moving away
        Subclasses must initialise  self._prev_transport_dist = None  in __init__
        and call  self._prev_transport_dist = None  in on_episode_reset().
        """
        if not is_grasped or inside_cab:
            self._prev_transport_dist = None
            return 0.0
        if self._prev_transport_dist is None:
            self._prev_transport_dist = curr_dist
            return 0.0
        delta = self._prev_transport_dist - curr_dist   # positive = closer to cabinet
        self._prev_transport_dist = curr_dist
        return delta


# ---------------------------------------------------------------------------
# 1. StagedPickPlaceReward (original, kept for backward compatibility)
# ---------------------------------------------------------------------------

class StagedPickPlaceReward(RewardFn):
    """
    Continuous 3-stage reward for pick-and-place tasks.

        Stage 1 — Reach      delta-based: positive when gripper approaches object,
                             negative when retreating, 0 on first step.
        Stage 2 — Transport  delta-based: positive when held object moves closer to
                             cabinet, negative when moving away, 0 on first grasped step.
        Stage 3 — Inside cabinet  1.0 (terminal).

    Note: success is detected by obj_inside_of(), not by dist-to-cab == 0.
    The cabinet position is only used as a directional target during transport.
    """

    def __init__(self):
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def on_episode_reset(self):
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def __call__(self, env, obs: dict, action=None) -> float:
        obj_pos = obs["obj_pos"]
        eef_pos = obs["robot0_eef_pos"]
        cab_pos = np.array(env.cab.pos)

        inside_cab = OU.obj_inside_of(env, "obj", env.cab)
        is_grasped = OU.check_obj_grasped(env, "obj")

        if inside_cab:
            self._prev_reach_dist     = None
            self._prev_transport_dist = None
            return 1.0
        if is_grasped:
            # reach phase is over — reset its state
            self._delta_reach(self._dist(eef_pos, obj_pos), is_grasped=True)
            return self._delta_transport(self._dist(obj_pos, cab_pos), is_grasped=True, inside_cab=False)

        # Stage 1: delta-based reach reward
        self._delta_transport(0.0, is_grasped=False, inside_cab=False)   # keep transport state reset
        return self._delta_reach(self._dist(eef_pos, obj_pos), is_grasped=False)


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

        Stage 1 — Reach           delta-based: positive when approaching,
                                  negative when retreating, 0 when grasped.
        Stage 2 — Grasp+Transport [0.25, 0.50) : grasped object → cabinet
        Stage 3 — Release         [0.50, 1.00) : gripper opens while obj is inside cabinet;
                                                  bonus for gripper moving away
        Stage 4 — Stabilised       1.00        : obj inside cabinet, gripper far away

    Args:
        retreat_scale:  Sharpness of retreat (gripper-away) reward decay.
        open_threshold: Gripper finger position above which we call the gripper "open".
    """

    def __init__(
        self,
        retreat_scale:  float = 2.0,
        open_threshold: float = 0.03,
    ):
        self.retreat_scale   = retreat_scale
        self.open_threshold  = open_threshold
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def on_episode_reset(self):
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def __call__(self, env, obs: dict, action=None) -> float:
        obj_pos = obs["obj_pos"]
        eef_pos = obs["robot0_eef_pos"]
        cab_pos = np.array(env.cab.pos)

        inside_cab = OU.obj_inside_of(env, "obj", env.cab)
        is_grasped = OU.check_obj_grasped(env, "obj")
        gripper_open = self._gripper_is_open(obs, self.open_threshold)
        gripper_far  = OU.gripper_obj_far(env, "obj")

        # Stage 4: object settled in cabinet, gripper has moved away
        if inside_cab and gripper_far:
            self._prev_reach_dist     = None
            self._prev_transport_dist = None
            return 1.0

        # Stage 3: object is inside cabinet — reward opening the gripper and retreating
        if inside_cab:
            self._prev_transport_dist = None
            dist_eef_obj = self._dist(eef_pos, obj_pos)
            retreat_bonus = self._exp_decay(dist_eef_obj, self.retreat_scale)
            if gripper_open:
                return 0.75 + 0.25 * retreat_bonus
            return 0.50 + 0.25 * retreat_bonus

        # Stage 2: object grasped — delta-based transport reward
        if is_grasped:
            self._delta_reach(self._dist(eef_pos, obj_pos), is_grasped=True)   # reset reach state
            return self._delta_transport(self._dist(obj_pos, cab_pos), is_grasped=True, inside_cab=False)

        # Stage 1: delta-based reach reward
        self._delta_transport(0.0, is_grasped=False, inside_cab=False)   # keep transport state reset
        return self._delta_reach(self._dist(eef_pos, obj_pos), is_grasped=False)


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
        reach     : delta(dist(gripper, object)) — positive when approaching,
                    negative when retreating, 0 when grasped or on first step.
        grasp     : +1 if object is grasped                               binary bonus
        transport : delta(dist(object, cabinet)) — positive when object moves toward
                    cabinet, negative when moving away, 0 when not grasped or inside.
        inside    : +1 if object is inside cabinet                         binary bonus
        release   : exp decay of dist(gripper, object) when inside cabinet active if inside
        stabilise : +1 if inside cabinet AND gripper far                   binary bonus

    Total reward is the weighted sum. Reach and transport components can be negative.

    Args:
        w_reach:      Weight for reach component.
        w_grasp:      Weight for grasp bonus.
        w_transport:  Weight for transport component.
        w_inside:     Weight for inside-cabinet bonus.
        w_release:    Weight for release component.
        w_stabilise:  Weight for stabilisation bonus.
        release_scale: Exponential scale for release retreat.
    """

    def __init__(
        self,
        w_reach:      float = 0.10,
        w_grasp:      float = 0.15,
        w_transport:  float = 0.25,
        w_inside:     float = 0.20,
        w_release:    float = 0.15,
        w_stabilise:  float = 0.15,
        release_scale: float = 2.0,
    ):
        self.w_reach     = w_reach
        self.w_grasp     = w_grasp
        self.w_transport = w_transport
        self.w_inside    = w_inside
        self.w_release   = w_release
        self.w_stabilise = w_stabilise

        self.release_scale        = release_scale
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def on_episode_reset(self):
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def __call__(self, env, obs: dict, action=None) -> float:
        obj_pos = obs["obj_pos"]
        eef_pos = obs["robot0_eef_pos"]
        cab_pos = np.array(env.cab.pos)

        inside_cab  = OU.obj_inside_of(env, "obj", env.cab)
        is_grasped  = OU.check_obj_grasped(env, "obj")
        gripper_far = OU.gripper_obj_far(env, "obj")

        # reach: delta-based, 0 when grasped
        r_reach = self.w_reach * self._delta_reach(self._dist(eef_pos, obj_pos), is_grasped)
        r_grasp = 1.0 if is_grasped else 0.0
        # transport: delta-based, 0 when not grasped or already inside cabinet
        r_transport = self.w_transport * self._delta_transport(
            self._dist(obj_pos, cab_pos), is_grasped, inside_cab
        )
        r_inside    = 1.0 if inside_cab else 0.0
        r_release   = self._exp_decay(self._dist(eef_pos, obj_pos), self.release_scale) if inside_cab else 0.0
        r_stabilise = 1.0 if (inside_cab and gripper_far) else 0.0

        return (
            r_reach
            + self.w_grasp     * r_grasp
            + r_transport
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

    def __call__(self, env, obs: dict, action=None) -> float:
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


# ---------------------------------------------------------------------------
# 5. ComposedPickPlaceReward
# ---------------------------------------------------------------------------

class ComposedPickPlaceReward(RewardFn):
    """
    Additive reward composed of three independent components.

    Components
    ──────────
        r_reach     = delta(dist(gripper, object))
                      Active only while not grasped. Positive when gripper approaches
                      the object, negative when retreating, 0 on the first step of
                      each reach phase and while the object is grasped.

        r_lift      = w_lift  if dist(gripper, object) < near_threshold AND grasped
                      Proximity-gated grasp bonus: the gripper must already be close
                      before the bonus is available, preventing the policy from
                      earning it by randomly closing fingers far from the object.

        r_transport = exp(-transport_scale * dist(object, cabinet))   ∈ (0, 1]
                      Active only while the object is grasped. Guides the held
                      object toward the cabinet opening.

        r_success   = success_bonus  if object is inside cabinet (one-time)
                      Large terminal bonus for task completion.

    Total reward each step:
        r = w_reach * r_reach  +  r_lift  +  w_transport * r_transport  +  r_success

    Note: r_reach can be negative, so the total reward is not strictly non-negative
    during the reach phase.

    Args:
        w_reach:         Weight on the reach component (default 1.0).
        w_transport:     Weight on the transport component (default 1.0).
        w_lift:          Flat bonus for grasping while close (default 1.0).
        transport_scale: Exponential decay rate for transport (default 3.0).
        near_threshold:  Gripper must be within this distance (m) to earn r_lift
                         (default 0.10).
        success_bonus:   One-time reward when object enters cabinet (default 10.0).
    """

    def __init__(
        self,
        w_reach:        float = 1.0,
        w_transport:    float = 1.0,
        w_lift:         float = 1.0,
        near_threshold: float = 0.10,
        success_bonus:  float = 10.0,
    ):
        self.w_reach        = w_reach
        self.w_transport    = w_transport
        self.w_lift         = w_lift
        self.near_threshold = near_threshold
        self.success_bonus  = success_bonus
        self._gave_success        = False
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def on_episode_reset(self):
        """Reset per-episode state at episode start."""
        self._gave_success        = False
        self._prev_reach_dist     = None
        self._prev_transport_dist = None

    def __call__(self, env, obs: dict, action=None) -> float:
        eef_pos = obs["robot0_eef_pos"]
        obj_pos = obs["obj_pos"]
        cab_pos = np.array(env.cab.pos)

        dist_reach = self._dist(eef_pos, obj_pos)
        is_grasped = OU.check_obj_grasped(env, "obj")
        inside_cab = OU.obj_inside_of(env, "obj", env.cab)

        # r_reach: delta-based — 0 when grasped, positive/negative otherwise
        r_reach = self.w_reach * self._delta_reach(dist_reach, is_grasped)

        # r_lift: proximity-gated grasp bonus
        r_lift = 0.0
        if dist_reach < self.near_threshold and is_grasped:
            r_lift = self.w_lift

        # r_transport: delta-based — positive when object moves toward cabinet,
        # negative when moving away, 0 when not grasped or already inside cabinet
        r_transport = self.w_transport * self._delta_transport(
            self._dist(obj_pos, cab_pos), is_grasped, inside_cab
        )

        # r_success: one-time terminal bonus
        r_success = 0.0
        if inside_cab and not self._gave_success:
            r_success = self.success_bonus
            self._gave_success = True

        return r_reach + r_lift + r_transport + r_success
