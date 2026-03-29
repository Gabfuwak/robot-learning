"""
Save a few demonstrations from the .demo_cache for inspection.

For each selected episode this script writes:
  inspect_demos/
    ep_XXXX/
      states.npy      (T, 110) float32  — full env observation per timestep
      actions.npy     (T, 12)  float32  — expert action per timestep
      trajectory.csv           — human-readable key signals over time

  all_episodes.csv   — per-episode summary (length, final success proxy)

State vector layout (110 dims, same as RoboCasaWrapper / train_IL.py)
  [ 0: 7]  robot0_joint_pos
  [ 7:14]  robot0_joint_pos_cos
  [14:21]  robot0_joint_pos_sin
  [21:28]  robot0_joint_vel
  [28:35]  robot0_joint_acc
  [35:38]  robot0_eef_pos          (world frame)
  [38:42]  robot0_eef_quat
  [42:46]  robot0_eef_quat_site
  [46:48]  robot0_gripper_qpos
  [48:50]  robot0_gripper_qvel
  [50:53]  robot0_base_pos
  [53:57]  robot0_base_quat
  [57:60]  robot0_base_to_eef_pos
  [60:64]  robot0_base_to_eef_quat
  [64:68]  robot0_base_to_eef_quat_site
  [68:71]  obj_pos
  [71:75]  obj_quat
  [75:78]  obj_to_eef_pos
  [78:82]  obj_to_eef_quat
  [82:96]  distr_counter (same 14-dim layout)
  [96:110] distr_cab     (same 14-dim layout)

Action vector layout (12 dims)
  [0:3]   EEF position delta  (x, y, z)
  [3:6]   EEF orientation delta (rotation vector)
  [6]     Gripper command  (-1=open, +1=close)
  [7:10]  Mobile base velocity (vx, vy, w)
  [10:12] Padding

Usage
─────
    python inspect_demos.py                  # save first 5 episodes
    python inspect_demos.py --n 10           # save first 10 episodes
    python inspect_demos.py --episodes 0 3 7 # save specific episode indices
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deps", "robocasa"))
sys.path.insert(0, os.path.join(ROOT, "deps", "robosuite"))

CACHE_STATES  = ".demo_cache/PickPlaceCounterToCabinet_target_human_None_states.npy"
CACHE_ACTIONS = ".demo_cache/PickPlaceCounterToCabinet_target_human_None_actions.npy"

# ── State field slices ────────────────────────────────────────────────────────
STATE_FIELDS = {
    "joint_pos":              slice(0,  7),
    "eef_pos":                slice(35, 38),
    "eef_quat":               slice(38, 42),
    "gripper_qpos":           slice(46, 48),
    "base_pos":               slice(50, 53),
    "base_to_eef_pos":        slice(57, 60),
    "obj_pos":                slice(68, 71),
    "obj_quat":               slice(71, 75),
    "obj_to_eef_pos":         slice(75, 78),
}

# ── Action field slices ───────────────────────────────────────────────────────
ACTION_FIELDS = {
    "eef_delta_pos":    slice(0, 3),
    "eef_delta_rot":    slice(3, 6),
    "gripper_cmd":      slice(6, 7),
    "base_vel":         slice(7, 10),
}


def _flat_row(step: int, state: np.ndarray, action: np.ndarray) -> dict:
    """Build one CSV row from a single (state, action) pair."""
    row: dict = {"step": step}
    for name, sl in STATE_FIELDS.items():
        vals = state[sl]
        if len(vals) == 1:
            row[name] = float(vals[0])
        else:
            for i, v in enumerate(vals):
                row[f"{name}_{i}"] = float(v)
    for name, sl in ACTION_FIELDS.items():
        vals = action[sl]
        if len(vals) == 1:
            row[name] = float(vals[0])
        else:
            for i, v in enumerate(vals):
                row[f"{name}_{i}"] = float(v)
    return row


def build_episode_index() -> list[tuple[int, int]]:
    """
    Return [(start, end), ...] index into the flat cache for every episode.
    Episode lengths are derived from the dataset's parquet action files.
    """
    from robocasa.utils.dataset_registry_utils import get_ds_path
    import robocasa.utils.lerobot_utils as LU

    dataset_dir = Path(get_ds_path("PickPlaceCounterToCabinet", source="human", split="target"))
    episodes    = LU.get_episodes(dataset_dir)

    index: list[tuple[int, int]] = []
    cursor = 0
    for ind in range(len(episodes)):
        actions = LU.get_episode_actions(dataset_dir, ind)
        T       = len(actions)
        index.append((cursor, cursor + T))
        cursor += T
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5,
                        help="Number of episodes to save (ignored if --episodes is set)")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Specific episode indices to save, e.g. --episodes 0 3 7")
    parser.add_argument("--out_dir", default="inspect_demos",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load flat cache ───────────────────────────────────────────────────────
    print("Loading cache ...")
    all_states  = np.load(CACHE_STATES)   # (N_total, 110)
    all_actions = np.load(CACHE_ACTIONS)  # (N_total, 12)
    print(f"  Cache: {all_states.shape[0]:,} total transitions")

    # ── Build episode boundaries ──────────────────────────────────────────────
    print("Building episode index from dataset ...")
    ep_index = build_episode_index()
    print(f"  {len(ep_index)} episodes found")

    selected = args.episodes if args.episodes is not None else list(range(args.n))
    selected = [i for i in selected if i < len(ep_index)]

    summary_rows: list[dict] = []

    for ep_idx in selected:
        start, end = ep_index[ep_idx]
        ep_states  = all_states[start:end]    # (T, 110)
        ep_actions = all_actions[start:end]   # (T, 12)
        T          = len(ep_states)

        ep_out = out_dir / f"ep_{ep_idx:04d}"
        ep_out.mkdir(exist_ok=True)

        # Raw numpy files
        np.save(ep_out / "states.npy",  ep_states)
        np.save(ep_out / "actions.npy", ep_actions)

        # Human-readable CSV
        rows = [_flat_row(t, ep_states[t], ep_actions[t]) for t in range(T)]
        df   = pd.DataFrame(rows)
        df.to_csv(ep_out / "trajectory.csv", index=False, float_format="%.5f")

        # Quick stats for summary
        obj_dist_to_eef  = np.linalg.norm(ep_states[:, 75:78], axis=1)  # obj_to_eef_pos
        gripper          = ep_states[:, 46]                               # finger 0
        min_obj_dist     = float(obj_dist_to_eef.min())
        ever_grasped     = bool((obj_dist_to_eef < 0.05).any())
        final_obj_pos    = ep_states[-1, 68:71].tolist()

        summary_rows.append({
            "episode":        ep_idx,
            "length":         T,
            "min_obj_eef_dist": round(min_obj_dist, 4),
            "ever_grasped":   ever_grasped,
            "final_obj_x":    round(final_obj_pos[0], 4),
            "final_obj_y":    round(final_obj_pos[1], 4),
            "final_obj_z":    round(final_obj_pos[2], 4),
        })

        print(
            f"  ep_{ep_idx:04d}: T={T:>4}  "
            f"min_obj_eef_dist={min_obj_dist:.3f}  "
            f"ever_grasped={ever_grasped}"
        )

    # ── Overall summary CSV ───────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "episodes_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.5f")

    print(f"\nSaved {len(selected)} episodes to {out_dir}/")
    print(f"  ep_XXXX/states.npy    — (T, 110) full state per timestep")
    print(f"  ep_XXXX/actions.npy   — (T, 12)  expert action per timestep")
    print(f"  ep_XXXX/trajectory.csv — key signals in human-readable form")
    print(f"  episodes_summary.csv  — per-episode stats")


if __name__ == "__main__":
    main()
