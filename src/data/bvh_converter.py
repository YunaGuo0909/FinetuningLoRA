"""Convert BVH motion capture files (e.g. 100STYLE) to HumanML3D-compatible format.

Pipeline:
    BVH -> joint positions (forward kinematics) -> retarget to 22-joint skeleton
    -> compute 263-dim features (positions, velocities, rotations, foot contacts)

Usage:
    converter = BVHToHumanML3D()
    features = converter.convert("path/to/animation.bvh")  # returns (T, 263) numpy array
"""

from __future__ import annotations

import re
import json
import numpy as np
from pathlib import Path


# HumanML3D 22-joint skeleton
HUMANML3D_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

# Kinematic chain (parent indices)
KINEMATIC_CHAIN = [
    [0, 1, 4, 7, 10],   # left leg
    [0, 2, 5, 8, 11],   # right leg
    [0, 3, 6, 9, 12, 15],  # spine -> head
    [9, 13, 16, 18, 20],   # left arm
    [9, 14, 17, 19, 21],   # right arm
]

# Foot joint indices for contact detection
L_FOOT_IDX, R_FOOT_IDX = 10, 11
L_ANKLE_IDX, R_ANKLE_IDX = 7, 8


# ---------------------------------------------------------------------------
# BVH Parser
# ---------------------------------------------------------------------------

class BVHParser:
    """Minimal BVH parser that extracts joint hierarchy and motion data."""

    def parse(self, filepath: str) -> dict:
        with open(filepath, "r") as f:
            content = f.read()

        hierarchy_str, motion_str = content.split("MOTION")

        joints = self._parse_hierarchy(hierarchy_str)
        frames, frame_time = self._parse_motion(motion_str, joints)

        return {
            "joints": joints,
            "frames": frames,       # (T, total_channels)
            "frame_time": frame_time,
            "num_frames": frames.shape[0],
        }

    def _parse_hierarchy(self, text: str) -> list[dict]:
        joints = []
        stack = []
        lines = text.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("ROOT") or line.startswith("JOINT"):
                name = line.split()[-1]
                joint = {"name": name, "channels": [], "offset": [0, 0, 0], "parent": stack[-1] if stack else -1}
                idx = len(joints)
                joints.append(joint)
                stack.append(idx)

            elif line.startswith("End Site"):
                # Skip end site
                i += 1
                while i < len(lines) and "}" not in lines[i]:
                    i += 1
                i += 1
                continue

            elif line.startswith("OFFSET"):
                vals = [float(x) for x in line.split()[1:4]]
                if stack:
                    joints[stack[-1]]["offset"] = vals

            elif line.startswith("CHANNELS"):
                parts = line.split()
                n_channels = int(parts[1])
                channel_names = parts[2:2 + n_channels]
                if stack:
                    joints[stack[-1]]["channels"] = channel_names

            elif line.strip() == "}":
                if stack:
                    stack.pop()

            i += 1
        return joints

    def _parse_motion(self, text: str, joints: list) -> tuple[np.ndarray, float]:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        num_frames = int(lines[0].split(":")[1])
        frame_time = float(lines[1].split(":")[1])

        frames = []
        for line in lines[2:2 + num_frames]:
            vals = [float(x) for x in line.split()]
            frames.append(vals)

        return np.array(frames, dtype=np.float64), frame_time


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

def euler_to_rotation_matrix(angles_deg: np.ndarray, order: str = "ZXY") -> np.ndarray:
    """Convert Euler angles (degrees) to 3x3 rotation matrix."""
    angles = np.radians(angles_deg)
    matrices = {}
    cx, sx = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cz, sz = np.cos(angles[2]), np.sin(angles[2])

    matrices["X"] = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    matrices["Y"] = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    matrices["Z"] = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    R = np.eye(3)
    for axis in order:
        R = R @ matrices[axis]
    return R


def forward_kinematics(joints: list, frame_data: np.ndarray):
    """Compute global joint positions and local rotations from BVH frame data.

    Returns:
        positions: (n_joints, 3) global positions
        local_rotations: list of (3, 3) local rotation matrices per joint
    """
    positions = np.zeros((len(joints), 3))
    global_rotations = [np.eye(3)] * len(joints)
    local_rotations = [np.eye(3)] * len(joints)

    ch_idx = 0
    for j, joint in enumerate(joints):
        offset = np.array(joint["offset"])
        channels = joint["channels"]
        parent = joint["parent"]

        parent_rot = global_rotations[parent] if parent >= 0 else np.eye(3)
        parent_pos = positions[parent] if parent >= 0 else np.zeros(3)

        # Parse channels
        local_pos = np.zeros(3)
        euler = np.zeros(3)
        rot_order = ""

        for ch_name in channels:
            val = frame_data[ch_idx]
            ch_idx += 1
            if "Xposition" in ch_name:
                local_pos[0] = val
            elif "Yposition" in ch_name:
                local_pos[1] = val
            elif "Zposition" in ch_name:
                local_pos[2] = val
            elif "Xrotation" in ch_name:
                euler[0] = val
                rot_order += "X"
            elif "Yrotation" in ch_name:
                euler[1] = val
                rot_order += "Y"
            elif "Zrotation" in ch_name:
                euler[2] = val
                rot_order += "Z"

        local_rot = euler_to_rotation_matrix(euler, rot_order if rot_order else "ZXY")
        global_rot = parent_rot @ local_rot

        if parent >= 0:
            positions[j] = parent_pos + parent_rot @ (offset + local_pos)
        else:
            positions[j] = offset + local_pos

        global_rotations[j] = global_rot
        local_rotations[j] = local_rot

    return positions, local_rotations


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (first two columns).

    Standard 6D: [col0, col1] = [R00, R10, R20, R01, R11, R21]
    """
    return np.concatenate([R[:, 0], R[:, 1]])  # (6,)


def compute_foot_contacts(positions_seq: np.ndarray,
                          height_threshold: float = 0.05,
                          velocity_threshold: float = 0.01) -> np.ndarray:
    """Detect foot contacts from joint height AND velocity.

    Contact = (foot height < threshold) AND (foot XZ velocity < threshold).
    This avoids false positives when the foot passes through low height at speed.

    Args:
        positions_seq: (T, n_joints, 3) global joint positions
        height_threshold: max height (Y) for contact
        velocity_threshold: max XZ speed for contact
    Returns:
        (T, 4) binary contacts [l_ankle, l_foot, r_ankle, r_foot]
    """
    T = positions_seq.shape[0]
    contacts = np.zeros((T, 4))

    foot_joints = [L_ANKLE_IDX, L_FOOT_IDX, R_ANKLE_IDX, R_FOOT_IDX]

    for t in range(T):
        for i, j in enumerate(foot_joints):
            height_ok = positions_seq[t, j, 1] < height_threshold
            if t > 0:
                dx = positions_seq[t, j, 0] - positions_seq[t - 1, j, 0]
                dz = positions_seq[t, j, 2] - positions_seq[t - 1, j, 2]
                vel = np.sqrt(dx * dx + dz * dz)
                vel_ok = vel < velocity_threshold
            else:
                vel_ok = True  # first frame: assume contact if low
            contacts[t, i] = float(height_ok and vel_ok)

    return contacts


def compute_humanml3d_features(positions_seq: np.ndarray,
                                rotations_seq: list = None) -> np.ndarray:
    """Convert joint positions sequence to HumanML3D 263-dim features.

    Args:
        positions_seq: (T, 22, 3) joint positions
        rotations_seq: list of T elements, each a list of 22 (3,3) local rotation matrices.
                       If None, identity rotations are used.

    Returns:
        (T, 263) feature array:
            [0]     root_rot_velocity (1)
            [1:3]   root_linear_velocity (2) - XZ plane
            [3]     root_y (1)
            [4:67]  ric_data - root-relative joint positions (21*3=63)
            [67:193] rot_data - 6D local joint rotations (21*6=126)
            [193:259] local_velocity (22*3=66)
            [259:263] foot_contact (4)
    """
    T, n_joints = positions_seq.shape[:2]
    assert n_joints == 22, f"Expected 22 joints, got {n_joints}"

    features = np.zeros((T, 263))

    for t in range(T):
        root_pos = positions_seq[t, 0]

        # [0] Root rotation velocity
        if t > 0:
            prev_fwd = positions_seq[t - 1, 2] - positions_seq[t - 1, 1]
            curr_fwd = positions_seq[t, 2] - positions_seq[t, 1]
            prev_angle = np.arctan2(prev_fwd[0], prev_fwd[2])
            curr_angle = np.arctan2(curr_fwd[0], curr_fwd[2])
            diff = curr_angle - prev_angle
            # Wrap to [-pi, pi]
            features[t, 0] = np.arctan2(np.sin(diff), np.cos(diff))

        # [1:3] Root linear velocity (XZ)
        if t > 0:
            features[t, 1] = positions_seq[t, 0, 0] - positions_seq[t - 1, 0, 0]
            features[t, 2] = positions_seq[t, 0, 2] - positions_seq[t - 1, 0, 2]

        # [3] Root height
        features[t, 3] = root_pos[1]

        # [4:67] Root-relative joint positions (21 joints)
        idx = 4
        for j in range(1, 22):
            features[t, idx:idx + 3] = positions_seq[t, j] - root_pos
            idx += 3

        # [67:193] 6D joint rotations (21 joints, exclude root)
        idx = 67
        if rotations_seq is not None:
            for j in range(1, 22):
                R = rotations_seq[t][j]
                features[t, idx:idx + 6] = rotation_matrix_to_6d(R)
                idx += 6
        else:
            for j in range(1, 22):
                features[t, idx:idx + 6] = [1, 0, 0, 0, 1, 0]  # identity 6D
                idx += 6

        # [193:259] Joint velocities (all 22 joints)
        idx = 193
        if t > 0:
            for j in range(22):
                features[t, idx:idx + 3] = positions_seq[t, j] - positions_seq[t - 1, j]
                idx += 3
        else:
            idx += 66

    # [259:263] Foot contacts
    features[:, 259:263] = compute_foot_contacts(positions_seq)

    return features


# ---------------------------------------------------------------------------
# BVH -> HumanML3D Converter
# ---------------------------------------------------------------------------

# Common BVH-to-HumanML3D joint name mapping
# Mapping: BVH joint name -> HumanML3D joint name
# Multiple common BVH naming conventions are supported.
DEFAULT_JOINT_MAP = {
    # --- Mixamo / CMU style ---
    "Hips": "Pelvis",
    "LeftUpLeg": "L_Hip", "RightUpLeg": "R_Hip",
    "Spine": "Spine1",
    "LeftLeg": "L_Knee", "RightLeg": "R_Knee",
    "Spine1": "Spine2",
    "LeftFoot": "L_Ankle", "RightFoot": "R_Ankle",
    "Spine2": "Spine3",
    "LeftToeBase": "L_Foot", "RightToeBase": "R_Foot",
    "Neck": "Neck",
    "LeftShoulder": "L_Collar", "RightShoulder": "R_Collar",
    "Head": "Head",
    "LeftArm": "L_Shoulder", "RightArm": "R_Shoulder",
    "LeftForeArm": "L_Elbow", "RightForeArm": "R_Elbow",
    "LeftHand": "L_Wrist", "RightHand": "R_Wrist",
    # --- 100STYLE skeleton ---
    "Chest": "Spine1",
    "Chest2": "Spine2",
    "Chest3": "Spine3",
    "Chest4": "Spine3",       # 100STYLE has 4 spine joints, we merge 3&4
    "LeftHip": "L_Hip", "RightHip": "R_Hip",
    "LeftKnee": "L_Knee", "RightKnee": "R_Knee",
    "LeftAnkle": "L_Ankle", "RightAnkle": "R_Ankle",
    "LeftToe": "L_Foot", "RightToe": "R_Foot",
    "LeftCollar": "L_Collar", "RightCollar": "R_Collar",
    "LeftShoulder": "L_Shoulder", "RightShoulder": "R_Shoulder",
    "LeftElbow": "L_Elbow", "RightElbow": "R_Elbow",
    "LeftWrist": "L_Wrist", "RightWrist": "R_Wrist",
}


class BVHToHumanML3D:
    """Convert BVH files to HumanML3D 263-dim features."""

    def __init__(self, joint_map: dict | None = None, target_fps: int = 20):
        self.parser = BVHParser()
        self.joint_map = joint_map or DEFAULT_JOINT_MAP
        self.target_fps = target_fps

    def convert(self, bvh_path: str) -> np.ndarray | None:
        """Convert a single BVH file to (T, 263) features.

        Returns None if conversion fails (e.g. missing joints).
        """
        try:
            data = self.parser.parse(bvh_path)
        except Exception as e:
            print(f"Failed to parse {bvh_path}: {e}")
            return None

        joints = data["joints"]
        frames = data["frames"]
        src_fps = round(1.0 / data["frame_time"])

        # Build name-to-index map for BVH joints
        bvh_name_to_idx = {j["name"]: i for i, j in enumerate(joints)}

        # Map BVH joints to HumanML3D order
        mapping = []
        for target_name in HUMANML3D_JOINTS:
            found = False
            for bvh_name, hml_name in self.joint_map.items():
                if hml_name == target_name and bvh_name in bvh_name_to_idx:
                    mapping.append(bvh_name_to_idx[bvh_name])
                    found = True
                    break
            if not found:
                print(f"Warning: joint '{target_name}' not found in BVH, using root")
                mapping.append(0)

        # Forward kinematics for all frames (positions + local rotations)
        T = data["num_frames"]
        all_positions = np.zeros((T, len(joints), 3))
        all_local_rots = []  # T x n_joints list of (3,3)
        for t in range(T):
            pos, local_rots = forward_kinematics(joints, frames[t])
            all_positions[t] = pos
            all_local_rots.append(local_rots)

        # Retarget to 22 joints (positions and rotations)
        positions_22 = all_positions[:, mapping]  # (T, 22, 3)
        rotations_22 = []
        for t in range(T):
            rots = [all_local_rots[t][mapping[j]] for j in range(22)]
            rotations_22.append(rots)

        # Convert units: BVH is often in cm, HumanML3D uses meters
        # Heuristic: if root height > 50, assume cm
        avg_height = positions_22[:, 0, 1].mean()
        if abs(avg_height) > 50:
            positions_22 /= 100.0

        # Resample to target FPS
        if src_fps != self.target_fps and src_fps > 0:
            ratio = self.target_fps / src_fps
            new_T = int(T * ratio)
            old_t = np.linspace(0, T - 1, T)
            new_t = np.linspace(0, T - 1, new_T)

            # Resample positions
            resampled_pos = np.zeros((new_T, 22, 3))
            for j in range(22):
                for axis in range(3):
                    resampled_pos[:, j, axis] = np.interp(new_t, old_t, positions_22[:, j, axis])
            positions_22 = resampled_pos

            # Resample rotations (nearest-neighbor for rotation matrices)
            indices = np.round(new_t).astype(int).clip(0, T - 1)
            rotations_22 = [rotations_22[i] for i in indices]

        # Compute features with real rotations
        features = compute_humanml3d_features(positions_22, rotations_22)
        return features

    def convert_directory(self, bvh_dir: str, output_dir: str, style_label: str = "") -> list[dict]:
        """Convert all BVH files in a directory. Returns metadata entries."""
        bvh_dir = Path(bvh_dir)
        out_dir = Path(output_dir) / "motions"
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        bvh_files = sorted(bvh_dir.glob("*.bvh"))
        print(f"Converting {len(bvh_files)} BVH files from {bvh_dir}...")

        for bvh_file in bvh_files:
            features = self.convert(str(bvh_file))
            if features is None:
                continue

            # Extract action/style from filename (e.g. "Walk_Zombie_001.bvh")
            stem = bvh_file.stem
            parts = stem.split("_")
            action = parts[0].lower() if parts else "unknown"
            style = style_label or (parts[1].lower() if len(parts) > 1 else "neutral")

            out_file = f"{stem}.npy"
            np.save(out_dir / out_file, features)

            caption = f"a person {action}ing in {style} style"
            metadata.append({
                "file": out_file,
                "action": action,
                "style": style,
                "caption": caption,
                "length": features.shape[0],
            })

        # Save metadata
        metadata_path = Path(output_dir) / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Converted {len(metadata)} motions, saved to {output_dir}")
        return metadata
