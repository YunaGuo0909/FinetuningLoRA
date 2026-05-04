"""Convert BVH motion capture files (e.g. 100STYLE) to HumanML3D-compatible format.

Pipeline:
    BVH -> joint positions (forward kinematics) -> retarget to 22-joint skeleton
    -> compute 263-dim features (positions, velocities, rotations, foot contacts)

Usage:
    converter = BVHToHumanML3D()
    features = converter.convert("path/to/animation.bvh")  # returns (T, 263) numpy array
"""

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


def forward_kinematics(joints: list, frame_data: np.ndarray) -> np.ndarray:
    """Compute global joint positions from BVH frame data.

    Returns: (n_joints, 3) global positions
    """
    positions = np.zeros((len(joints), 3))
    rotations = [np.eye(3)] * len(joints)

    ch_idx = 0
    for j, joint in enumerate(joints):
        offset = np.array(joint["offset"])
        channels = joint["channels"]
        parent = joint["parent"]

        parent_rot = rotations[parent] if parent >= 0 else np.eye(3)
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

        rotations[j] = global_rot

    return positions


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return R[:, :2].T.flatten()  # (6,)


def compute_foot_contacts(positions_seq: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Detect foot contacts from joint height and velocity.

    Args:
        positions_seq: (T, n_joints, 3) global joint positions
        threshold: height threshold for contact
    Returns:
        (T, 4) binary contacts [l_ankle, l_foot, r_ankle, r_foot]
    """
    T = positions_seq.shape[0]
    contacts = np.zeros((T, 4))

    for t in range(T):
        contacts[t, 0] = positions_seq[t, L_ANKLE_IDX, 1] < threshold  # y-axis = up
        contacts[t, 1] = positions_seq[t, L_FOOT_IDX, 1] < threshold
        contacts[t, 2] = positions_seq[t, R_ANKLE_IDX, 1] < threshold
        contacts[t, 3] = positions_seq[t, R_FOOT_IDX, 1] < threshold

    return contacts


def compute_humanml3d_features(positions_seq: np.ndarray) -> np.ndarray:
    """Convert joint positions sequence to HumanML3D 263-dim features.

    Args:
        positions_seq: (T, 22, 3) joint positions

    Returns:
        (T, 263) feature array:
            [0]     root_rot_velocity (1)
            [1:3]   root_linear_velocity (2) - XZ plane
            [3]     root_y (1)
            [4:67]  ric_data - root-relative joint positions (21*3=63)
            [67:193] rot_data - placeholder 6D rotations (21*6=126)
            [193:259] local_velocity (22*3=66)
            [259:263] foot_contact (4)
    """
    T, n_joints = positions_seq.shape[:2]
    assert n_joints == 22, f"Expected 22 joints, got {n_joints}"

    features = np.zeros((T, 263))

    for t in range(T):
        root_pos = positions_seq[t, 0]
        idx = 0

        # Root rotation velocity (approximate from facing direction changes)
        if t > 0:
            prev_fwd = positions_seq[t - 1, 2] - positions_seq[t - 1, 1]  # R_hip - L_hip cross
            curr_fwd = positions_seq[t, 2] - positions_seq[t, 1]
            prev_angle = np.arctan2(prev_fwd[0], prev_fwd[2])
            curr_angle = np.arctan2(curr_fwd[0], curr_fwd[2])
            features[t, 0] = curr_angle - prev_angle
        idx = 1

        # Root linear velocity (XZ)
        if t > 0:
            features[t, 1] = positions_seq[t, 0, 0] - positions_seq[t - 1, 0, 0]
            features[t, 2] = positions_seq[t, 0, 2] - positions_seq[t - 1, 0, 2]
        idx = 3

        # Root height
        features[t, 3] = root_pos[1]
        idx = 4

        # Root-relative joint positions (exclude root itself)
        for j in range(1, 22):
            rel = positions_seq[t, j] - root_pos
            features[t, idx:idx + 3] = rel
            idx += 3
        # idx = 67

        # 6D rotation placeholders (identity-based when we only have positions)
        for j in range(1, 22):
            features[t, idx:idx + 6] = [1, 0, 0, 1, 0, 0]  # identity 6D
            idx += 6
        # idx = 193

        # Local velocities
        if t > 0:
            for j in range(22):
                vel = positions_seq[t, j] - positions_seq[t - 1, j]
                features[t, idx:idx + 3] = vel
                idx += 3
        else:
            idx += 66
        # idx = 259

    # Foot contacts
    features[:, 259:263] = compute_foot_contacts(positions_seq)

    return features


# ---------------------------------------------------------------------------
# BVH -> HumanML3D Converter
# ---------------------------------------------------------------------------

# Common BVH-to-HumanML3D joint name mapping
DEFAULT_JOINT_MAP = {
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

        # Forward kinematics for all frames
        T = data["num_frames"]
        all_positions = np.zeros((T, len(joints), 3))
        for t in range(T):
            all_positions[t] = forward_kinematics(joints, frames[t])

        # Retarget to 22 joints
        positions_22 = all_positions[:, mapping]  # (T, 22, 3)

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
            resampled = np.zeros((new_T, 22, 3))
            for j in range(22):
                for axis in range(3):
                    resampled[:, j, axis] = np.interp(new_t, old_t, positions_22[:, j, axis])
            positions_22 = resampled

        # Compute features
        features = compute_humanml3d_features(positions_22)
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
