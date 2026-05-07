"""Visualize generated motions as skeleton animations.

Outputs:
    - Matplotlib skeleton animation (saved as MP4/GIF)
    - Per-frame skeleton plots (saved as PNG sequence)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# HumanML3D kinematic chains for drawing bones
KINEMATIC_CHAINS = [
    [0, 1, 4, 7, 10],      # left leg
    [0, 2, 5, 8, 11],      # right leg
    [0, 3, 6, 9, 12, 15],  # spine -> head
    [9, 13, 16, 18, 20],   # left arm
    [9, 14, 17, 19, 21],   # right arm
]

CHAIN_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def _smooth_root_trajectory(positions: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth root XZ trajectory to reduce accumulated drift.

    Uses a simple moving average on root XZ velocity, then re-integrates.
    This suppresses the small per-frame errors that cause unwanted turning/drift.
    """
    T = positions.shape[0]
    if T < window * 2:
        return positions

    # Extract root XZ velocities
    root_vx = np.diff(positions[:, 0, 0])  # (T-1,)
    root_vz = np.diff(positions[:, 0, 2])  # (T-1,)

    # Smooth velocities with moving average
    kernel = np.ones(window) / window
    root_vx_smooth = np.convolve(root_vx, kernel, mode='same')
    root_vz_smooth = np.convolve(root_vz, kernel, mode='same')

    # Re-integrate smoothed velocities
    new_root_x = np.zeros(T)
    new_root_z = np.zeros(T)
    new_root_x[0] = positions[0, 0, 0]
    new_root_z[0] = positions[0, 0, 2]
    for t in range(1, T):
        new_root_x[t] = new_root_x[t - 1] + root_vx_smooth[t - 1]
        new_root_z[t] = new_root_z[t - 1] + root_vz_smooth[t - 1]

    # Apply offset to all joints (shift entire skeleton)
    dx = new_root_x - positions[:, 0, 0]
    dz = new_root_z - positions[:, 0, 2]
    positions[:, :, 0] += dx[:, None]
    positions[:, :, 2] += dz[:, None]

    return positions


def _enforce_bone_lengths(positions: np.ndarray) -> np.ndarray:
    """Enforce consistent bone lengths across all frames.

    Computes median bone length from all frames, then rescales each bone
    to match. This prevents the skeleton from shrinking/growing.
    """
    T = positions.shape[0]
    bone_pairs = []
    for chain in KINEMATIC_CHAINS:
        for i in range(len(chain) - 1):
            bone_pairs.append((chain[i], chain[i + 1]))

    # Compute median bone lengths
    all_lengths = np.zeros((T, len(bone_pairs)))
    for t in range(T):
        for b, (j1, j2) in enumerate(bone_pairs):
            all_lengths[t, b] = np.linalg.norm(positions[t, j2] - positions[t, j1])

    median_lengths = np.median(all_lengths, axis=0)

    # Rescale bones from root outward
    for t in range(T):
        for chain in KINEMATIC_CHAINS:
            for i in range(len(chain) - 1):
                parent, child = chain[i], chain[i + 1]
                b_idx = bone_pairs.index((parent, child))
                target_len = median_lengths[b_idx]

                direction = positions[t, child] - positions[t, parent]
                current_len = np.linalg.norm(direction)
                if current_len > 1e-6:
                    positions[t, child] = (
                        positions[t, parent] + direction * (target_len / current_len)
                    )

    return positions


def motion_features_to_positions(motion: np.ndarray) -> np.ndarray:
    """Extract joint positions from HumanML3D 263-dim features.

    Args:
        motion: (T, 263) motion features
    Returns:
        (T, 22, 3) joint positions (root-relative + root)
    """
    T = motion.shape[0]
    positions = np.zeros((T, 22, 3))

    for t in range(T):
        # Root position (integrate velocity)
        if t > 0:
            positions[t, 0, 0] = positions[t - 1, 0, 0] + motion[t, 1]  # x velocity
            positions[t, 0, 2] = positions[t - 1, 0, 2] + motion[t, 2]  # z velocity
        positions[t, 0, 1] = motion[t, 3]  # root height

        # Other joints (root-relative, dims 4:67)
        for j in range(1, 22):
            offset = 4 + (j - 1) * 3
            positions[t, j] = positions[t, 0] + motion[t, offset:offset + 3]

    # --- Post-processing: root trajectory smoothing ---
    positions = _smooth_root_trajectory(positions, window=5)

    # --- Post-processing: improved foot sliding reduction ---
    # Uses both foot contact signal AND velocity/height heuristics
    if motion.shape[1] >= 263:
        foot_contact = motion[:, 259:263]  # (T, 4)
        l_contact = (foot_contact[:, 0] + foot_contact[:, 1]) / 2
        r_contact = (foot_contact[:, 2] + foot_contact[:, 3]) / 2

        # Also detect contact from height + velocity (backup when contact signal is noisy)
        # Foot joints: 7=L_Ankle, 8=R_Ankle, 10=L_Foot, 11=R_Foot
        height_thresh = 0.15
        vel_thresh = 0.02
        for t in range(1, T):
            # Left foot: combine contact signal with height/velocity check
            l_height_ok = positions[t, 10, 1] < height_thresh
            l_vel = np.linalg.norm(positions[t, 10] - positions[t - 1, 10])
            l_vel_ok = l_vel < vel_thresh
            l_grounded = l_contact[t] > 0.5 or (l_height_ok and l_vel_ok)

            # Right foot
            r_height_ok = positions[t, 11, 1] < height_thresh
            r_vel = np.linalg.norm(positions[t, 11] - positions[t - 1, 11])
            r_vel_ok = r_vel < vel_thresh
            r_grounded = r_contact[t] > 0.5 or (r_height_ok and r_vel_ok)

            if l_grounded:
                # Blend toward pinned position (soft pin) instead of hard snap
                alpha = 0.8
                for j in [10, 7]:  # foot and ankle
                    positions[t, j, 0] = alpha * positions[t - 1, j, 0] + (1 - alpha) * positions[t, j, 0]
                    positions[t, j, 2] = alpha * positions[t - 1, j, 2] + (1 - alpha) * positions[t, j, 2]

            if r_grounded:
                alpha = 0.8
                for j in [11, 8]:
                    positions[t, j, 0] = alpha * positions[t - 1, j, 0] + (1 - alpha) * positions[t, j, 0]
                    positions[t, j, 2] = alpha * positions[t - 1, j, 2] + (1 - alpha) * positions[t, j, 2]

    # --- Post-processing: enforce bone lengths ---
    positions = _enforce_bone_lengths(positions)

    # Fix root visual position: place pelvis at midpoint of hips
    for t in range(T):
        positions[t, 0] = (positions[t, 1] + positions[t, 2]) / 2

    return positions


def plot_skeleton_frame(
    ax, positions: np.ndarray, title: str = "", alpha: float = 1.0,
    xlim=None, ylim=None, zlim=None,
):
    """Plot a single skeleton frame on a 3D axis.

    Args:
        ax: matplotlib 3D axis
        positions: (22, 3) joint positions
        xlim, ylim, zlim: axis limits (auto-computed if None)
    """
    ax.cla()

    # Draw bones (swap Y/Z: data Y=height -> plot Z=vertical)
    for chain, color in zip(KINEMATIC_CHAINS, CHAIN_COLORS):
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i + 1]
            xs = [positions[j1, 0], positions[j2, 0]]
            ys = [positions[j1, 2], positions[j2, 2]]
            zs = [positions[j1, 1], positions[j2, 1]]
            ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=alpha)

    # Draw joints (swap Y/Z)
    ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
               c="black", s=15, depthshade=True, alpha=alpha)

    # Axis settings - auto-scale if not provided
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)   # data Z -> plot Y
        ax.set_zlim(ylim)   # data Y (height) -> plot Z
    else:
        center = positions.mean(axis=0)
        max_range = max(positions.max(axis=0) - positions.min(axis=0)) / 2 + 0.2
        max_range = max(max_range, 0.5)  # minimum range
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)  # data Z -> plot Y
        ax.set_zlim(center[1] - max_range, center[1] + max_range)  # data Y -> plot Z

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.set_title(title)
    ax.view_init(elev=15, azim=45)


def _compute_axis_limits(positions: np.ndarray, padding: float = 0.3):
    """Compute fixed axis limits from all frames to prevent jittering.

    Args:
        positions: (T, 22, 3) joint positions across all frames
    """
    all_min = positions.reshape(-1, 3).min(axis=0)
    all_max = positions.reshape(-1, 3).max(axis=0)
    center = (all_min + all_max) / 2
    max_range = (all_max - all_min).max() / 2 + padding
    max_range = max(max_range, 0.5)
    xlim = (center[0] - max_range, center[0] + max_range)
    ylim = (center[1] - max_range, center[1] + max_range)
    zlim = (center[2] - max_range, center[2] + max_range)
    return xlim, ylim, zlim


def render_motion_animation(
    positions: np.ndarray,
    output_path: str,
    fps: int = 20,
    title: str = "Generated Motion",
) -> str:
    """Render a skeleton animation video.

    Args:
        positions: (T, 22, 3) joint positions
        output_path: path to save (supports .mp4, .gif)
        fps: frames per second
        title: plot title
    Returns:
        output_path
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    xlim, ylim, zlim = _compute_axis_limits(positions)

    def update(frame):
        plot_skeleton_frame(ax, positions[frame], title=f"{title} (frame {frame})",
                            xlim=xlim, ylim=ylim, zlim=zlim)

    T = positions.shape[0]
    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".gif":
        anim.save(str(out), writer="pillow", fps=fps)
    else:
        anim.save(str(out), writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Animation saved to {out}")
    return str(out)


def render_comparison(
    base_positions: np.ndarray,
    lora_positions: np.ndarray,
    output_path: str,
    fps: int = 20,
    title: str = "Base vs LoRA",
):
    """Side-by-side comparison animation of base model vs LoRA model."""
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    T = min(base_positions.shape[0], lora_positions.shape[0])
    # Use shared axis limits so comparison is fair
    combined = np.concatenate([base_positions[:T], lora_positions[:T]], axis=0)
    xlim, ylim, zlim = _compute_axis_limits(combined)

    def update(frame):
        plot_skeleton_frame(ax1, base_positions[frame], title="Base Model",
                            xlim=xlim, ylim=ylim, zlim=zlim)
        plot_skeleton_frame(ax2, lora_positions[frame], title="With LoRA",
                            xlim=xlim, ylim=ylim, zlim=zlim)

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".gif":
        anim.save(str(out), writer="pillow", fps=fps)
    else:
        anim.save(str(out), writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Comparison animation saved to {out}")


def save_frame_sequence(
    positions: np.ndarray,
    output_dir: str,
    step: int = 5,
    title: str = "",
):
    """Save every N-th frame as a static PNG image."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, positions.shape[0], step):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        plot_skeleton_frame(ax, positions[i], title=f"{title} frame={i}")
        fig.savefig(out_dir / f"frame_{i:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
