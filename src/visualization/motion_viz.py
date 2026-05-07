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


def _smooth_root_trajectory(positions: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooth root XZ trajectory to reduce accumulated drift."""
    T = positions.shape[0]
    if T < window * 2:
        return positions

    root_vx = np.diff(positions[:, 0, 0])
    root_vz = np.diff(positions[:, 0, 2])

    kernel = np.ones(window) / window
    root_vx_smooth = np.convolve(root_vx, kernel, mode='same')
    root_vz_smooth = np.convolve(root_vz, kernel, mode='same')

    new_root_x = np.cumsum(np.concatenate([[positions[0, 0, 0]], root_vx_smooth]))
    new_root_z = np.cumsum(np.concatenate([[positions[0, 0, 2]], root_vz_smooth]))

    dx = new_root_x - positions[:, 0, 0]
    dz = new_root_z - positions[:, 0, 2]
    positions[:, :, 0] += dx[:, None]
    positions[:, :, 2] += dz[:, None]

    return positions


def _enforce_bone_lengths(positions: np.ndarray) -> np.ndarray:
    """Enforce consistent bone lengths across all frames."""
    T = positions.shape[0]
    bone_pairs = []
    for chain in KINEMATIC_CHAINS:
        for i in range(len(chain) - 1):
            bone_pairs.append((chain[i], chain[i + 1]))

    all_lengths = np.zeros((T, len(bone_pairs)))
    for t in range(T):
        for b, (j1, j2) in enumerate(bone_pairs):
            all_lengths[t, b] = np.linalg.norm(positions[t, j2] - positions[t, j1])

    median_lengths = np.median(all_lengths, axis=0)

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


def _detect_contact_phases(contact_signal: np.ndarray, positions: np.ndarray,
                           foot_idx: int, ankle_idx: int) -> list:
    """Detect contiguous foot-contact phases using contact signal + height + velocity.

    Returns list of (start, end) frame ranges where foot is grounded.
    """
    T = len(contact_signal)
    grounded = np.zeros(T, dtype=bool)

    for t in range(T):
        by_signal = contact_signal[t] > 0.3
        by_height = positions[t, foot_idx, 1] < 0.25
        vel = np.linalg.norm(positions[t, foot_idx] - positions[max(0, t - 1), foot_idx]) if t > 0 else 0.0
        by_vel = vel < 0.05
        grounded[t] = by_signal or (by_height and by_vel)

    # Merge short gaps (< 3 frames) between contact phases
    for t in range(1, T - 1):
        if not grounded[t] and grounded[t - 1] and any(grounded[t + 1:min(t + 4, T)]):
            grounded[t] = True

    # Extract contiguous phases
    phases = []
    in_phase = False
    start = 0
    for t in range(T):
        if grounded[t] and not in_phase:
            start = t
            in_phase = True
        elif not grounded[t] and in_phase:
            if t - start >= 2:  # ignore single-frame contacts
                phases.append((start, t))
            in_phase = False
    if in_phase and T - start >= 2:
        phases.append((start, T))

    return phases


def _fix_foot_sliding(positions: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Phase-based foot sliding fix.

    For each detected contact phase, locks the foot to a single XZ position.
    Smoothly blends at phase boundaries to avoid discontinuities.
    """
    T = positions.shape[0]
    if motion.shape[1] < 263:
        return positions

    foot_contact = motion[:, 259:263]
    l_contact = (foot_contact[:, 0] + foot_contact[:, 1]) / 2
    r_contact = (foot_contact[:, 2] + foot_contact[:, 3]) / 2

    blend_frames = 3  # frames to blend at phase boundaries

    for contact, foot_j, ankle_j in [
        (l_contact, 10, 7),
        (r_contact, 11, 8),
    ]:
        phases = _detect_contact_phases(contact, positions, foot_j, ankle_j)

        for start, end in phases:
            # Lock foot XZ to the position at phase start
            lock_x = positions[start, foot_j, 0]
            lock_z = positions[start, foot_j, 2]
            lock_ax = positions[start, ankle_j, 0]
            lock_az = positions[start, ankle_j, 2]

            for t in range(start, end):
                # Compute blend weight: smooth transition at boundaries
                if t < start + blend_frames:
                    w = (t - start) / blend_frames
                elif t > end - blend_frames - 1:
                    w = (end - 1 - t) / blend_frames
                else:
                    w = 1.0
                w = max(0.0, min(1.0, w))

                # Blend between original and locked position
                positions[t, foot_j, 0] = w * lock_x + (1 - w) * positions[t, foot_j, 0]
                positions[t, foot_j, 2] = w * lock_z + (1 - w) * positions[t, foot_j, 2]
                positions[t, ankle_j, 0] = w * lock_ax + (1 - w) * positions[t, ankle_j, 0]
                positions[t, ankle_j, 2] = w * lock_az + (1 - w) * positions[t, ankle_j, 2]

                # Also clamp foot height to ground during contact
                if w > 0.5:
                    positions[t, foot_j, 1] = min(positions[t, foot_j, 1], 0.05)

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

    # Step 1: Root trajectory smoothing (reduces drift/turning)
    positions = _smooth_root_trajectory(positions, window=7)

    # Step 2: Bone length enforcement FIRST (before foot fix)
    positions = _enforce_bone_lengths(positions)

    # Step 3: Phase-based foot sliding fix LAST (so bone enforcement doesn't undo it)
    positions = _fix_foot_sliding(positions, motion)

    # Step 4: Fix root visual position
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
