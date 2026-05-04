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

    # Draw bones
    for chain, color in zip(KINEMATIC_CHAINS, CHAIN_COLORS):
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i + 1]
            xs = [positions[j1, 0], positions[j2, 0]]
            ys = [positions[j1, 1], positions[j2, 1]]
            zs = [positions[j1, 2], positions[j2, 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=alpha)

    # Draw joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c="black", s=15, depthshade=True, alpha=alpha)

    # Axis settings - auto-scale if not provided
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    else:
        center = positions.mean(axis=0)
        max_range = max(positions.max(axis=0) - positions.min(axis=0)) / 2 + 0.2
        max_range = max(max_range, 0.5)  # minimum range
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z")
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
