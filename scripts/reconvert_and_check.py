"""Reconvert 100STYLE BVH with fixed rotations, then check normalization compatibility.

Usage:
    python scripts/reconvert_and_check.py
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.bvh_converter import BVHToHumanML3D

STYLE_DIR = "/transfer/loradataset/100STYLE"
OUTPUT_DIR = "/transfer/loradataset/style_converted_v2"
HML3D_DIR = "/transfer/loradataset/humanml3d"

STYLES = ["Zombie", "Elated", "Old", "Depressed", "Drunk"]

def main():
    converter = BVHToHumanML3D()

    # Convert each style
    for style in STYLES:
        style_bvh_dir = Path(STYLE_DIR) / style
        if not style_bvh_dir.exists():
            # Try finding BVH files with style in name
            style_bvh_dir = Path(STYLE_DIR)
            bvh_files = sorted(style_bvh_dir.glob(f"*{style}*.bvh"))
            if not bvh_files:
                print(f"  {style}: no BVH files found, skipping")
                continue
            # Convert individual files
            print(f"\n  Converting {style} ({len(bvh_files)} files)...")
            converter.convert_directory(str(style_bvh_dir), OUTPUT_DIR, style_label=style.lower())
        else:
            print(f"\n  Converting {style}...")
            converter.convert_directory(str(style_bvh_dir), OUTPUT_DIR, style_label=style.lower())

    # Check normalization compatibility
    print("\n" + "=" * 50)
    print("Checking normalization compatibility")
    print("=" * 50)

    hml_mean = np.load(Path(HML3D_DIR) / "Mean.npy")
    hml_std = np.load(Path(HML3D_DIR) / "Std.npy")
    hml_std_safe = hml_std.copy()
    hml_std_safe[hml_std_safe < 1e-5] = 1.0

    # Load converted data
    motion_dir = Path(OUTPUT_DIR) / "motions"
    npy_files = sorted(motion_dir.glob("*.npy"))
    if not npy_files:
        print("  No converted files found!")
        return

    all_data = np.concatenate([np.load(f) for f in npy_files], axis=0)
    normed = (all_data - hml_mean) / hml_std_safe

    # Load HumanML3D reference
    hml_motion_dir = Path(HML3D_DIR) / "new_joint_vecs"
    hml_files = sorted(hml_motion_dir.glob("*.npy"))[:100]
    hml_data = np.concatenate([np.load(f) for f in hml_files], axis=0)
    hml_normed = (hml_data - hml_mean) / hml_std_safe

    print(f"\n  Converted data: {all_data.shape[0]} frames, {len(npy_files)} files")
    print(f"  HumanML3D ref:  {hml_data.shape[0]} frames, {len(hml_files)} files")

    print(f"\n  --- Raw feature ranges ---")
    print(f"  Converted:  [{all_data.min():.3f}, {all_data.max():.3f}]")
    print(f"  HumanML3D:  [{hml_data.min():.3f}, {hml_data.max():.3f}]")

    print(f"\n  --- After HumanML3D normalization ---")
    print(f"  Converted:  [{normed.min():.1f}, {normed.max():.1f}]")
    print(f"  HumanML3D:  [{hml_normed.min():.1f}, {hml_normed.max():.1f}]")

    # Per-feature-group analysis
    groups = [
        ("root_rot_vel", 0, 1),
        ("root_vel_xz", 1, 3),
        ("root_height", 3, 4),
        ("joint_positions", 4, 67),
        ("joint_rotations_6d", 67, 193),
        ("joint_velocities", 193, 259),
        ("foot_contacts", 259, 263),
    ]

    print(f"\n  --- Per-feature-group normalized ranges ---")
    print(f"  {'Group':<25s} {'Converted':>20s} {'HumanML3D':>20s}")
    for name, start, end in groups:
        c_range = f"[{normed[:, start:end].min():.1f}, {normed[:, start:end].max():.1f}]"
        h_range = f"[{hml_normed[:, start:end].min():.1f}, {hml_normed[:, start:end].max():.1f}]"
        print(f"  {name:<25s} {c_range:>20s} {h_range:>20s}")

    pct_clipped = (np.abs(normed) > 5).mean() * 100
    print(f"\n  Values outside [-5, 5]: {pct_clipped:.1f}%")
    if pct_clipped < 2:
        print("  GOOD: Data is compatible with HumanML3D normalization!")
    elif pct_clipped < 10:
        print("  OK: Minor clipping needed, should work.")
    else:
        print("  WARNING: Still significant mismatch.")


if __name__ == "__main__":
    main()
