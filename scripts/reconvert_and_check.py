"""Reconvert 100STYLE BVH with fixed rotations, then check normalization compatibility.

Outputs per-style directories for single-style LoRA training:
    output_dir/
        zombie/motions/ + metadata.jsonl
        elated/motions/ + metadata.jsonl
        ...
        mixed/motions/ + metadata.jsonl  (all styles combined)

Usage:
    python scripts/reconvert_and_check.py
"""
from __future__ import annotations
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.bvh_converter import BVHToHumanML3D

STYLE_DIR = "/transfer/loradataset/100STYLE"
OUTPUT_DIR = "/transfer/loradataset/style_bvh"
HML3D_DIR = "/transfer/loradataset/humanml3d"

STYLES = [
    # --- Original 5 ---
    "Zombie", "Elated", "Old", "Depressed", "Drunk",
    # --- Emotion / attitude ---
    "Angry", "Chicken", "Cat", "Dinosaur",
    # --- Physical characteristics ---
    "Heavyset", "BentKnees", "Crouched", "BigSteps",
    # --- Exaggerated / performative ---
    "HighKnees", "FairySteps", "Flapping", "KarateChop",
    # --- Asymmetric / constrained ---
    "DragLeftLeg", "HandsInPockets", "Sneaky",
]


def main():
    converter = BVHToHumanML3D()
    style_dir = Path(STYLE_DIR)
    out_dir = Path(OUTPUT_DIR)

    all_metadata = []  # for mixed

    for style in STYLES:
        style_lower = style.lower()
        style_out = out_dir / style_lower
        motions_dir = style_out / "motions"
        motions_dir.mkdir(parents=True, exist_ok=True)

        sub_dir = style_dir / style
        if sub_dir.is_dir():
            bvh_files = sorted(sub_dir.glob("*.bvh"))
        else:
            bvh_files = sorted(style_dir.glob(f"*{style}*.bvh"))

        if not bvh_files:
            print(f"  {style}: no BVH files found, skipping")
            continue

        print(f"\nConverting {style} ({len(bvh_files)} files)...")
        style_metadata = []
        for bvh_file in bvh_files:
            features = converter.convert(str(bvh_file))
            if features is None:
                continue

            stem = bvh_file.stem
            parts = stem.split("_")
            action = parts[0].lower() if parts else "walk"

            out_file = f"{stem}.npy"
            np.save(motions_dir / out_file, features)

            caption = f"a person {action}ing in {style_lower} style"
            entry = {
                "file": out_file,
                "action": action,
                "style": style_lower,
                "caption": caption,
                "length": features.shape[0],
            }
            style_metadata.append(entry)
            all_metadata.append(entry)

        # Write per-style metadata
        with open(style_out / "metadata.jsonl", "w", encoding="utf-8") as f:
            for entry in style_metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  {style}: {len(style_metadata)} motions -> {style_out}")

    # Write mixed (all styles combined)
    mixed_dir = out_dir / "mixed"
    mixed_motions = mixed_dir / "motions"
    mixed_motions.mkdir(parents=True, exist_ok=True)

    for style in STYLES:
        style_lower = style.lower()
        src_dir = out_dir / style_lower / "motions"
        if not src_dir.exists():
            continue
        for npy in src_dir.glob("*.npy"):
            import shutil
            shutil.copy2(npy, mixed_motions / npy.name)

    with open(mixed_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for entry in all_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_metadata)} motions")
    print(f"  Mixed -> {mixed_dir}")

    if not all_metadata:
        print("No data converted! Check 100STYLE directory structure.")
        print(f"\nContents of {style_dir}:")
        for item in sorted(style_dir.iterdir())[:20]:
            print(f"  {item.name}")
        return

    # --- Check normalization compatibility ---
    print("\n" + "=" * 50)
    print("Checking normalization compatibility")
    print("=" * 50)

    hml_mean = np.load(Path(HML3D_DIR) / "Mean.npy")
    hml_std = np.load(Path(HML3D_DIR) / "Std.npy")
    hml_std_safe = hml_std.copy()
    hml_std_safe[hml_std_safe < 1e-5] = 1.0

    # Load all converted data
    all_npy = []
    for style in STYLES:
        d = out_dir / style.lower() / "motions"
        if d.exists():
            all_npy.extend(sorted(d.glob("*.npy")))

    all_data = np.concatenate([np.load(f) for f in all_npy], axis=0)
    normed = (all_data - hml_mean) / hml_std_safe

    # Load HumanML3D reference
    hml_motion_dir = Path(HML3D_DIR) / "new_joint_vecs"
    hml_files = sorted(hml_motion_dir.glob("*.npy"))[:100]
    hml_data = np.concatenate([np.load(f) for f in hml_files], axis=0)
    hml_normed = (hml_data - hml_mean) / hml_std_safe

    print(f"\n  Converted: {all_data.shape[0]} frames, {len(all_npy)} files")
    print(f"  HumanML3D: {hml_data.shape[0]} frames, {len(hml_files)} files")

    print(f"\n  --- After HumanML3D normalization ---")
    print(f"  Converted:  [{normed.min():.1f}, {normed.max():.1f}]")
    print(f"  HumanML3D:  [{hml_normed.min():.1f}, {hml_normed.max():.1f}]")

    groups = [
        ("root_rot_vel", 0, 1),
        ("root_vel_xz", 1, 3),
        ("root_height", 3, 4),
        ("joint_positions", 4, 67),
        ("joint_rotations_6d", 67, 193),
        ("joint_velocities", 193, 259),
        ("foot_contacts", 259, 263),
    ]

    print(f"\n  --- Per-group normalized ranges ---")
    print(f"  {'Group':<25s} {'Converted':>20s} {'HumanML3D':>20s}")
    for name, start, end in groups:
        c_range = f"[{normed[:, start:end].min():.1f}, {normed[:, start:end].max():.1f}]"
        h_range = f"[{hml_normed[:, start:end].min():.1f}, {hml_normed[:, start:end].max():.1f}]"
        print(f"  {name:<25s} {c_range:>20s} {h_range:>20s}")

    pct_outside = (np.abs(normed) > 5).mean() * 100
    print(f"\n  Values outside [-5, 5]: {pct_outside:.1f}%")


if __name__ == "__main__":
    main()
