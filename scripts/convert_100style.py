"""Convert 100STYLE BVH dataset to HumanML3D-compatible format for LoRA training.

100STYLE dataset: https://www.ianmaurice.com/100style/
Contains 100 locomotion styles with BVH motion capture data.

Skips styles that have already been converted (checks for existing metadata.jsonl).

Usage:
    # Convert specific styles (uses /transfer paths by default)
    python scripts/convert_100style.py --styles Zombie,Happy,Old

    # Convert all styles
    python scripts/convert_100style.py --all

    # List available styles
    python scripts/convert_100style.py --list

    # Force reconvert (ignore existing)
    python scripts/convert_100style.py --styles Zombie --force
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bvh_converter import BVHToHumanML3D


def list_styles(input_dir: str):
    """List all available style directories."""
    root = Path(input_dir)
    # 100STYLE organizes by style name in subdirectories
    # or as prefixed filenames like "Zombie_Walk_001.bvh"

    # Check for subdirectories first
    subdirs = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if subdirs:
        print(f"Found {len(subdirs)} style directories:")
        for d in subdirs:
            bvh_count = len(list((root / d).glob("*.bvh")))
            print(f"  {d}: {bvh_count} BVH files")
        return subdirs

    # Otherwise, extract style from filenames
    bvh_files = sorted(root.glob("*.bvh"))
    styles = set()
    for f in bvh_files:
        parts = f.stem.split("_")
        if parts:
            styles.add(parts[0])
    styles = sorted(styles)
    print(f"Found {len(styles)} styles from {len(bvh_files)} BVH files:")
    for s in styles:
        count = len([f for f in bvh_files if f.stem.startswith(s)])
        print(f"  {s}: {count} files")
    return styles


def convert_styles(input_dir: str, output_dir: str, styles: list[str], force: bool = False):
    """Convert selected styles to HumanML3D format. Skips already-converted styles."""
    root = Path(input_dir)
    converter = BVHToHumanML3D(target_fps=20)

    all_metadata = []

    for style in styles:
        print(f"\n{'=' * 40}")
        print(f"Converting style: {style}")
        print(f"{'=' * 40}")

        style_out = Path(output_dir) / style

        # Skip if already converted
        if not force and (style_out / "metadata.jsonl").exists():
            existing = sum(1 for _ in open(style_out / "metadata.jsonl"))
            print(f"  Already converted ({existing} motions), skipping. Use --force to reconvert.")
            continue

        style_out.mkdir(parents=True, exist_ok=True)

        # Check if style is a subdirectory
        style_dir = root / style
        if style_dir.is_dir():
            bvh_dir = str(style_dir)
        else:
            # Filter files by style prefix
            bvh_files = sorted(root.glob(f"{style}*.bvh"))
            if not bvh_files:
                bvh_files = sorted(root.glob(f"{style.lower()}*.bvh"))
            if not bvh_files:
                print(f"  No BVH files found for style '{style}', skipping")
                continue

            # Create temp directory with symlinks or copy files
            tmp_dir = style_out / "_bvh_temp"
            tmp_dir.mkdir(exist_ok=True)
            for f in bvh_files:
                import shutil
                shutil.copy2(f, tmp_dir / f.name)
            bvh_dir = str(tmp_dir)

        metadata = converter.convert_directory(bvh_dir, str(style_out), style_label=style.lower())
        all_metadata.extend(metadata)
        print(f"  Converted {len(metadata)} motions for style '{style}'")

    # Save combined metadata (always rebuild from all style subdirs)
    combined_out = Path(output_dir)
    combined_motions = combined_out / "motions"
    combined_motions.mkdir(parents=True, exist_ok=True)

    # Merge all style motions into one directory with combined metadata
    combined_meta = []
    import shutil
    for style in styles:
        style_dir = Path(output_dir) / style / "motions"
        if not style_dir.exists():
            continue
        for npy_file in style_dir.glob("*.npy"):
            # Prefix with style name to avoid collisions
            new_name = f"{style.lower()}_{npy_file.name}"
            shutil.copy2(npy_file, combined_motions / new_name)

        meta_file = Path(output_dir) / style / "metadata.jsonl"
        if meta_file.exists():
            with open(meta_file) as f:
                for line in f:
                    entry = json.loads(line)
                    entry["file"] = f"{style.lower()}_{entry['file']}"
                    combined_meta.append(entry)

    with open(combined_out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for entry in combined_meta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nCombined dataset: {len(combined_meta)} motions in {combined_out}")
    return combined_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 100STYLE BVH to HumanML3D format")
    parser.add_argument("--input", type=str, default="/transfer/datasets/100STYLE",
                        help="Path to 100STYLE dataset")
    parser.add_argument("--output", type=str, default="/transfer/datasets/style_converted",
                        help="Output directory for converted data")
    parser.add_argument("--styles", type=str, help="Comma-separated style names")
    parser.add_argument("--all", action="store_true", help="Convert all styles")
    parser.add_argument("--list", action="store_true", help="List available styles")
    parser.add_argument("--force", action="store_true", help="Force reconvert even if already done")

    args = parser.parse_args()

    if args.list:
        list_styles(args.input)
    elif args.all:
        styles = list_styles(args.input)
        convert_styles(args.input, args.output, styles, force=args.force)
    elif args.styles:
        styles = [s.strip() for s in args.styles.split(",")]
        convert_styles(args.input, args.output, styles, force=args.force)
    else:
        parser.print_help()
