"""Filter HumanML3D motions by text captions to create style-specific LoRA training sets.

Uses the existing HumanML3D data (already in correct feature space) instead of
BVH-converted data, avoiding normalization mismatch entirely.

Outputs:
    output_dir/
        old/
            motions/
            metadata.jsonl
        angry/
            motions/
            metadata.jsonl
        ...
        mixed/          # all styles combined
            motions/    (symlinks or copies)
            metadata.jsonl

Usage:
    python scripts/filter_style_data.py \
        --humanml3d_dir /transfer/loradataset/humanml3d \
        --output_dir /transfer/loradataset/style_filtered
"""

from __future__ import annotations

import argparse
import json
import shutil
import numpy as np
from pathlib import Path

# Style definitions: keywords to search in captions
STYLE_FILTERS = {
    "zombie": {
        "keywords": ["zombie", "stiff", "dragging", "limp", "stumbl"],
        "caption_override": "a person walking like a zombie",
    },
    "old": {
        "keywords": ["old", "elderly", "slow walk", "slowly walk", "careful", "weak", "frail", "hobble"],
        "caption_override": "a person walking slowly like an old person",
    },
    "drunk": {
        "keywords": ["drunk", "stumbl", "stagger", "sway", "unsteady", "wobbl", "dizzy"],
        "caption_override": "a person stumbling around drunk",
    },
    "happy": {
        "keywords": ["happy", "happily", "joyful", "cheerful", "excited", "energetic", "skip", "bounce", "hop"],
        "caption_override": "a person walking happily with energy",
    },
    "angry": {
        "keywords": ["angry", "angrily", "aggressive", "stomp", "march", "furious"],
        "caption_override": "a person walking angrily",
    },
    "sneak": {
        "keywords": ["sneak", "creep", "tiptoe", "stealth", "crouch walk", "quietly"],
        "caption_override": "a person sneaking quietly",
    },
    "tired": {
        "keywords": ["tired", "exhaust", "weary", "fatigue", "drag", "sluggish", "lethargic"],
        "caption_override": "a person walking while tired",
    },
    "sad": {
        "keywords": ["sad", "sadly", "depressed", "slouch", "dejected", "head down"],
        "caption_override": "a person walking sadly",
    },
}


def parse_captions(text_file: Path) -> list[str]:
    """Parse HumanML3D caption file."""
    captions = []
    with open(text_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("#")
            if parts and parts[0].strip():
                captions.append(parts[0].strip().lower())
    return captions


def match_style(captions: list[str], keywords: list[str]) -> bool:
    """Check if any caption matches any keyword."""
    for cap in captions:
        for kw in keywords:
            if kw in cap:
                return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--humanml3d_dir", type=str, default="/transfer/loradataset/humanml3d")
    p.add_argument("--output_dir", type=str, default="/transfer/loradataset/style_filtered")
    p.add_argument("--min_length", type=int, default=40)
    p.add_argument("--max_length", type=int, default=600)
    args = p.parse_args()

    hml_dir = Path(args.humanml3d_dir)
    out_dir = Path(args.output_dir)
    motion_dir = hml_dir / "new_joint_vecs"
    text_dir = hml_dir / "texts"

    # Collect all motion IDs
    train_file = hml_dir / "train.txt"
    with open(train_file) as f:
        all_ids = [line.strip() for line in f if line.strip()]

    print(f"Total training motions: {len(all_ids)}")
    print(f"Searching for {len(STYLE_FILTERS)} styles...\n")

    # Filter by style
    results = {}
    all_matched = set()

    for style_name, style_cfg in STYLE_FILTERS.items():
        keywords = style_cfg["keywords"]
        matched = []

        for mid in all_ids:
            motion_file = motion_dir / f"{mid}.npy"
            text_file = text_dir / f"{mid}.txt"
            if not motion_file.exists() or not text_file.exists():
                continue

            captions = parse_captions(text_file)
            if match_style(captions, keywords):
                motion = np.load(motion_file)
                if args.min_length <= motion.shape[0] <= args.max_length:
                    matched.append({
                        "id": mid,
                        "captions": captions,
                        "length": motion.shape[0],
                    })

        results[style_name] = matched
        all_matched.update(m["id"] for m in matched)
        print(f"  {style_name:10s}: {len(matched):4d} motions  (keywords: {keywords[:3]}...)")

    print(f"\n  Total unique: {len(all_matched)} motions")

    # --- Save per-style directories ---
    for style_name, matched in results.items():
        style_dir = out_dir / style_name
        motions_out = style_dir / "motions"
        motions_out.mkdir(parents=True, exist_ok=True)

        caption_override = STYLE_FILTERS[style_name]["caption_override"]
        metadata = []

        for m in matched:
            mid = m["id"]
            src = motion_dir / f"{mid}.npy"
            dst = motions_out / f"{mid}.npy"
            shutil.copy2(src, dst)

            original_caption = m["captions"][0] if m["captions"] else caption_override
            metadata.append({
                "file": f"{mid}.npy",
                "style": style_name,
                "caption": original_caption,
                "style_caption": caption_override,
                "length": m["length"],
            })

        with open(style_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"\n  {style_name}: {len(metadata)} motions -> {style_dir}")

    # --- Save mixed (all styles combined) ---
    mixed_dir = out_dir / "mixed"
    mixed_motions = mixed_dir / "motions"
    mixed_motions.mkdir(parents=True, exist_ok=True)

    mixed_metadata = []
    copied = set()

    for style_name, matched in results.items():
        caption_override = STYLE_FILTERS[style_name]["caption_override"]
        for m in matched:
            mid = m["id"]
            src = motion_dir / f"{mid}.npy"
            dst = mixed_motions / f"{mid}.npy"

            if mid not in copied:
                shutil.copy2(src, dst)
                copied.add(mid)

            original_caption = m["captions"][0] if m["captions"] else caption_override
            mixed_metadata.append({
                "file": f"{mid}.npy",
                "style": style_name,
                "caption": original_caption,
                "style_caption": caption_override,
                "length": m["length"],
            })

    with open(mixed_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for entry in mixed_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n  mixed: {len(mixed_metadata)} entries ({len(copied)} unique files) -> {mixed_dir}")

    # Summary
    print(f"\n{'=' * 50}")
    print("Summary")
    print(f"{'=' * 50}")
    for style_name in STYLE_FILTERS:
        count = len(results[style_name])
        print(f"  {style_name:10s}: {count:4d} motions  -> {out_dir / style_name}")
    print(f"  {'mixed':10s}: {len(mixed_metadata):4d} entries  -> {mixed_dir}")


if __name__ == "__main__":
    main()
