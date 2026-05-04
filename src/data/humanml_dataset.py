"""HumanML3D dataset loader for motion diffusion training.

Expected directory structure (after running prepare_data.py):
    data/HumanML3D/
        new_joint_vecs/     # .npy motion features (T, 263)
        texts/              # .txt caption files (multiple captions per motion)
        Mean.npy            # feature-wise mean
        Std.npy             # feature-wise std
        train.txt / val.txt / test.txt   # split lists
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class HumanML3DDataset(Dataset):
    """HumanML3D dataset with text conditioning for MDM training."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        nfeats: int = 263,
        unit_length: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.nfeats = nfeats
        self.unit_length = unit_length

        # Load normalization stats
        self.mean = np.load(self.data_dir / "Mean.npy")  # (263,)
        self.std = np.load(self.data_dir / "Std.npy")    # (263,)
        self.std[self.std < 1e-5] = 1.0  # avoid division by zero

        # Load split
        split_file = self.data_dir / f"{split}.txt"
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # Pre-filter by motion length and load captions
        self.data = []
        motion_dir = self.data_dir / "new_joint_vecs"
        text_dir = self.data_dir / "texts"

        for motion_id in self.ids:
            motion_file = motion_dir / f"{motion_id}.npy"
            text_file = text_dir / f"{motion_id}.txt"
            if not motion_file.exists() or not text_file.exists():
                continue

            motion = np.load(motion_file)
            if motion.shape[0] < min_motion_length or motion.shape[0] > 600:
                continue

            # Parse captions (HumanML3D format: "caption#start#end" per line)
            captions = []
            with open(text_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("#")
                    if len(parts) >= 1 and parts[0]:
                        captions.append(parts[0].strip())
            if not captions:
                continue

            self.data.append({
                "id": motion_id,
                "motion_path": str(motion_file),
                "captions": captions,
                "length": motion.shape[0],
            })

        print(f"HumanML3D [{split}]: {len(self.data)} motions loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        motion = np.load(entry["motion_path"])  # (T, 263)

        # Random crop to max_motion_length
        T = motion.shape[0]
        if T > self.max_motion_length:
            start = random.randint(0, T - self.max_motion_length)
            motion = motion[start:start + self.max_motion_length]
            T = self.max_motion_length

        # Normalize
        motion = (motion - self.mean) / self.std

        # Pad to max_motion_length
        pad_len = self.max_motion_length - T
        mask = np.zeros(self.max_motion_length, dtype=bool)
        if pad_len > 0:
            motion = np.concatenate([motion, np.zeros((pad_len, self.nfeats))], axis=0)
            mask[T:] = True  # True = padding

        # Random caption
        caption = random.choice(entry["captions"])

        return {
            "motion": torch.tensor(motion, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "caption": caption,
            "length": T,
            "id": entry["id"],
        }


class StyleMotionDataset(Dataset):
    """Dataset for style-specific LoRA fine-tuning (e.g. 100STYLE data).

    Expects motions converted to HumanML3D 263-dim format via bvh_converter.
    Directory structure:
        style_data_dir/
            motions/          # .npy files (T, 263)
            metadata.jsonl    # {"file": "xxx.npy", "action": "walk", "style": "zombie", "caption": "..."}
    """

    def __init__(
        self,
        data_dir: str,
        mean: np.ndarray,
        std: np.ndarray,
        max_motion_length: int = 196,
        nfeats: int = 263,
    ):
        self.data_dir = Path(data_dir)
        self.max_motion_length = max_motion_length
        self.nfeats = nfeats
        self.mean = mean
        self.std = std.copy()
        self.std[self.std < 1e-5] = 1.0

        metadata_path = self.data_dir / "metadata.jsonl"
        self.entries = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.entries.append(entry)

        print(f"StyleMotionDataset: {len(self.entries)} motions from {data_dir}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        motion_path = self.data_dir / "motions" / entry["file"]
        motion = np.load(motion_path)  # (T, 263)

        T = motion.shape[0]
        if T > self.max_motion_length:
            start = random.randint(0, T - self.max_motion_length)
            motion = motion[start:start + self.max_motion_length]
            T = self.max_motion_length

        motion = (motion - self.mean) / self.std

        pad_len = self.max_motion_length - T
        mask = np.zeros(self.max_motion_length, dtype=bool)
        if pad_len > 0:
            motion = np.concatenate([motion, np.zeros((pad_len, self.nfeats))], axis=0)
            mask[T:] = True

        caption = entry.get("caption", f"a person {entry.get('action', 'moving')} in {entry.get('style', 'normal')} style")

        return {
            "motion": torch.tensor(motion, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "caption": caption,
            "length": T,
            "action": entry.get("action", ""),
            "style": entry.get("style", ""),
        }
