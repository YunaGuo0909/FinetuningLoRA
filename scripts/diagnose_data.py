"""Diagnose NaN loss issue by comparing converted style data with HumanML3D data."""

import numpy as np
from pathlib import Path

print("=" * 50)
print("Data Diagnosis")
print("=" * 50)

# Check converted style data
motions_dir = Path("/transfer/loradataset/style_converted/motions")
files = sorted(motions_dir.glob("*.npy"))
print(f"\nConverted style data: {len(files)} files")

nan_count = 0
inf_count = 0
for f in files:
    m = np.load(f)
    nans = np.isnan(m).sum()
    infs = np.isinf(m).sum()
    nan_count += nans
    inf_count += infs

for f in files[:5]:
    m = np.load(f)
    print(f"  {f.name}: shape={m.shape}, min={m.min():.2f}, max={m.max():.2f}, nan={np.isnan(m).sum()}, inf={np.isinf(m).sum()}")

print(f"  Total NaN: {nan_count}, Total Inf: {inf_count}")

# Check HumanML3D data for comparison
hml_dir = Path("/transfer/loradataset/humanml3d/new_joint_vecs")
hml_files = sorted(hml_dir.glob("*.npy"))
print(f"\nHumanML3D data: {len(hml_files)} files")

for f in hml_files[:5]:
    m = np.load(f)
    print(f"  {f.name}: shape={m.shape}, min={m.min():.2f}, max={m.max():.2f}")

# Check mean/std
mean = np.load("/transfer/loradataset/humanml3d/Mean.npy")
std = np.load("/transfer/loradataset/humanml3d/Std.npy")
print(f"\nNormalization stats:")
print(f"  Mean: shape={mean.shape}, range=[{mean.min():.2f}, {mean.max():.2f}]")
print(f"  Std:  shape={std.shape}, range=[{std.min():.4f}, {std.max():.2f}]")
print(f"  Std near-zero (<1e-5): {(std < 1e-5).sum()} dims")

# Test normalization on style data
print(f"\nAfter normalization (style data):")
std_safe = std.copy()
std_safe[std_safe < 1e-5] = 1.0
for f in files[:3]:
    m = np.load(f)
    normed = (m - mean) / std_safe
    print(f"  {f.name}: min={normed.min():.2f}, max={normed.max():.2f}, nan={np.isnan(normed).sum()}")

print(f"\nAfter normalization (HumanML3D):")
for f in hml_files[:3]:
    m = np.load(f)
    normed = (m - mean) / std_safe
    print(f"  {f.name}: min={normed.min():.2f}, max={normed.max():.2f}")
