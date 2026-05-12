"""Download and prepare datasets on the remote training machine.

All data goes under /transfer/. Skips downloads if files already exist.

Usage:
    python scripts/prepare_data.py                    # prepare everything
    python scripts/prepare_data.py --only humanml3d   # just HumanML3D
    python scripts/prepare_data.py --only 100style    # just 100STYLE
    python scripts/prepare_data.py --only pretrained   # just MDM weights
    python scripts/prepare_data.py --verify            # verify only, no download
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
import tarfile
import numpy as np
from pathlib import Path


TRANSFER_ROOT = Path("/transfer")
DATASETS_DIR = TRANSFER_ROOT / "loradataset"
PRETRAINED_DIR = TRANSFER_ROOT / "lorapretrain" / "humanml_trans_enc_512" / "humanml_trans_enc_512"

HUMANML3D_DIR = DATASETS_DIR / "humanml3d"
STYLE100_DIR = DATASETS_DIR / "100STYLE"
MDM_WEIGHTS_PATH = PRETRAINED_DIR / "model000475000.pt"

# HumanML3D repo (contains processed data links)
HUMANML3D_REPO = "https://github.com/EricGuo5513/HumanML3D.git"
# MDM pretrained weights (Google Drive file ID from official repo)
MDM_GDRIVE_ID = "1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821"
# 100STYLE direct download
STYLE100_URL = "https://www.ianmaurice.com/100style/100STYLE.zip"



def exists_and_nonempty(path: Path) -> bool:
    """Check if path exists and is non-empty (file > 0 bytes, dir has children)."""
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def run_cmd(cmd: list[str] | str, **kwargs):
    """Run shell command, print it first."""
    if isinstance(cmd, list):
        print(f"  $ {' '.join(cmd)}")
    else:
        print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kwargs)


def download_gdrive(file_id: str, output_path: str):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        run_cmd([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    gdown.download(id=file_id, output=output_path, quiet=False)



def prepare_humanml3d():
    """Download and prepare HumanML3D dataset."""
    print("\n" + "=" * 50)
    print("Preparing HumanML3D dataset")
    print("=" * 50)

    required_items = [
        HUMANML3D_DIR / "new_joint_vecs",
        HUMANML3D_DIR / "texts",
        HUMANML3D_DIR / "Mean.npy",
        HUMANML3D_DIR / "Std.npy",
        HUMANML3D_DIR / "train.txt",
    ]

    # Check if already prepared
    if all(exists_and_nonempty(p) for p in required_items):
        print("  Already exists and complete, skipping.")
        return True

    HUMANML3D_DIR.mkdir(parents=True, exist_ok=True)

    # Clone HumanML3D repo to get processing scripts and text annotations
    repo_dir = DATASETS_DIR / "HumanML3D_repo"
    if not exists_and_nonempty(repo_dir / "HumanML3D"):
        print("  Cloning HumanML3D repository...")
        if repo_dir.exists():
            import shutil
            shutil.rmtree(repo_dir)
        run_cmd(["git", "clone", HUMANML3D_REPO, str(repo_dir)])
    else:
        print("  HumanML3D repo already cloned.")

    # The repo provides Google Drive links for processed data
    # Check for the key processed files
    source_vecs = repo_dir / "HumanML3D" / "new_joint_vecs"
    source_texts = repo_dir / "HumanML3D" / "texts"

    if exists_and_nonempty(source_vecs):
        print("  Copying processed motion vectors...")
        import shutil
        if not exists_and_nonempty(HUMANML3D_DIR / "new_joint_vecs"):
            shutil.copytree(source_vecs, HUMANML3D_DIR / "new_joint_vecs", dirs_exist_ok=True)
    else:
        print("  WARNING: Processed motion vectors not found in repo.")
        print("  You may need to download them manually from the HumanML3D Google Drive.")
        print("  See: https://github.com/EricGuo5513/HumanML3D#download")

    if exists_and_nonempty(source_texts):
        print("  Copying text annotations...")
        import shutil
        if not exists_and_nonempty(HUMANML3D_DIR / "texts"):
            shutil.copytree(source_texts, HUMANML3D_DIR / "texts", dirs_exist_ok=True)
    else:
        if not exists_and_nonempty(HUMANML3D_DIR / "texts"):
            print("  WARNING: Text annotations not found in repo (stored on Google Drive).")
            print("  Download 'texts.zip' from the HumanML3D Google Drive and extract to:")
            print(f"    {HUMANML3D_DIR / 'texts'}")
            print("  Google Drive: https://drive.google.com/drive/folders/1MnixObHR8EB1yKoR1OGua9BMjECPbNBD")
            print("  Or from your local machine:")
            print(f"    scp -r texts.zip user@trainmachine:{HUMANML3D_DIR}/")
            print(f"    cd {HUMANML3D_DIR} && unzip texts.zip")

    # Copy split files and stats
    for fname in ["train.txt", "val.txt", "test.txt", "Mean.npy", "Std.npy"]:
        src = repo_dir / "HumanML3D" / fname
        dst = HUMANML3D_DIR / fname
        if src.exists() and not exists_and_nonempty(dst):
            import shutil
            shutil.copy2(src, dst)

    return verify_humanml3d()



def prepare_100style():
    """Download 100STYLE BVH dataset."""
    print("\n" + "=" * 50)
    print("Preparing 100STYLE dataset")
    print("=" * 50)

    # Check if already exists
    if exists_and_nonempty(STYLE100_DIR):
        bvh_count = len(list(STYLE100_DIR.rglob("*.bvh")))
        if bvh_count > 0:
            print(f"  Already exists ({bvh_count} BVH files), skipping.")
            return True

    STYLE100_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASETS_DIR / "100STYLE.zip"

    # Download
    if not exists_and_nonempty(zip_path):
        print("  Downloading 100STYLE dataset...")
        try:
            run_cmd(["wget", "-O", str(zip_path), STYLE100_URL])
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                run_cmd(["curl", "-L", "-o", str(zip_path), STYLE100_URL])
            except subprocess.CalledProcessError:
                print("  Download failed (no internet access?).")
                print("  Please download 100STYLE manually and transfer to training machine:")
                print(f"    1. Download from: {STYLE100_URL}")
                print(f"       Or: https://github.com/ianmaurice/100STYLE/releases")
                print(f"    2. scp 100STYLE.zip user@trainmachine:{zip_path}")
                print(f"    3. Re-run this script to extract.")
                return False
    else:
        print("  ZIP already downloaded.")

    # Extract
    if zip_path.exists():
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(STYLE100_DIR)
        print(f"  Extracted to {STYLE100_DIR}")

        # Check if files are in a subdirectory
        subdirs = [d for d in STYLE100_DIR.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not list(STYLE100_DIR.glob("*.bvh")):
            # Move contents up one level
            import shutil
            for item in subdirs[0].iterdir():
                shutil.move(str(item), str(STYLE100_DIR / item.name))
            subdirs[0].rmdir()

    bvh_count = len(list(STYLE100_DIR.rglob("*.bvh")))
    print(f"  100STYLE ready: {bvh_count} BVH files")
    return bvh_count > 0



def prepare_pretrained():
    """Download official MDM pretrained weights."""
    print("\n" + "=" * 50)
    print("Preparing MDM pretrained weights")
    print("=" * 50)

    if exists_and_nonempty(MDM_WEIGHTS_PATH):
        size_mb = MDM_WEIGHTS_PATH.stat().st_size / 1024 / 1024
        print(f"  Already exists ({size_mb:.1f} MB), skipping.")
        return True

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print("  Downloading MDM pretrained weights from Google Drive...")
    try:
        download_gdrive(MDM_GDRIVE_ID, str(MDM_WEIGHTS_PATH))
    except Exception as e:
        print(f"  Auto-download failed: {e}")
        print(f"  Please download manually from the MDM repo and place at:")
        print(f"    {MDM_WEIGHTS_PATH}")
        print(f"  URL: https://github.com/GuyTevet/motion-diffusion-model")
        return False

    if exists_and_nonempty(MDM_WEIGHTS_PATH):
        size_mb = MDM_WEIGHTS_PATH.stat().st_size / 1024 / 1024
        print(f"  Downloaded: {size_mb:.1f} MB")
        return True
    return False



def verify_humanml3d() -> bool:
    """Verify HumanML3D dataset completeness."""
    print("\n  Verifying HumanML3D...")
    checks = {
        "new_joint_vecs": HUMANML3D_DIR / "new_joint_vecs",
        "texts": HUMANML3D_DIR / "texts",
        "Mean.npy": HUMANML3D_DIR / "Mean.npy",
        "Std.npy": HUMANML3D_DIR / "Std.npy",
        "train.txt": HUMANML3D_DIR / "train.txt",
    }

    all_ok = True
    for name, path in checks.items():
        ok = exists_and_nonempty(path)
        status = "OK" if ok else "MISSING"
        if not ok:
            all_ok = False
        print(f"    [{status}] {name}")

    if all_ok:
        motions = list((HUMANML3D_DIR / "new_joint_vecs").glob("*.npy"))
        print(f"    Total motions: {len(motions)}")
        if motions:
            sample = np.load(motions[0])
            print(f"    Sample shape: {sample.shape} (expected (T, 263))")
        for split in ["train", "val", "test"]:
            sf = HUMANML3D_DIR / f"{split}.txt"
            if sf.exists():
                with open(sf) as f:
                    count = sum(1 for l in f if l.strip())
                print(f"    {split}: {count} sequences")

    return all_ok


def verify_all():
    """Verify all datasets."""
    print("\nVerification Report")
    print("=" * 50)

    results = {}
    results["HumanML3D"] = verify_humanml3d()

    print("\n  Verifying 100STYLE...")
    if exists_and_nonempty(STYLE100_DIR):
        bvh_count = len(list(STYLE100_DIR.rglob("*.bvh")))
        print(f"    [OK] {bvh_count} BVH files")
        results["100STYLE"] = True
    else:
        print(f"    [MISSING] {STYLE100_DIR}")
        results["100STYLE"] = False

    print("\n  Verifying MDM weights...")
    if exists_and_nonempty(MDM_WEIGHTS_PATH):
        size_mb = MDM_WEIGHTS_PATH.stat().st_size / 1024 / 1024
        print(f"    [OK] {size_mb:.1f} MB")
        results["MDM weights"] = True
    else:
        print(f"    [MISSING] {MDM_WEIGHTS_PATH}")
        results["MDM weights"] = False

    print(f"\n{'=' * 50}")
    for name, ok in results.items():
        print(f"  {name}: {'READY' if ok else 'NOT READY'}")
    return all(results.values())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets on training machine")
    parser.add_argument("--only", type=str, choices=["humanml3d", "100style", "pretrained"],
                        help="Only prepare specific dataset")
    parser.add_argument("--verify", action="store_true", help="Verify only, no downloads")
    args = parser.parse_args()

    if args.verify:
        verify_all()
        sys.exit(0)

    if args.only == "humanml3d":
        prepare_humanml3d()
    elif args.only == "100style":
        prepare_100style()
    elif args.only == "pretrained":
        prepare_pretrained()
    else:
        prepare_humanml3d()
        prepare_100style()
        prepare_pretrained()
        print("\n")
        verify_all()
