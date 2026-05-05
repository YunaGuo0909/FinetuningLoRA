"""Diagnose whether LoRA weights are loaded and having any effect.

Run on training machine:
    python scripts/diagnose_lora.py
"""

from __future__ import annotations
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mdm_official import (
    load_official_mdm,
    replace_attention_layers,
    motion_to_mdm_input,
    build_y_dict,
)
from src.models.diffusion import GaussianDiffusion

CHECKPOINT_DIR = "/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
LORA_PATH = "/transfer/loraoutputs/models/style_lora_v3/final"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Check LoRA files ---
    lora_path = Path(LORA_PATH)
    print(f"\n{'='*50}")
    print("1. Checking LoRA files")
    print(f"{'='*50}")
    if not lora_path.exists():
        print(f"  ERROR: LoRA path does not exist: {lora_path}")
        return
    files = list(lora_path.iterdir())
    print(f"  Path: {lora_path}")
    print(f"  Files: {[f.name for f in files]}")

    # Check adapter weights
    adapter_file = lora_path / "adapter_model.bin"
    safetensors_file = lora_path / "adapter_model.safetensors"
    if adapter_file.exists():
        weights = torch.load(str(adapter_file), map_location="cpu")
    elif safetensors_file.exists():
        from safetensors.torch import load_file
        weights = load_file(str(safetensors_file))
    else:
        print("  ERROR: No adapter weights found!")
        return

    print(f"\n  LoRA weight keys ({len(weights)}):")
    all_zero = True
    for k, v in weights.items():
        norm = v.float().norm().item()
        if norm > 0:
            all_zero = False
        print(f"    {k}: shape={list(v.shape)}, norm={norm:.6f}")

    if all_zero:
        print("\n  WARNING: All LoRA weights are ZERO! Training didn't learn anything.")
    else:
        print("\n  LoRA weights are non-zero (good).")

    # --- Compare base vs LoRA output ---
    print(f"\n{'='*50}")
    print("2. Comparing base vs LoRA output (single forward pass)")
    print(f"{'='*50}")

    # Load base model
    base_model = load_official_mdm(CHECKPOINT_DIR, device=device)
    base_model.eval()

    # Load LoRA model
    lora_model = load_official_mdm(CHECKPOINT_DIR, device="cpu")
    lora_model = replace_attention_layers(lora_model)
    from peft import PeftModel
    lora_model = PeftModel.from_pretrained(lora_model, str(lora_path))
    lora_model.to(device)
    lora_model.eval()

    # Single forward pass with same input
    torch.manual_seed(42)
    x = torch.randn(1, 263, 1, 196, device=device)
    t = torch.tensor([500], device=device)
    y = build_y_dict(["a person walking like a zombie"], [196], 196, device)

    with torch.no_grad():
        base_out = base_model(x, t, y)
        lora_out = lora_model(x, t, y)

    diff = (base_out - lora_out).abs()
    print(f"\n  Base output range:  [{base_out.min():.4f}, {base_out.max():.4f}]")
    print(f"  LoRA output range:  [{lora_out.min():.4f}, {lora_out.max():.4f}]")
    print(f"  Absolute diff:      mean={diff.mean():.6f}, max={diff.max():.6f}")
    print(f"  Relative diff:      {(diff / base_out.abs().clamp(min=1e-8)).mean():.6f}")

    if diff.max() < 1e-5:
        print("\n  RESULT: Outputs are IDENTICAL. LoRA is having NO effect!")
        print("  Possible causes:")
        print("    - LoRA weights are all zero")
        print("    - Adapter names don't match model structure")
        print("    - PeftModel loaded but adapters not active")
    elif diff.mean() < 0.01:
        print("\n  RESULT: Outputs differ slightly. LoRA effect is VERY WEAK.")
        print("  Likely cause: insufficient training or normalization mismatch.")
    else:
        print(f"\n  RESULT: LoRA is working! Mean diff = {diff.mean():.4f}")

    # --- Check normalization spaces ---
    print(f"\n{'='*50}")
    print("3. Checking normalization mismatch")
    print(f"{'='*50}")

    hml_mean = np.load("/transfer/loradataset/humanml3d/Mean.npy")
    hml_std = np.load("/transfer/loradataset/humanml3d/Std.npy")

    style_dir = Path("/transfer/loradataset/style_converted")
    if style_dir.exists():
        style_files = sorted(style_dir.glob("*.npy"))
        if style_files:
            all_style = np.concatenate([np.load(f) for f in style_files], axis=0)
            style_mean = all_style.mean(axis=0)
            style_std = all_style.std(axis=0)
            style_std[style_std < 1e-5] = 1.0

            # How different are the normalization spaces?
            mean_diff = np.abs(hml_mean - style_mean)
            std_ratio = hml_std / np.clip(style_std, 1e-5, None)

            print(f"  HumanML3D mean range:  [{hml_mean.min():.3f}, {hml_mean.max():.3f}]")
            print(f"  Style data mean range: [{style_mean.min():.3f}, {style_mean.max():.3f}]")
            print(f"  Mean difference:       avg={mean_diff.mean():.3f}, max={mean_diff.max():.3f}")
            print(f"  Std ratio (hml/style): avg={std_ratio.mean():.3f}, max={std_ratio.max():.3f}")

            # Show what style data looks like in HumanML3D normalization
            style_in_hml = (all_style - hml_mean) / np.clip(hml_std, 1e-5, None)
            print(f"\n  Style data in HumanML3D norm space: [{style_in_hml.min():.1f}, {style_in_hml.max():.1f}]")
            print(f"  (Should be roughly [-5, +5] for stable training)")

            if abs(style_in_hml.max()) > 50:
                print("\n  CONFIRMED: Style data is incompatible with HumanML3D normalization.")
                print("  This means LoRA trained in self-normalized space cannot work at inference.")
                print("  FIX: Need to fix BVH converter to produce HumanML3D-compatible features.")

    print(f"\n{'='*50}")
    print("Done.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
