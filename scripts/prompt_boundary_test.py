"""Experiment: test LoRA style transfer effectiveness across prompt categories.

Hypothesis: LoRA trained on locomotion data (100STYLE) applies style well to
walking prompts but weakens or fails on non-locomotion prompts (sitting, lying,
dancing) because the adapter's locomotion prior conflicts with the prompt semantics.

Output:
    outputs/prompt_boundary/
        results.json            per-prompt style-shift scores
        viz/                    comparison GIFs

Usage:
    python scripts/prompt_boundary_test.py
    python scripts/prompt_boundary_test.py --version v6 --styles zombie drunk robot
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mdm_official import (
    load_official_mdm,
    replace_attention_layers,
    motion_to_mdm_input,
    mdm_output_to_motion,
    build_y_dict,
)
from src.models.diffusion import GaussianDiffusion
from src.visualization.motion_viz import (
    motion_features_to_positions,
    render_comparison,
)

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
HML3D_DIR      = "/transfer/loradataset/humanml3d"
MODEL_BASE     = "/transfer/loraoutputs/models"
OUTPUT_DIR     = "/transfer/loraoutputs/eval/prompt_boundary"

PROMPTS = {
    "locomotion": [
        "a person walking forward",
        "a person walking in a circle",
        "a person jogging",
        "a person walking slowly",
        "a person marching",
    ],
    "non_locomotion": [
        "a person sitting down",
        "a person lying on the ground",
        "a person dancing",
        "a person waving their hands",
        "a person doing a squat",
    ],
}

NUM_SAMPLES  = 4
MOTION_LEN   = 196
DDIM_STEPS   = 50
SEED         = 42


# ── Generation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, diffusion, prompts: list[str], device: str) -> np.ndarray:
    model.eval()
    all_motions = []
    for prompt in prompts:
        x = torch.randn((NUM_SAMPLES, 263, 1, MOTION_LEN), device=device)
        y = build_y_dict([prompt] * NUM_SAMPLES, [MOTION_LEN] * NUM_SAMPLES,
                         MOTION_LEN, device)
        step_size = diffusion.num_timesteps // DDIM_STEPS
        timesteps = list(reversed(range(0, diffusion.num_timesteps, step_size)))
        for i, t_val in enumerate(timesteps):
            t = torch.full((NUM_SAMPLES,), t_val, device=device, dtype=torch.long)
            x0_pred = model(x, t, y)
            ab = diffusion.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
            noise = (x - ab.sqrt() * x0_pred) / (1 - ab).clamp(min=1e-8).sqrt()
            next_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            ab_prev = diffusion.alphas_cumprod.to(device)[next_t].view(1, 1, 1, 1)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * noise
        all_motions.append(mdm_output_to_motion(x).cpu().numpy())
    return np.concatenate(all_motions, axis=0)  # (N*num_prompts, T, 263)


# ── Style shift score ─────────────────────────────────────────────────────────
def style_shift(base: np.ndarray, lora: np.ndarray) -> dict:
    """Measure how much LoRA shifts the output relative to base.

    Returns:
        feature_l2:  mean L2 distance between base and lora in feature space
        diversity_base / diversity_lora: pairwise distances within each set
        jitter_base / jitter_lora: mean acceleration magnitude
    """
    flat_base = base.reshape(len(base), -1)
    flat_lora = lora.reshape(len(lora), -1)
    feature_l2 = float(np.linalg.norm(flat_base - flat_lora, axis=1).mean())

    def diversity(m):
        f = m.reshape(len(m), -1)
        n = len(f)
        dists = [np.linalg.norm(f[i] - f[j])
                 for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(dists)) if dists else 0.0

    def jitter(m):
        # m: (N, T, 263) — use joint positions (dims 4:67)
        pos = m[:, :, 4:67].reshape(len(m), -1, 21, 3)
        vel = np.diff(pos, axis=1) * 20
        acc = np.diff(vel, axis=1) * 20
        return float(np.linalg.norm(acc, axis=-1).mean())

    return {
        "feature_l2":     feature_l2,
        "diversity_base": diversity(base),
        "diversity_lora": diversity(lora),
        "jitter_base":    jitter(base),
        "jitter_lora":    jitter(lora),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v6",
                   help="LoRA version suffix (v1/v6 etc.)")
    p.add_argument("--styles", nargs="+",
                   default=["zombie", "drunk", "robot"],
                   help="Which styles to test")
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    suffix = "" if args.version == "v1" else f"_{args.version}"

    out_dir = Path(args.output_dir)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    mean = np.load(Path(HML3D_DIR) / "Mean.npy")
    std  = np.load(Path(HML3D_DIR) / "Std.npy")
    std_safe = std.copy()
    std_safe[std_safe < 1e-5] = 1.0

    diffusion = GaussianDiffusion(1000, "cosine")
    all_prompts = PROMPTS["locomotion"] + PROMPTS["non_locomotion"]

    # Base model
    print("Generating: base model")
    base_model = load_official_mdm(CHECKPOINT_DIR, device=device)
    base_model.eval()
    torch.manual_seed(SEED)
    base_motions = generate(base_model, diffusion, all_prompts, device)
    del base_model
    torch.cuda.empty_cache()

    results = {}

    for style in args.styles:
        lora_path = Path(MODEL_BASE) / f"lora_bvh_{style}{suffix}" / "final"
        if not lora_path.exists():
            print(f"Skip {style}: {lora_path} not found")
            continue

        print(f"Generating: {style}")
        lora_model = load_official_mdm(CHECKPOINT_DIR, device="cpu")
        lora_model = replace_attention_layers(lora_model)
        lora_model = PeftModel.from_pretrained(lora_model, str(lora_path))
        lora_model = lora_model.to(device).eval()
        torch.manual_seed(SEED)
        lora_motions = generate(lora_model, diffusion, all_prompts, device)
        del lora_model
        torch.cuda.empty_cache()

        results[style] = {}
        n_loco = len(PROMPTS["locomotion"]) * NUM_SAMPLES
        n_nonl = len(PROMPTS["non_locomotion"]) * NUM_SAMPLES

        for category, base_idx, lora_idx in [
            ("locomotion",     slice(0, n_loco),          slice(0, n_loco)),
            ("non_locomotion", slice(n_loco, n_loco + n_nonl), slice(n_loco, n_loco + n_nonl)),
        ]:
            results[style][category] = style_shift(
                base_motions[base_idx], lora_motions[lora_idx]
            )

        # Render one comparison per category (first prompt, first sample)
        base_denorm = base_motions * std_safe + mean
        lora_denorm = lora_motions * std_safe + mean

        for cat_name, prompt_idx in [("locomotion", 0), ("non_locomotion", 5)]:
            idx = prompt_idx * NUM_SAMPLES
            base_pos = motion_features_to_positions(base_denorm[idx])
            lora_pos = motion_features_to_positions(lora_denorm[idx])
            prompt_slug = all_prompts[prompt_idx].replace(" ", "_")[:30]
            render_comparison(
                base_pos, lora_pos,
                str(viz_dir / f"{style}_{cat_name}_{prompt_slug}.gif"),
                title=f"{style} | {all_prompts[prompt_idx][:40]}", fps=20,
            )

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n── Style shift: feature L2 (higher = more style effect) ──")
    print(f"{'Style':<15} {'Locomotion':>12} {'Non-Loco':>12} {'Ratio':>8}")
    print("-" * 50)
    for style, cats in results.items():
        loco = cats.get("locomotion", {}).get("feature_l2", 0)
        nonl = cats.get("non_locomotion", {}).get("feature_l2", 0)
        ratio = loco / nonl if nonl > 0 else float("inf")
        print(f"{style:<15} {loco:>12.2f} {nonl:>12.2f} {ratio:>8.2f}x")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
