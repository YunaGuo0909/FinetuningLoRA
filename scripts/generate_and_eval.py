"""Generate motions with base MDM vs LoRA MDM, evaluate and visualize.

Uses the official MDM model for guaranteed correct generation.
"""

from __future__ import annotations

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mdm_official import (
    load_official_mdm,
    replace_attention_layers,
    motion_to_mdm_input,
    mdm_output_to_motion,
    build_y_dict,
)
from src.models.diffusion import GaussianDiffusion
from src.evaluation.evaluator import MotionEvaluator
from src.visualization.motion_viz import (
    motion_features_to_positions,
    render_motion_animation,
    render_comparison,
    save_frame_sequence,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
LORA_PATH = "/transfer/loraoutputs/models/style_lora_v3/final"
STYLE_DATA_DIR = "/transfer/loradataset/style_converted"
HML3D_DIR = "/transfer/loradataset/humanml3d"
OUTPUT_DIR = "/transfer/loraoutputs/eval/style_lora_v3"

PROMPTS = [
    "a person walking like a zombie",
    "a person walking happily with energy",
    "a person walking slowly like an old person",
    "a person walking like a robot",
    "a person stumbling around drunk",
]

NUM_SAMPLES = 4
MOTION_LENGTH = 196
DDIM_STEPS = 50
SEED = 42


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, diffusion, prompts, device, num_samples=4, motion_length=196):
    """Generate motions using the official MDM model.

    Official MDM predicts x_0 (predict_xstart=True), not noise.
    Uses DDIM sampling adapted for x_0 prediction.
    """
    model.eval()
    all_motions = []

    for prompt in prompts:
        shape = (num_samples, 263, 1, motion_length)
        x = torch.randn(shape, device=device)

        captions = [prompt] * num_samples
        lengths = [motion_length] * num_samples
        y = build_y_dict(captions, lengths, motion_length, device)

        # DDIM sampling for x_0 prediction model
        step_size = diffusion.num_timesteps // DDIM_STEPS
        timesteps = list(range(0, diffusion.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        for i, t_val in enumerate(timesteps):
            t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)

            # Model predicts x_0 directly
            x_0_pred = model(x, t, y)

            # Derive predicted noise from x_0 prediction
            alpha_bar = diffusion.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
            pred_noise = (x - alpha_bar.sqrt() * x_0_pred) / (1 - alpha_bar).clamp(min=1e-8).sqrt()

            # DDIM step
            next_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_bar_prev = diffusion.alphas_cumprod.to(device)[next_t].view(1, 1, 1, 1)

            dir_xt = (1 - alpha_bar_prev).sqrt() * pred_noise
            x = alpha_bar_prev.sqrt() * x_0_pred + dir_xt

        motions = mdm_output_to_motion(x)
        all_motions.append(motions.cpu().numpy())

    return np.concatenate(all_motions, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = np.load(Path(HML3D_DIR) / "Mean.npy")
    std = np.load(Path(HML3D_DIR) / "Std.npy")
    std_safe = std.copy()
    std_safe[std_safe < 1e-5] = 1.0

    diffusion = GaussianDiffusion(1000, "cosine")

    # --- Base model generation ---
    print("\n" + "=" * 50)
    print("Generating with BASE model (official MDM)...")
    print("=" * 50)
    base_model = load_official_mdm(CHECKPOINT_DIR, device=device)
    base_model.eval()

    torch.manual_seed(SEED)
    base_motions = generate(base_model, diffusion, PROMPTS, device,
                            NUM_SAMPLES, MOTION_LENGTH)
    np.save(out_dir / "base_motions.npy", base_motions)
    print(f"  Generated {base_motions.shape[0]} motions")
    print(f"  Value range: [{base_motions.min():.3f}, {base_motions.max():.3f}]")
    del base_model
    torch.cuda.empty_cache()

    # --- LoRA model generation ---
    lora_path = Path(LORA_PATH)
    if lora_path.exists():
        print("\n" + "=" * 50)
        print("Generating with LORA model...")
        print("=" * 50)
        lora_model = load_official_mdm(CHECKPOINT_DIR, device="cpu")
        lora_model = replace_attention_layers(lora_model)
        from peft import PeftModel
        lora_model = PeftModel.from_pretrained(lora_model, str(lora_path))
        lora_model = lora_model.to(device).eval()

        torch.manual_seed(SEED)
        lora_motions = generate(lora_model, diffusion, PROMPTS, device,
                                NUM_SAMPLES, MOTION_LENGTH)
        np.save(out_dir / "lora_motions.npy", lora_motions)
        print(f"  Generated {lora_motions.shape[0]} motions")
        print(f"  Value range: [{lora_motions.min():.3f}, {lora_motions.max():.3f}]")
        del lora_model
        torch.cuda.empty_cache()
    else:
        print(f"\n  LoRA weights not found at {lora_path}, generating base-only.")
        lora_motions = base_motions

    # --- Evaluation ---
    print("\n" + "=" * 50)
    print("Evaluating...")
    print("=" * 50)

    evaluator = MotionEvaluator(mean, std)
    results = evaluator.compare_base_vs_lora(base_motions, lora_motions)

    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 50}")
    print("Results")
    print(f"{'=' * 50}")
    for model_name in ["base_model", "with_lora"]:
        print(f"\n  {model_name}:")
        for k, v in results[model_name].items():
            print(f"    {k}: {v:.4f}")

    # --- Visualization ---
    print("\n" + "=" * 50)
    print("Rendering visualizations...")
    print("=" * 50)

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    # Denormalize
    base_denorm = base_motions * std_safe + mean
    lora_denorm = lora_motions * std_safe + mean

    # Debug ranges
    base_pos_sample = motion_features_to_positions(base_denorm[0])
    lora_pos_sample = motion_features_to_positions(lora_denorm[0])
    print(f"  Base positions: [{base_pos_sample.min():.3f}, {base_pos_sample.max():.3f}]")
    print(f"  LoRA positions: [{lora_pos_sample.min():.3f}, {lora_pos_sample.max():.3f}]")

    # Also render a real HumanML3D motion for sanity check
    hml_file = sorted(Path(HML3D_DIR).joinpath("new_joint_vecs").glob("*.npy"))[0]
    hml_motion = np.load(hml_file)[:MOTION_LENGTH]
    hml_pos = motion_features_to_positions(hml_motion)
    print(f"  HumanML3D ref:  [{hml_pos.min():.3f}, {hml_pos.max():.3f}]")
    render_motion_animation(hml_pos, str(viz_dir / "reference_humanml3d.gif"),
                            title="Reference (HumanML3D)", fps=20)

    for i, prompt in enumerate(PROMPTS):
        idx = i * NUM_SAMPLES
        prompt_short = prompt.replace(" ", "_")[:30]

        base_pos = motion_features_to_positions(base_denorm[idx])
        lora_pos = motion_features_to_positions(lora_denorm[idx])

        render_motion_animation(
            base_pos, str(viz_dir / f"{i}_base_{prompt_short}.gif"),
            title=f"Base: {prompt[:40]}", fps=20,
        )
        render_motion_animation(
            lora_pos, str(viz_dir / f"{i}_lora_{prompt_short}.gif"),
            title=f"LoRA: {prompt[:40]}", fps=20,
        )
        render_comparison(
            base_pos, lora_pos,
            str(viz_dir / f"{i}_comparison_{prompt_short}.gif"),
            title=f"{prompt[:40]}", fps=20,
        )
        print(f"  Rendered: {prompt[:50]}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
