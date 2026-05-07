"""Generate motions with base MDM vs multiple LoRA MDMs, evaluate and visualize.

Supports comparing multiple single-style LoRAs against the base model.
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
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
HML3D_DIR = "/transfer/loradataset/humanml3d"
OUTPUT_DIR = f"/transfer/loraoutputs/eval/multi_style{'_v2' if _LORA_VER == 'v2' else ''}"

# LoRA models to compare (100STYLE BVH data)
# Set LORA_VERSION="v2" env var to use v2 models (with foot velocity penalty)
import os
_LORA_VER = os.environ.get("LORA_VERSION", "v1")
_SUFFIX = "_v2" if _LORA_VER == "v2" else ""
LORA_MODELS = {
    "zombie":    f"/transfer/loraoutputs/models/lora_bvh_zombie{_SUFFIX}/final",
    "elated":    f"/transfer/loraoutputs/models/lora_bvh_elated{_SUFFIX}/final",
    "old":       f"/transfer/loraoutputs/models/lora_bvh_old{_SUFFIX}/final",
    "depressed": f"/transfer/loraoutputs/models/lora_bvh_depressed{_SUFFIX}/final",
    "drunk":     f"/transfer/loraoutputs/models/lora_bvh_drunk{_SUFFIX}/final",
    "mixed":     f"/transfer/loraoutputs/models/lora_bvh_mixed{_SUFFIX}/final",
}

PROMPTS = [
    "a person walking forward",
    "a person walking in a circle",
    "a person stepping sideways",
    "a person turning around",
    "a person walking and then stopping",
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
    """Generate motions using the official MDM model with DDIM sampling."""
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
            x_0_pred = model(x, t, y)

            alpha_bar = diffusion.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
            pred_noise = (x - alpha_bar.sqrt() * x_0_pred) / (1 - alpha_bar).clamp(min=1e-8).sqrt()

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
    print("\n" + "=" * 60)
    print("Generating with BASE model...")
    print("=" * 60)
    base_model = load_official_mdm(CHECKPOINT_DIR, device=device)
    base_model.eval()

    torch.manual_seed(SEED)
    base_motions = generate(base_model, diffusion, PROMPTS, device, NUM_SAMPLES, MOTION_LENGTH)
    np.save(out_dir / "base_motions.npy", base_motions)
    print(f"  Generated {base_motions.shape[0]} motions, range: [{base_motions.min():.3f}, {base_motions.max():.3f}]")
    del base_model
    torch.cuda.empty_cache()

    # --- LoRA models generation ---
    lora_results = {}
    for style_name, lora_path in LORA_MODELS.items():
        lora_path = Path(lora_path)
        if not lora_path.exists():
            print(f"\n  SKIP {style_name}: {lora_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Generating with LoRA: {style_name}")
        print("=" * 60)

        lora_model = load_official_mdm(CHECKPOINT_DIR, device="cpu")
        lora_model = replace_attention_layers(lora_model)
        from peft import PeftModel
        lora_model = PeftModel.from_pretrained(lora_model, str(lora_path))
        lora_model = lora_model.to(device).eval()

        torch.manual_seed(SEED)
        motions = generate(lora_model, diffusion, PROMPTS, device, NUM_SAMPLES, MOTION_LENGTH)
        np.save(out_dir / f"lora_{style_name}_motions.npy", motions)
        print(f"  Generated {motions.shape[0]} motions, range: [{motions.min():.3f}, {motions.max():.3f}]")

        lora_results[style_name] = motions
        del lora_model
        torch.cuda.empty_cache()

    if not lora_results:
        print("\nNo LoRA models found! Train first with: bash run_train.sh")
        return

    # --- Evaluation ---
    print(f"\n{'=' * 60}")
    print("Evaluating...")
    print("=" * 60)

    evaluator = MotionEvaluator(mean, std)
    all_eval = {"base_model": evaluator.evaluate_batch(base_motions)}

    for style_name, motions in lora_results.items():
        results = evaluator.compare_base_vs_lora(base_motions, motions)
        all_eval[f"lora_{style_name}"] = results["with_lora"]
        print(f"\n  {style_name}:")
        for k, v in results["with_lora"].items():
            print(f"    {k}: {v:.4f}")

    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(all_eval, f, indent=2)

    # --- Visualization ---
    print(f"\n{'=' * 60}")
    print("Rendering visualizations...")
    print("=" * 60)

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    # Denormalize
    base_denorm = base_motions * std_safe + mean

    lora_denorms = {}
    for style_name, motions in lora_results.items():
        lora_denorms[style_name] = motions * std_safe + mean

    # Render per prompt: base + each LoRA
    for i, prompt in enumerate(PROMPTS):
        idx = i * NUM_SAMPLES  # first sample of each prompt
        prompt_short = prompt.replace(" ", "_")[:30]

        base_pos = motion_features_to_positions(base_denorm[idx])
        render_motion_animation(
            base_pos, str(viz_dir / f"{i}_base_{prompt_short}.gif"),
            title=f"Base: {prompt[:40]}", fps=20,
        )

        for style_name, denorm in lora_denorms.items():
            lora_pos = motion_features_to_positions(denorm[idx])

            render_motion_animation(
                lora_pos, str(viz_dir / f"{i}_lora_{style_name}_{prompt_short}.gif"),
                title=f"LoRA-{style_name}: {prompt[:40]}", fps=20,
            )
            render_comparison(
                base_pos, lora_pos,
                str(viz_dir / f"{i}_cmp_{style_name}_{prompt_short}.gif"),
                title=f"Base vs {style_name}: {prompt[:30]}", fps=20,
            )

        print(f"  Rendered: {prompt[:50]}")

    # Reference HumanML3D motion
    hml_file = sorted(Path(HML3D_DIR).joinpath("new_joint_vecs").glob("*.npy"))[0]
    hml_motion = np.load(hml_file)[:MOTION_LENGTH]
    hml_pos = motion_features_to_positions(hml_motion)
    render_motion_animation(hml_pos, str(viz_dir / "reference_humanml3d.gif"),
                            title="Reference (HumanML3D)", fps=20)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
