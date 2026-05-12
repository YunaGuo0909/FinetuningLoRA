"""Generate, evaluate and visualise base MDM vs LoRA outputs."""

from __future__ import annotations

import os
import sys
import json
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
from src.evaluation.evaluator import MotionEvaluator
from src.visualization.motion_viz import (
    motion_features_to_positions,
    render_motion_animation,
    render_comparison,
)

CHECKPOINT_DIR = "/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
HML3D_DIR     = "/transfer/loradataset/humanml3d"

# Use LORA_VERSION env var to select model version (v1..v5)
_LORA_VER = os.environ.get("LORA_VERSION", "v1")
_SUFFIX = {"v1": "", "v2": "_v2", "v3": "_v3", "v4": "_v4", "v5": "_v5", "v6": "_v6"}.get(_LORA_VER, "")

OUTPUT_DIR = f"/transfer/loraoutputs/eval/multi_style{_SUFFIX}"

# All styles that may have been trained. Missing directories are skipped at runtime.
_ALL_STYLES = [
    "zombie", "elated", "old", "depressed", "drunk",
    "angry", "chicken", "proud",
    "heavyset", "bigsteps", "stiff", "duckfoot",
    "highknees", "flapping", "punch", "wildarms",
    "handsinpockets", "handsbetweenlegs", "onphoneleft",
    "penguin", "robot",
]
LORA_MODELS = {
    s: f"/transfer/loraoutputs/models/lora_bvh_{s}{_SUFFIX}/final"
    for s in _ALL_STYLES
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

    print("Generating: base model")
    base_model = load_official_mdm(CHECKPOINT_DIR, device=device)
    base_model.eval()
    torch.manual_seed(SEED)
    base_motions = generate(base_model, diffusion, PROMPTS, device, NUM_SAMPLES, MOTION_LENGTH)
    np.save(out_dir / "base_motions.npy", base_motions)
    del base_model
    torch.cuda.empty_cache()

    lora_results = {}
    for style_name, lora_path in LORA_MODELS.items():
        lora_path = Path(lora_path)
        if not lora_path.exists():
            continue
        print(f"Generating: {style_name}")
        lora_model = load_official_mdm(CHECKPOINT_DIR, device="cpu")
        lora_model = replace_attention_layers(lora_model)
        lora_model = PeftModel.from_pretrained(lora_model, str(lora_path))
        lora_model = lora_model.to(device).eval()
        torch.manual_seed(SEED)
        motions = generate(lora_model, diffusion, PROMPTS, device, NUM_SAMPLES, MOTION_LENGTH)
        np.save(out_dir / f"lora_{style_name}_motions.npy", motions)
        lora_results[style_name] = motions
        del lora_model
        torch.cuda.empty_cache()

    if not lora_results:
        print("No LoRA models found.")
        return

    print("Evaluating...")
    evaluator = MotionEvaluator(mean, std)
    all_eval = {"base_model": evaluator.evaluate_batch(base_motions)}
    for style_name, motions in lora_results.items():
        results = evaluator.compare_base_vs_lora(base_motions, motions)
        all_eval[f"lora_{style_name}"] = results["with_lora"]
        print(f"  {style_name}: " + "  ".join(f"{k}={v:.4f}" for k, v in results["with_lora"].items()))

    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(all_eval, f, indent=2)

    print("Rendering...")
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    base_denorm = base_motions * std_safe + mean
    lora_denorms = {s: m * std_safe + mean for s, m in lora_results.items()}

    for i, prompt in enumerate(PROMPTS):
        idx = i * NUM_SAMPLES
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
                title=f"{style_name}: {prompt[:40]}", fps=20,
            )
            render_comparison(
                base_pos, lora_pos,
                str(viz_dir / f"{i}_cmp_{style_name}_{prompt_short}.gif"),
                title=f"Base vs {style_name}: {prompt[:30]}", fps=20,
            )

    hml_file = sorted(Path(HML3D_DIR).joinpath("new_joint_vecs").glob("*.npy"))[0]
    hml_pos = motion_features_to_positions(np.load(hml_file)[:MOTION_LENGTH])
    render_motion_animation(hml_pos, str(viz_dir / "reference_humanml3d.gif"),
                            title="Reference (HumanML3D)", fps=20)

    print(f"Done. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
