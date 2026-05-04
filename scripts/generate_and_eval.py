"""Generate motions with base MDM vs LoRA MDM, evaluate and visualize."""

from __future__ import annotations

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mdm import MDM, load_pretrained_mdm
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

PRETRAINED_PATH = "/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512/model000475000.pt"
LORA_PATH = "/transfer/outputs/models/style_lora_v2/final"
STYLE_DATA_DIR = "/transfer/loradataset/style_converted"
HML3D_DIR = "/transfer/loradataset/humanml3d"
OUTPUT_DIR = "/transfer/outputs/eval/style_lora_v2"

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
# CLIP encoder
# ---------------------------------------------------------------------------

class CLIPTextEncoder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        import open_clip
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(tokens).float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(device):
    model = MDM(
        nfeats=263, latent_dim=512, ff_size=1024,
        num_layers=8, num_heads=4, dropout=0.1,
        clip_dim=512, cond_mode="text", max_seq_len=MOTION_LENGTH,
    )
    load_pretrained_mdm(model, PRETRAINED_PATH)
    return model.to(device)


@torch.no_grad()
def generate(model, diffusion, clip_encoder, prompts, device):
    model.eval()
    all_motions = []
    for prompt in prompts:
        text_emb = clip_encoder.encode([prompt] * NUM_SAMPLES)
        shape = (NUM_SAMPLES, MOTION_LENGTH, 263)
        motions = diffusion.ddim_sample(model, shape, text_emb=text_emb, device=device, num_steps=DDIM_STEPS)
        all_motions.append(motions.cpu().numpy())
    return np.concatenate(all_motions, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP encoder...")
    clip_encoder = CLIPTextEncoder(device=device)

    diffusion = GaussianDiffusion(1000, "cosine")

    # --- Base model generation ---
    print("\n" + "=" * 50)
    print("Generating with BASE model...")
    print("=" * 50)
    base_model = build_model(device)
    torch.manual_seed(SEED)
    base_motions = generate(base_model, diffusion, clip_encoder, PROMPTS, device)
    np.save(out_dir / "base_motions.npy", base_motions)
    print(f"  Generated {base_motions.shape[0]} motions, shape={base_motions.shape}")
    del base_model
    torch.cuda.empty_cache()

    # --- LoRA model generation ---
    print("\n" + "=" * 50)
    print("Generating with LORA model...")
    print("=" * 50)
    lora_model = build_model(device)
    from peft import PeftModel
    lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)
    lora_model = lora_model.to(device)
    torch.manual_seed(SEED)
    lora_motions = generate(lora_model, diffusion, clip_encoder, PROMPTS, device)
    np.save(out_dir / "lora_motions.npy", lora_motions)
    print(f"  Generated {lora_motions.shape[0]} motions, shape={lora_motions.shape}")
    del lora_model
    torch.cuda.empty_cache()

    # --- Evaluation ---
    print("\n" + "=" * 50)
    print("Evaluating...")
    print("=" * 50)

    # Load style data as reference
    style_motions = []
    motions_dir = Path(STYLE_DATA_DIR) / "motions"
    for f in sorted(motions_dir.glob("*.npy")):
        style_motions.append(np.load(f)[:MOTION_LENGTH])
    # Pad to same length
    style_padded = []
    for m in style_motions:
        if m.shape[0] < MOTION_LENGTH:
            m = np.concatenate([m, np.zeros((MOTION_LENGTH - m.shape[0], 263))], axis=0)
        style_padded.append(m[:MOTION_LENGTH])
    reference = np.stack(style_padded)

    mean = np.load(Path(HML3D_DIR) / "Mean.npy")
    std = np.load(Path(HML3D_DIR) / "Std.npy")
    evaluator = MotionEvaluator(mean, std)

    results = evaluator.compare_base_vs_lora(base_motions, lora_motions, reference)

    # Save results
    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 50}")
    print("Results")
    print(f"{'=' * 50}")
    for model_name in ["base_model", "with_lora"]:
        print(f"\n  {model_name}:")
        for k, v in results[model_name].items():
            print(f"    {k}: {v:.4f}")
    print(f"\n  Improvement (lora - base):")
    for k, v in results["improvement"].items():
        direction = "lower is better" if k in ["fid", "jitter_mean"] else ""
        print(f"    {k}: {v:+.4f}  {direction}")

    # --- Visualization ---
    print("\n" + "=" * 50)
    print("Rendering visualizations...")
    print("=" * 50)

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    for i, prompt in enumerate(PROMPTS):
        idx = i * NUM_SAMPLES  # first sample for each prompt
        prompt_short = prompt.replace(" ", "_")[:30]

        # Denormalize (approximate - use zeros as mean since these are model outputs)
        base_pos = motion_features_to_positions(base_motions[idx])
        lora_pos = motion_features_to_positions(lora_motions[idx])

        # Individual animations
        render_motion_animation(
            base_pos, str(viz_dir / f"{i}_base_{prompt_short}.gif"),
            title=f"Base: {prompt[:40]}", fps=20,
        )
        render_motion_animation(
            lora_pos, str(viz_dir / f"{i}_lora_{prompt_short}.gif"),
            title=f"LoRA: {prompt[:40]}", fps=20,
        )

        # Side-by-side comparison
        render_comparison(
            base_pos, lora_pos,
            str(viz_dir / f"{i}_comparison_{prompt_short}.gif"),
            title=f"{prompt[:40]}", fps=20,
        )

        print(f"  Rendered: {prompt[:50]}")

    # Save static frame sequences for the first prompt
    save_frame_sequence(
        motion_features_to_positions(base_motions[0]),
        str(viz_dir / "frames_base"), step=10, title="Base",
    )
    save_frame_sequence(
        motion_features_to_positions(lora_motions[0]),
        str(viz_dir / "frames_lora"), step=10, title="LoRA",
    )

    print(f"\nAll outputs saved to {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
