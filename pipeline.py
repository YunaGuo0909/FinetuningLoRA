"""Motion Style-Adaptive LoRA Pipeline

End-to-end pipeline: style analysis -> data preparation -> LoRA training -> evaluation -> visualization

Usage:
    # Convert 100STYLE BVH data and train LoRA
    python pipeline.py --style-data data/100STYLE/Zombie --style "zombie" --run-name zombie_walk

    # Evaluate and compare base vs LoRA
    python pipeline.py --evaluate --lora-path outputs/models/zombie_walk/final --run-name zombie_walk

    # Generate and visualize
    python pipeline.py --generate --lora-path outputs/models/zombie_walk/final \
        --prompt "a person walking in zombie style" --run-name zombie_walk
"""

import argparse
import json
import subprocess
import sys
import numpy as np
import torch
from pathlib import Path

from src.data.bvh_converter import BVHToHumanML3D
from src.data.humanml_dataset import HumanML3DDataset, StyleMotionDataset
from src.models.mdm import MDM, load_pretrained_mdm
from src.models.diffusion import GaussianDiffusion
from src.evaluation.evaluator import MotionEvaluator
from src.visualization.motion_viz import (
    motion_features_to_positions,
    render_motion_animation,
    render_comparison,
)


def load_config(config_path: str = "configs/default.json") -> dict:
    with open(config_path) as f:
        return json.load(f)



def stage_convert_data(style_bvh_dir: str, output_dir: str, style_label: str):
    """Stage 1: Convert BVH files to HumanML3D format."""
    print("\n[1/4] Converting BVH data to HumanML3D format...")
    converter = BVHToHumanML3D(target_fps=20)
    metadata = converter.convert_directory(style_bvh_dir, output_dir, style_label=style_label)
    print(f"  Converted {len(metadata)} motions")
    return metadata


def stage_train_lora(
    style_data_dir: str,
    run_name: str,
    config: dict,
    pretrained_path: str = None,
):
    """Stage 2: Launch LoRA fine-tuning."""
    print("\n[2/4] Training LoRA...")
    output_dir = f"{config['output']['model_dir']}/{run_name}"

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "src/training/train_mdm_lora.py",
        f"--humanml3d_dir={config['data']['humanml3d_dir']}",
        f"--style_data_dir={style_data_dir}",
        f"--output_dir={output_dir}",
        f"--mode=lora",
        f"--lora_rank={config['lora']['rank']}",
        f"--lora_alpha={config['lora']['alpha']}",
        f"--lora_dropout={config['lora']['dropout']}",
        f"--lora_targets={','.join(config['lora']['target_modules'])}",
        f"--batch_size={config['training']['batch_size']}",
        f"--learning_rate={config['training']['learning_rate']}",
        f"--max_train_steps={config['training']['max_train_steps']}",
        f"--seed={config['training']['seed']}",
        f"--mixed_precision={config['training']['mixed_precision']}",
        f"--checkpointing_steps={config['training']['checkpointing_steps']}",
    ]

    if pretrained_path:
        cmd.append(f"--pretrained_path={pretrained_path}")

    print(f"  Command: {' '.join(cmd[:5])} ...")
    result = subprocess.run(cmd, check=True)
    print(f"  Training complete. Model saved to {output_dir}")
    return Path(output_dir) / "final"


def stage_generate(
    model: MDM,
    diffusion: GaussianDiffusion,
    prompts: list[str],
    clip_encoder,
    config: dict,
    output_dir: str,
    num_samples: int = 4,
    motion_length: int = 196,
) -> np.ndarray:
    """Stage 3: Generate motions with the model."""
    import open_clip

    device = next(model.parameters()).device
    model.eval()

    all_motions = []

    with torch.no_grad():
        for prompt in prompts:
            text_emb = clip_encoder.encode([prompt] * num_samples)
            shape = (num_samples, motion_length, config["data"]["nfeats"])

            motions = diffusion.ddim_sample(
                model, shape, text_emb=text_emb, device=device, num_steps=50,
            )
            all_motions.append(motions.cpu().numpy())

    all_motions = np.concatenate(all_motions, axis=0)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "generated_motions.npy", all_motions)
    print(f"  Generated {all_motions.shape[0]} motions, saved to {out}")

    return all_motions


def stage_evaluate_and_visualize(
    base_motions: np.ndarray,
    lora_motions: np.ndarray,
    config: dict,
    run_name: str,
    reference_motions: np.ndarray = None,
):
    """Stage 4: Evaluate metrics and render visualizations."""
    print("\n[4/4] Evaluating and visualizing...")

    mean = np.load(config["data"]["mean_path"])
    std = np.load(config["data"]["std_path"])

    evaluator = MotionEvaluator(mean, std)
    results = evaluator.compare_base_vs_lora(base_motions, lora_motions, reference_motions)

    eval_dir = f"{config['output']['eval_dir']}/{run_name}"
    evaluator.save_results(results, f"{eval_dir}/evaluation_results.json")

    # Visualize first sample from each
    viz_dir = f"{config['output']['viz_dir']}/{run_name}"

    base_denorm = base_motions[0] * std + mean
    lora_denorm = lora_motions[0] * std + mean

    base_pos = motion_features_to_positions(base_denorm)
    lora_pos = motion_features_to_positions(lora_denorm)

    render_motion_animation(base_pos, f"{viz_dir}/base_sample.gif", title="Base MDM")
    render_motion_animation(lora_pos, f"{viz_dir}/lora_sample.gif", title="LoRA MDM")
    render_comparison(base_pos, lora_pos, f"{viz_dir}/comparison.gif")

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Evaluation Results: {run_name}")
    print(f"{'=' * 50}")
    for model_name in ["base_model", "with_lora"]:
        print(f"\n  {model_name}:")
        for k, v in results[model_name].items():
            print(f"    {k}: {v:.4f}")
    print(f"\n  Improvements:")
    for k, v in results["improvement"].items():
        direction = "lower is better" if k in ["fid", "jitter_mean"] else "higher is better"
        print(f"    {k}: {v:+.4f} ({direction})")

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Style-Adaptive LoRA Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--run-name", type=str, required=True)

    # Stage 1: Convert
    parser.add_argument("--style-data", type=str, help="Path to BVH directory for conversion")
    parser.add_argument("--style", type=str, default="", help="Style label")

    # Stage 2: Train
    parser.add_argument("--train", action="store_true", help="Run LoRA training")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained MDM weights")

    # Stage 3: Generate
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", type=str, nargs="+", help="Generation prompts")
    parser.add_argument("--lora-path", type=str, help="Path to trained LoRA weights")

    # Stage 4: Evaluate
    parser.add_argument("--evaluate", action="store_true")

    # Full pipeline
    parser.add_argument("--full", action="store_true", help="Run all stages")

    args = parser.parse_args()
    config = load_config(args.config)

    style_data_dir = f"{config['data']['style_converted_dir']}/{args.run_name}"

    if args.full or args.style_data:
        if args.style_data:
            # Skip if already converted
            meta_file = Path(style_data_dir) / "metadata.jsonl"
            if meta_file.exists():
                count = sum(1 for _ in open(meta_file))
                print(f"\n[1/4] Style data already converted ({count} motions), skipping.")
            else:
                stage_convert_data(args.style_data, style_data_dir, args.style)

    if args.full or args.train:
        lora_final = Path(config["output"]["model_dir"]) / args.run_name / "final"
        if lora_final.exists() and any(lora_final.iterdir()):
            print(f"\n[2/4] LoRA weights already exist at {lora_final}, skipping training.")
        else:
            pretrained = args.pretrained or config.get("pretrained_weights")
            stage_train_lora(style_data_dir, args.run_name, config, pretrained)

    if args.full or args.generate or args.evaluate:
        from src.training.train_mdm_lora import CLIPTextEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_encoder = CLIPTextEncoder(config["model"]["clip_model"], device=device)
        diffusion = GaussianDiffusion(
            config["diffusion"]["num_timesteps"],
            config["diffusion"]["beta_schedule"],
        )

        prompts = args.prompt or [
            f"a person walking in {args.style} style",
            f"a person running in {args.style} style",
            f"a person standing in {args.style} style",
        ]

        gen_dir = f"{config['output']['generation_dir']}/{args.run_name}"

        # Generate with base model
        print("\n[3/4] Generating motions...")
        base_model = MDM(**{k: v for k, v in config["model"].items()
                           if k in ["latent_dim", "ff_size", "num_layers", "num_heads",
                                    "dropout", "clip_dim", "cond_mode"]},
                         nfeats=config["data"]["nfeats"],
                         max_seq_len=config["data"]["max_motion_length"])

        pretrained = args.pretrained or config.get("pretrained_weights")
        if pretrained and Path(pretrained).exists():
            load_pretrained_mdm(base_model, pretrained)
        base_model = base_model.to(device).eval()

        print("  Generating base model samples...")
        base_motions = stage_generate(
            base_model, diffusion, prompts, clip_encoder, config,
            f"{gen_dir}/base", num_samples=4,
        )

        # Generate with LoRA
        lora_path = args.lora_path or f"{config['output']['model_dir']}/{args.run_name}/final"
        if Path(lora_path).exists():
            from peft import PeftModel
            lora_model = MDM(**{k: v for k, v in config["model"].items()
                               if k in ["latent_dim", "ff_size", "num_layers", "num_heads",
                                        "dropout", "clip_dim", "cond_mode"]},
                             nfeats=config["data"]["nfeats"],
                             max_seq_len=config["data"]["max_motion_length"])
            if pretrained and Path(pretrained).exists():
                load_pretrained_mdm(lora_model, pretrained)
            lora_model = PeftModel.from_pretrained(lora_model, lora_path)
            lora_model = lora_model.to(device).eval()

            print("  Generating LoRA model samples...")
            lora_motions = stage_generate(
                lora_model, diffusion, prompts, clip_encoder, config,
                f"{gen_dir}/lora", num_samples=4,
            )

            # Evaluate
            stage_evaluate_and_visualize(base_motions, lora_motions, config, args.run_name)
        else:
            print(f"  LoRA weights not found at {lora_path}, skipping comparison")
