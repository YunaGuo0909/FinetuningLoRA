"""Automated LoRA training on Stable Diffusion using diffusers."""

import subprocess
import json
import math
from pathlib import Path


class LoRATrainer:
    """Configures and launches LoRA fine-tuning automatically."""

    # Sensible defaults for different dataset sizes
    PRESETS = {
        "small": {"max_train_steps": 1000, "learning_rate": 1e-4, "rank": 8},    # <20 images
        "medium": {"max_train_steps": 2000, "learning_rate": 5e-5, "rank": 16},   # 20-100 images
        "large": {"max_train_steps": 4000, "learning_rate": 3e-5, "rank": 32},    # >100 images
    }

    def __init__(self, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 output_dir: str = "outputs/models"):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def auto_config(self, num_images: int) -> dict:
        """Automatically select training parameters based on dataset size."""
        if num_images < 20:
            preset = self.PRESETS["small"]
        elif num_images < 100:
            preset = self.PRESETS["medium"]
        else:
            preset = self.PRESETS["large"]

        config = {
            "pretrained_model_name_or_path": self.base_model,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_train_steps": preset["max_train_steps"],
            "learning_rate": preset["learning_rate"],
            "rank": preset["rank"],
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 100,
            "resolution": 1024 if "xl" in self.base_model.lower() else 512,
            "mixed_precision": "fp16",
            "seed": 42,
            "checkpointing_steps": 500,
        }
        return config

    def train(self, dataset_dir: str, run_name: str, num_images: int = None, config_overrides: dict = None) -> Path:
        """Launch LoRA training.

        Args:
            dataset_dir: path to preprocessed dataset (with metadata.jsonl)
            run_name: name for this training run
            num_images: number of images (for auto config); counted from dataset if None
            config_overrides: optional dict to override auto-config values

        Returns:
            Path to the trained LoRA weights
        """
        dataset_path = Path(dataset_dir)
        if num_images is None:
            metadata_file = dataset_path / "metadata.jsonl"
            with open(metadata_file) as f:
                num_images = sum(1 for _ in f)

        config = self.auto_config(num_images)
        if config_overrides:
            config.update(config_overrides)

        output_path = self.output_dir / run_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility
        with open(output_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Build the training command using diffusers' built-in script
        cmd = [
            "accelerate", "launch",
            "src/training/train_lora_sdxl.py",
            f"--pretrained_model_name_or_path={config['pretrained_model_name_or_path']}",
            f"--train_data_dir={dataset_dir}",
            f"--output_dir={output_path}",
            f"--resolution={config['resolution']}",
            f"--train_batch_size={config['train_batch_size']}",
            f"--gradient_accumulation_steps={config['gradient_accumulation_steps']}",
            f"--max_train_steps={config['max_train_steps']}",
            f"--learning_rate={config['learning_rate']}",
            f"--rank={config['rank']}",
            f"--lr_scheduler={config['lr_scheduler']}",
            f"--lr_warmup_steps={config['lr_warmup_steps']}",
            f"--mixed_precision={config['mixed_precision']}",
            f"--seed={config['seed']}",
            f"--checkpointing_steps={config['checkpointing_steps']}",
            "--enable_xformers_memory_efficient_attention",
        ]

        print(f"Starting LoRA training: {run_name}")
        print(f"  Dataset: {dataset_dir} ({num_images} images)")
        print(f"  Steps: {config['max_train_steps']}, LR: {config['learning_rate']}, Rank: {config['rank']}")

        result = subprocess.run(cmd, check=True)
        print(f"Training complete. Weights saved to {output_path}")
        return output_path
