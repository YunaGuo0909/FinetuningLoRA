"""Evaluate trained LoRA models: generate images and compute style similarity."""

import torch
from pathlib import Path
from PIL import Image

from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from src.style_analysis.clip_analyzer import CLIPStyleAnalyzer


class LoRAEvaluator:
    """Generate images with trained LoRA and evaluate style transfer quality."""

    def __init__(self, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0", device: str = None):
        self.base_model = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.analyzer = CLIPStyleAnalyzer(device=self.device)

    def load_pipeline(self, lora_path: str = None):
        """Load the base model, optionally with LoRA weights."""
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model, torch_dtype=torch.float16
        ).to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()

        if lora_path:
            self.pipe.load_lora_weights(lora_path)
            print(f"Loaded LoRA from {lora_path}")

    def generate(self, prompts: list[str], output_dir: str, num_images_per_prompt: int = 1,
                 seed: int = 42, guidance_scale: float = 7.5) -> list[str]:
        """Generate images from prompts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        generated_paths = []

        generator = torch.Generator(self.device).manual_seed(seed)

        for i, prompt in enumerate(prompts):
            for j in range(num_images_per_prompt):
                image = self.pipe(
                    prompt, generator=generator, guidance_scale=guidance_scale
                ).images[0]
                path = output_path / f"{i:03d}_{j:02d}.png"
                image.save(path)
                generated_paths.append(str(path))

        print(f"Generated {len(generated_paths)} images in {output_dir}")
        return generated_paths

    def compute_style_similarity(self, generated_paths: list[str], reference_embedding: torch.Tensor) -> dict:
        """Compute CLIP similarity between generated images and reference style."""
        gen_embeddings = self.analyzer.encode_images(generated_paths)
        reference_embedding = reference_embedding / reference_embedding.norm(dim=-1, keepdim=True)

        similarities = (gen_embeddings @ reference_embedding.T).squeeze(-1)

        return {
            "mean_similarity": similarities.mean().item(),
            "std_similarity": similarities.std().item(),
            "min_similarity": similarities.min().item(),
            "max_similarity": similarities.max().item(),
            "per_image": similarities.tolist(),
        }

    def compare_with_without_lora(self, prompts: list[str], lora_path: str,
                                   reference_embedding: torch.Tensor,
                                   output_dir: str, seed: int = 42) -> dict:
        """Generate images with and without LoRA, compare style similarity."""
        out = Path(output_dir)

        # Without LoRA
        self.load_pipeline(lora_path=None)
        base_paths = self.generate(prompts, str(out / "base"), seed=seed)
        base_scores = self.compute_style_similarity(base_paths, reference_embedding)

        # With LoRA
        self.load_pipeline(lora_path=lora_path)
        lora_paths = self.generate(prompts, str(out / "lora"), seed=seed)
        lora_scores = self.compute_style_similarity(lora_paths, reference_embedding)

        return {
            "base_model": base_scores,
            "with_lora": lora_scores,
            "improvement": lora_scores["mean_similarity"] - base_scores["mean_similarity"],
        }
