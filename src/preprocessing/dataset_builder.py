"""Preprocesses retrieved images into a LoRA training dataset with auto-captioning."""

import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class DatasetBuilder:
    """Builds a captioned training dataset from retrieved images."""

    def __init__(self, output_dir: str, resolution: int = 512, trigger_word: str = "sty1e"):
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.trigger_word = trigger_word
        self.captioner = None

    def _load_captioner(self, device=None):
        """Lazy-load BLIP-2 captioning model."""
        if self.captioner is not None:
            return
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Salesforce/blip2-opt-2.7b"
        self.caption_processor = Blip2Processor.from_pretrained(model_name)
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.caption_device = device
        self.captioner = True

    def resize_and_crop(self, image: Image.Image) -> Image.Image:
        """Resize image to training resolution with center crop."""
        w, h = image.size
        scale = self.resolution / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - self.resolution) // 2
        top = (new_h - self.resolution) // 2
        return image.crop((left, top, left + self.resolution, top + self.resolution))

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        """Generate a caption for an image using BLIP-2."""
        self._load_captioner()
        inputs = self.caption_processor(image, return_tensors="pt").to(self.caption_device)
        output = self.caption_model.generate(**inputs, max_new_tokens=50)
        caption = self.caption_processor.decode(output[0], skip_special_tokens=True).strip()
        return caption

    def build(self, image_paths: list[str], style_keywords: str = "") -> Path:
        """Process images and create captioned training dataset.

        Args:
            image_paths: list of source image file paths
            style_keywords: style description to inject into captions

        Returns:
            Path to the created dataset directory
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = []

        for i, src_path in enumerate(tqdm(image_paths, desc="Building dataset")):
            try:
                image = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {src_path}: {e}")
                continue

            processed = self.resize_and_crop(image)
            filename = f"{i:04d}.png"
            processed.save(self.output_dir / filename)

            caption = self.generate_caption(processed)
            # Inject trigger word and style keywords
            full_caption = f"{self.trigger_word}, {style_keywords}, {caption}" if style_keywords else f"{self.trigger_word}, {caption}"

            metadata.append({"file_name": filename, "text": full_caption})

        # Save metadata in the format expected by diffusers
        metadata_path = self.output_dir / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dataset built: {len(metadata)} images in {self.output_dir}")
        return self.output_dir
