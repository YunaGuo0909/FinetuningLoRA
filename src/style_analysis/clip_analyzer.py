"""Style analysis using CLIP - extracts visual style embeddings from reference images."""

import torch
import open_clip
from PIL import Image
from pathlib import Path


class CLIPStyleAnalyzer:
    """Analyzes reference images using CLIP to extract style embeddings."""

    def __init__(self, model_name="ViT-L-14", pretrained="openai", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode a single image to CLIP embedding."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu()

    @torch.no_grad()
    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode multiple images to CLIP embeddings."""
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Skipping {p}: {e}")
                    continue
            if not images:
                continue
            batch_tensor = torch.stack(images).to(self.device)
            embeddings = self.model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text descriptions to CLIP embeddings."""
        tokens = self.tokenizer(texts).to(self.device)
        embeddings = self.model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu()

    def analyze_style(self, image_path: str, style_description: str = None) -> dict:
        """Analyze a reference image's style, returning embedding and optional text alignment."""
        image_embedding = self.encode_image(image_path)
        result = {"image_embedding": image_embedding}

        if style_description:
            text_embedding = self.encode_text([style_description])
            similarity = (image_embedding @ text_embedding.T).item()
            result["text_embedding"] = text_embedding
            result["text_image_similarity"] = similarity

        return result
