"""Retrieve style-similar images from WikiArt dataset using CLIP embeddings."""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.style_analysis.clip_analyzer import CLIPStyleAnalyzer


class WikiArtRetriever:
    """Retrieves similar-style images from a pre-indexed WikiArt dataset."""

    def __init__(self, dataset_dir: str, embeddings_path: str = None, analyzer: CLIPStyleAnalyzer = None):
        self.dataset_dir = Path(dataset_dir)
        self.analyzer = analyzer or CLIPStyleAnalyzer()
        self.embeddings_path = Path(embeddings_path) if embeddings_path else self.dataset_dir / "clip_embeddings.pt"
        self.image_paths: list[str] = []
        self.embeddings: torch.Tensor = None

    def build_index(self, batch_size: int = 64):
        """Pre-compute CLIP embeddings for all images in the WikiArt dataset."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_paths = sorted([
            str(p) for p in self.dataset_dir.rglob("*")
            if p.suffix.lower() in image_extensions
        ])
        print(f"Found {len(self.image_paths)} images in {self.dataset_dir}")

        self.embeddings = self.analyzer.encode_images(self.image_paths, batch_size=batch_size)

        torch.save({
            "image_paths": self.image_paths,
            "embeddings": self.embeddings,
        }, self.embeddings_path)
        print(f"Saved embeddings to {self.embeddings_path}")

    def load_index(self):
        """Load pre-computed embeddings from disk."""
        data = torch.load(self.embeddings_path, weights_only=True)
        self.image_paths = data["image_paths"]
        self.embeddings = data["embeddings"]
        print(f"Loaded {len(self.image_paths)} embeddings from {self.embeddings_path}")

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 50, min_similarity: float = 0.5) -> list[dict]:
        """Find top-K most similar images to the query embedding."""
        if self.embeddings is None:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")

        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.view(1, -1)

        similarities = (self.embeddings @ query_embedding.T).squeeze(-1)
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            sim = similarities[idx].item()
            if sim < min_similarity:
                break
            results.append({
                "path": self.image_paths[idx],
                "similarity": sim,
            })
        return results

    def retrieve_by_image(self, image_path: str, top_k: int = 50, min_similarity: float = 0.5) -> list[dict]:
        """Retrieve similar images given a reference image."""
        embedding = self.analyzer.encode_image(image_path)
        return self.retrieve(embedding, top_k=top_k, min_similarity=min_similarity)

    def retrieve_by_text(self, text: str, top_k: int = 50, min_similarity: float = 0.2) -> list[dict]:
        """Retrieve images matching a text description."""
        embedding = self.analyzer.encode_text([text])
        return self.retrieve(embedding, top_k=top_k, min_similarity=min_similarity)

    def retrieve_combined(self, image_path: str, text: str, top_k: int = 50,
                          image_weight: float = 0.7, min_similarity: float = 0.4) -> list[dict]:
        """Retrieve using both image and text query (weighted combination)."""
        img_emb = self.analyzer.encode_image(image_path)
        txt_emb = self.analyzer.encode_text([text])
        combined = image_weight * img_emb + (1 - image_weight) * txt_emb
        return self.retrieve(combined, top_k=top_k, min_similarity=min_similarity)
