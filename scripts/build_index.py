"""Build CLIP embedding index for the WikiArt dataset.

Run on training machine (one-time, takes ~30-60 min depending on dataset size):
    python scripts/build_index.py
"""

import sys
sys.path.insert(0, ".")

from src.style_analysis.clip_analyzer import CLIPStyleAnalyzer
from src.retrieval.wikiart_retriever import WikiArtRetriever

WIKIART_DIR = "/transfer/wikidatasets/steubk/wikiart/version"
EMBEDDINGS_PATH = "/transfer/embeddings/wikiart_clip.pt"

if __name__ == "__main__":
    import os
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    print("Loading CLIP model...")
    analyzer = CLIPStyleAnalyzer(model_name="ViT-L-14", pretrained="openai")

    print(f"Building index from {WIKIART_DIR}...")
    retriever = WikiArtRetriever(WIKIART_DIR, EMBEDDINGS_PATH, analyzer)
    retriever.build_index(batch_size=64)

    print("Done! Index saved to", EMBEDDINGS_PATH)
