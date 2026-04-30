"""
Automated Style-Adaptive LoRA Pipeline

Usage:
    python pipeline.py --reference path/to/image.jpg --style "Japanese ink wash" --run-name ink_wash
    python pipeline.py --build-index  # Pre-compute WikiArt embeddings (run once)
"""

import argparse
import json
from pathlib import Path

from src.style_analysis.clip_analyzer import CLIPStyleAnalyzer
from src.retrieval.wikiart_retriever import WikiArtRetriever
from src.preprocessing.dataset_builder import DatasetBuilder
from src.training.lora_trainer import LoRATrainer
from src.evaluation.evaluator import LoRAEvaluator


def load_config(config_path: str = "configs/default.json") -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_index(config: dict):
    """One-time: pre-compute CLIP embeddings for the WikiArt dataset."""
    analyzer = CLIPStyleAnalyzer(config["clip_model"], config["clip_pretrained"])
    retriever = WikiArtRetriever(config["wikiart_dir"], config["embeddings_path"], analyzer)
    retriever.build_index()


def run_pipeline(reference_image: str, style_description: str, run_name: str, config: dict):
    """Full pipeline: analyze -> retrieve -> preprocess -> train -> evaluate."""

    print("=" * 60)
    print(f"Style-Adaptive LoRA Pipeline: {run_name}")
    print(f"Reference: {reference_image}")
    print(f"Style: {style_description}")
    print("=" * 60)

    # 1. Style Analysis
    print("\n[1/5] Analyzing reference style...")
    analyzer = CLIPStyleAnalyzer(config["clip_model"], config["clip_pretrained"])
    style_info = analyzer.analyze_style(reference_image, style_description)
    print(f"  Text-image alignment: {style_info.get('text_image_similarity', 'N/A'):.3f}")

    # 2. Retrieval
    print("\n[2/5] Retrieving similar images from WikiArt...")
    retriever = WikiArtRetriever(config["wikiart_dir"], config["embeddings_path"], analyzer)
    retriever.load_index()

    results = retriever.retrieve_combined(
        reference_image, style_description,
        top_k=config["retrieval_top_k"],
        image_weight=config["image_weight"],
        min_similarity=config["retrieval_min_similarity"],
    )
    print(f"  Retrieved {len(results)} images (similarity range: "
          f"{results[-1]['similarity']:.3f} - {results[0]['similarity']:.3f})")

    retrieved_paths = [r["path"] for r in results]

    # 3. Dataset Preprocessing
    print("\n[3/5] Building training dataset...")
    dataset_dir = f"/transfer/training_sets/{run_name}"
    builder = DatasetBuilder(dataset_dir, resolution=config["resolution"], trigger_word=config["trigger_word"])
    builder.build(retrieved_paths, style_keywords=style_description)

    # 4. LoRA Training
    print("\n[4/5] Training LoRA...")
    trainer = LoRATrainer(base_model=config["base_model"], output_dir=config["training_output_dir"])
    lora_path = trainer.train(dataset_dir, run_name, num_images=len(retrieved_paths))

    # 5. Evaluation
    print("\n[5/5] Evaluating...")
    evaluator = LoRAEvaluator(base_model=config["base_model"])
    test_prompts = [
        f"{config['trigger_word']}, a landscape painting",
        f"{config['trigger_word']}, a portrait",
        f"{config['trigger_word']}, a still life with flowers",
        f"{config['trigger_word']}, an abstract composition",
    ]
    eval_dir = f"{config['generation_output_dir']}/{run_name}"
    comparison = evaluator.compare_with_without_lora(
        test_prompts, str(lora_path), style_info["image_embedding"], eval_dir
    )

    # Save results
    results_path = Path(eval_dir) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "style_description": style_description,
            "num_training_images": len(retrieved_paths),
            "evaluation": {k: v for k, v in comparison.items() if k != "per_image"},
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  Base model similarity:  {comparison['base_model']['mean_similarity']:.3f}")
    print(f"  LoRA model similarity:  {comparison['with_lora']['mean_similarity']:.3f}")
    print(f"  Improvement:            +{comparison['improvement']:.3f}")
    print(f"  Results saved to: {eval_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Style-Adaptive LoRA Pipeline")
    parser.add_argument("--reference", type=str, help="Path to reference style image")
    parser.add_argument("--style", type=str, default="", help="Style description (e.g. 'Japanese ink wash')")
    parser.add_argument("--run-name", type=str, help="Name for this training run")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Config file path")
    parser.add_argument("--build-index", action="store_true", help="Build WikiArt CLIP index (run once)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.build_index:
        build_index(config)
    elif args.reference and args.run_name:
        run_pipeline(args.reference, args.style, args.run_name, config)
    else:
        parser.print_help()
