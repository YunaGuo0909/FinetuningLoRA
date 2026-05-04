"""Evaluate motion generation quality: FID, diversity, style consistency.

Metrics:
    - FID (Frechet Inception Distance) on motion feature space
    - Diversity: average pairwise distance of generated motions
    - Style Consistency: CLIP similarity between motion descriptions and target style
    - Jitter: acceleration magnitude (measures smoothness)
"""

import json
import numpy as np
import torch
from pathlib import Path
from scipy import linalg


def compute_fid(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    """Compute FID between two sets of motion features.

    Args:
        real_features: (N, D) features from real motions
        gen_features: (M, D) features from generated motions
    """
    mu_r, sigma_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def compute_diversity(motions: np.ndarray, num_pairs: int = 300) -> float:
    """Average pairwise L2 distance of generated motions.

    Args:
        motions: (N, T, D) generated motion sequences
    """
    N = motions.shape[0]
    if N < 2:
        return 0.0

    flat = motions.reshape(N, -1)
    indices = np.random.choice(N, size=(min(num_pairs, N * (N - 1) // 2), 2), replace=True)
    distances = np.linalg.norm(flat[indices[:, 0]] - flat[indices[:, 1]], axis=1)
    return float(distances.mean())


def compute_jitter(positions: np.ndarray, fps: int = 20) -> float:
    """Compute average jitter (acceleration magnitude) as smoothness metric.

    Args:
        positions: (T, n_joints, 3) joint positions
        fps: frames per second
    """
    if positions.shape[0] < 3:
        return 0.0

    velocity = np.diff(positions, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jitter = np.linalg.norm(acceleration, axis=-1).mean()
    return float(jitter)


def extract_motion_features(motions: np.ndarray) -> np.ndarray:
    """Extract simple features from motion sequences for FID computation.

    Uses statistical features: mean, std, min, max per joint dimension.

    Args:
        motions: (N, T, D) motion sequences
    Returns:
        (N, D*4) feature vectors
    """
    N = motions.shape[0]
    features = np.zeros((N, motions.shape[2] * 4))
    for i in range(N):
        m = motions[i]
        features[i] = np.concatenate([m.mean(0), m.std(0), m.min(0), m.max(0)])
    return features


class MotionEvaluator:
    """Evaluate LoRA-tuned motion generation vs base model."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std.copy()
        self.std[self.std < 1e-5] = 1.0

    def denormalize(self, motion: np.ndarray) -> np.ndarray:
        return motion * self.std + self.mean

    def evaluate_batch(self, generated: np.ndarray, reference: np.ndarray = None) -> dict:
        """Evaluate a batch of generated motions.

        Args:
            generated: (N, T, D) generated motions (normalized)
            reference: (M, T, D) reference motions for FID (normalized)
        Returns:
            dict of metrics
        """
        results = {}

        # Diversity
        results["diversity"] = compute_diversity(generated)

        # FID (if reference provided)
        if reference is not None and reference.shape[0] > 1:
            gen_feat = extract_motion_features(generated)
            ref_feat = extract_motion_features(reference)
            results["fid"] = compute_fid(ref_feat, gen_feat)

        # Jitter (smoothness) - use denormalized positions
        jitters = []
        for i in range(generated.shape[0]):
            motion = self.denormalize(generated[i])
            # Extract joint positions from features (dims 4:67 = 21 joints * 3)
            n_pos_dims = 21 * 3
            if motion.shape[1] >= 67:
                positions = motion[:, 4:67].reshape(-1, 21, 3)
                jitters.append(compute_jitter(positions))
        if jitters:
            results["jitter_mean"] = float(np.mean(jitters))
            results["jitter_std"] = float(np.std(jitters))

        return results

    def compare_base_vs_lora(
        self,
        base_motions: np.ndarray,
        lora_motions: np.ndarray,
        reference: np.ndarray = None,
    ) -> dict:
        """Compare base model vs LoRA model outputs.

        Args:
            base_motions: (N, T, D) motions from base MDM
            lora_motions: (N, T, D) motions from LoRA MDM
            reference: (M, T, D) real motions for FID
        """
        base_eval = self.evaluate_batch(base_motions, reference)
        lora_eval = self.evaluate_batch(lora_motions, reference)

        return {
            "base_model": base_eval,
            "with_lora": lora_eval,
            "improvement": {
                k: lora_eval.get(k, 0) - base_eval.get(k, 0)
                for k in base_eval
            },
        }

    def save_results(self, results: dict, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {path}")
