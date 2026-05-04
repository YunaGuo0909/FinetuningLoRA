# Motion Style-Adaptive LoRA Fine-Tuning Pipeline

A pipeline for fine-tuning Motion Diffusion Models (MDM) with LoRA adapters to learn stylised human motion generation. Given a set of style-specific motion capture data (e.g. "zombie walk", "happy run"), the system trains lightweight LoRA adapters that steer the base model toward generating motions in that style.

## Project Structure

```
FinetuningLoRA/
├── configs/default.json              # All hyperparameters and paths
├── pipeline.py                       # End-to-end orchestrator
├── src/
│   ├── models/
│   │   ├── mdm.py                    # MDM Transformer (LoRA-friendly Q/K/V)
│   │   └── diffusion.py              # Gaussian diffusion (DDPM + DDIM sampling)
│   ├── data/
│   │   ├── humanml_dataset.py        # HumanML3D & StyleMotion dataloaders
│   │   └── bvh_converter.py          # 100STYLE BVH -> 263-dim features
│   ├── training/
│   │   └── train_mdm_lora.py         # LoRA fine-tuning script (accelerate)
│   ├── evaluation/
│   │   └── evaluator.py              # FID, Diversity, Jitter metrics
│   └── visualization/
│       └── motion_viz.py             # Skeleton animation rendering (GIF/MP4)
├── scripts/
│   ├── prepare_data.py               # Download & verify datasets
│   └── convert_100style.py           # Batch BVH conversion
└── requirements.txt
```

## Architecture

### Base Model: MDM (Motion Diffusion Model)

The denoising network is a Transformer encoder that takes noisy motion sequences `(B, T, 263)` conditioned on timestep and CLIP text embeddings. Our implementation replaces PyTorch's fused `MultiheadAttention` with separate `to_q`, `to_k`, `to_v`, `to_out` linear layers so that PEFT LoRA can target them directly.

- **Input**: 263-dim HumanML3D features (joint positions, velocities, rotations, foot contacts)
- **Conditioning**: CLIP ViT-B/32 text embeddings (512-dim)
- **Parameters**: ~18.5M total; LoRA (rank=16) trains only ~200K

### LoRA Fine-Tuning

LoRA adapters are injected into the self-attention layers of all 8 Transformer blocks. The base model weights are frozen; only the low-rank matrices are updated during training.

### Diffusion

Cosine beta schedule with 1000 timesteps. Training uses noise prediction loss with Min-SNR weighting. Inference uses DDIM sampling (50 steps) for fast generation.

## Setup (Training Machine)

All data and outputs live under `/transfer/` to avoid home directory quota limits.

```bash
# 1. Clone the repo
git clone https://github.com/YunaGuo0909/FinetuningLoRA.git
cd FinetuningLoRA

# 2. Create virtual environment on /transfer (avoids disk quota)
uv venv /transfer/lora_venv --python 3.11
ln -s /transfer/lora_venv .venv
source .venv/bin/activate
export UV_CACHE_DIR=/transfer/uv_cache
uv pip install -r requirements.txt

# 3. Download datasets and pretrained weights (skips if already present)
python scripts/prepare_data.py

# 4. Verify everything is ready
python scripts/prepare_data.py --verify
```

## Usage

### Step 1: Convert Style Data

Convert 100STYLE BVH files to HumanML3D 263-dim format. Skips styles that are already converted.

```bash
# List available styles
python scripts/convert_100style.py --list

# Convert specific styles
python scripts/convert_100style.py --styles Zombie,Happy,Old

# Convert all styles
python scripts/convert_100style.py --all
```

### Step 2: Train LoRA

```bash
accelerate launch src/training/train_mdm_lora.py \
    --style_data_dir /transfer/datasets/style_converted \
    --output_dir /transfer/outputs/models/zombie \
    --lora_rank 16 \
    --max_train_steps 2000
```

### Step 3: Generate and Evaluate

```bash
# Full pipeline: convert -> train -> generate -> evaluate
python pipeline.py --full \
    --style-data /transfer/datasets/100STYLE/Zombie \
    --style "zombie" \
    --run-name zombie_walk

# Or just generate + evaluate with existing LoRA
python pipeline.py --generate --evaluate \
    --lora-path /transfer/outputs/models/zombie/final \
    --prompt "a person walking in zombie style" \
    --run-name zombie_walk
```

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| FID | Frechet Inception Distance on motion features | Lower is better |
| Diversity | Average pairwise distance of generated motions | Higher is better |
| Jitter | Mean acceleration magnitude (smoothness) | Lower is better |

The evaluator runs generation with and without LoRA using the same seed, producing a side-by-side comparison.

## Datasets

- **HumanML3D** (Guo et al., 2022): 14,616 motions with 44,970 text descriptions. Used for base model pre-training.
- **100STYLE** (Mason et al., 2022): 100 locomotion styles, ~4M frames in BVH format. Used for LoRA style fine-tuning.

## Key Dependencies

- PyTorch >= 2.1
- diffusers, accelerate, peft (Hugging Face ecosystem)
- open-clip-torch (text conditioning)
- matplotlib (visualisation)

## Acknowledgements

- [Motion Diffusion Model (MDM)](https://github.com/GuyTevet/motion-diffusion-model) - Tevet et al., 2023
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) - Guo et al., 2022
- [100STYLE](https://www.ianmaurice.com/100style/) - Mason et al., 2022
- [PEFT](https://github.com/huggingface/peft) - Hugging Face LoRA implementation
