# Project Progress Log - 2026.05.04

## Project Overview

**Goal**: Build a Motion Style-Adaptive LoRA fine-tuning pipeline for Motion Diffusion Models (MDM). Given style-specific motion capture data (e.g. "zombie walk"), train lightweight LoRA adapters that steer the base model toward generating motions in that style.

**Coursework**: Generative AI for Media (Level 7), Deadline 15/05/2026

---

## Phase 1: Project Pivot

### Original Plan
- Image style transfer using Stable Diffusion XL + LoRA
- WikiArt dataset for style retrieval
- CLIP-based style analysis pipeline

### Decision to Pivot
- Changed direction to **3D human motion generation** with LoRA fine-tuning
- Chose **MDM (Motion Diffusion Model)** as base model for simplicity and LoRA compatibility
- Plan to also train **MLD** later for comparison in the reflective paper

### New Architecture
- **Base Model**: MDM (Transformer encoder, ~18M params)
- **LoRA targets**: Self-attention Q/K/V/Out projections (only ~524K trainable params)
- **Conditioning**: CLIP ViT-B/32 text embeddings
- **Diffusion**: Cosine beta schedule, 1000 timesteps, DDIM sampling (50 steps)

---

## Phase 2: Code Restructuring

### Removed (old image pipeline)
- `src/style_analysis/` - CLIP image analyzer
- `src/retrieval/` - WikiArt retriever
- `src/preprocessing/` - Image dataset builder
- `src/training/lora_trainer.py` - SD LoRA trainer
- `src/training/train_lora_sdxl.py` - SDXL training script

### Created (new motion pipeline)
| File | Purpose |
|------|---------|
| `src/models/mdm.py` | MDM architecture with separate Q/K/V for LoRA |
| `src/models/diffusion.py` | Gaussian diffusion (DDPM + DDIM) |
| `src/data/humanml_dataset.py` | HumanML3D + StyleMotion dataloaders |
| `src/data/bvh_converter.py` | 100STYLE BVH -> 263-dim features |
| `src/training/train_mdm_lora.py` | LoRA fine-tuning with accelerate |
| `src/evaluation/evaluator.py` | FID, Diversity, Jitter metrics |
| `src/visualization/motion_viz.py` | Skeleton animation rendering |
| `scripts/prepare_data.py` | Dataset download/verification |
| `scripts/convert_100style.py` | Batch BVH conversion |
| `pipeline.py` | End-to-end orchestrator |

### Key Design Decision: LoRA-friendly Attention
Official MDM uses `nn.MultiheadAttention` with fused `in_proj_weight` (combined QKV). PEFT LoRA cannot target this. Solution: replaced with separate `to_q`, `to_k`, `to_v`, `to_out` Linear layers. Weight loader splits the official fused weights automatically.

---

## Phase 3: Training Machine Setup

### Environment
- **Machine**: w33107 (university lab)
- **GPU**: NVIDIA RTX 4080 (16GB)
- **OS**: Linux
- **Python**: 3.11 via uv

### Problem: Home Directory Disk Quota
The home directory has a strict quota. PyTorch, CLIP model weights, and venv all exceed it.

**Solution**: Put everything on `/transfer/` (no quota):
```
/transfer/
  lora_venv/           -> symlinked as .venv
  loradataset/
    humanml3d/         -> from Kaggle
    100STYLE/          -> from Zenodo
    style_converted/   -> BVH conversion output
  lorapretrain/
    humanml_trans_enc_512/  -> official MDM weights
  uv_cache/            -> uv package cache
  hf_cache/            -> HuggingFace model cache
  outputs/             -> training outputs
```

Environment variables needed:
```bash
export UV_CACHE_DIR=/transfer/uv_cache
export HF_HOME=/transfer/hf_cache
export TRANSFORMERS_CACHE=/transfer/hf_cache
```

### Problem: Python 3.9 on System
Code uses `X | Y` type syntax (Python 3.10+). System only has 3.9.

**Solution**: `uv venv --python 3.11` auto-downloads 3.11.

### Problem: venv on Home Directory
Even the venv itself exceeds home quota.

**Solution**: Create venv on `/transfer`, symlink back:
```bash
uv venv /transfer/lora_venv --python 3.11
ln -s /transfer/lora_venv .venv
```

---

## Phase 4: Data Preparation

### HumanML3D (base model training data)
- 17,126 motions, 263-dim features, with 44,970 text descriptions
- **Problem**: Official repo doesn't distribute data directly (AMASS license)
- **Solution**: Downloaded from Kaggle (mrriandmstique/humanml3d, 5.8GB)
- **Problem**: Kaggle ZIP had nested directories (`humanml3d/HumanML3D/humanml/`)
- **Solution**: Flattened with `for item in *; do mv "$item" ...; done`

### 100STYLE (style-specific LoRA data)
- 100 locomotion styles, 810 BVH files
- **Solution**: Downloaded directly from Zenodo on training machine (has internet for academic sites):
  ```
  wget "https://zenodo.org/records/8127870/files/100STYLE.zip?download=1"
  ```

### MDM Pretrained Weights
- `humanml_trans_enc_512/model000475000.pt` (78MB)
- Downloaded from Google Drive (official MDM repo link)

### Verification Script
`prepare_data.py --verify` checks all three datasets:
```
HumanML3D: READY (17,126 motions, texts OK, stats OK)
100STYLE: READY (810 BVH files)
MDM weights: READY (78.0 MB)
```

---

## Phase 5: BVH Conversion

### Problem: Joint Name Mismatch
Default joint mapping assumed Mixamo naming convention (`LeftUpLeg`, `RightUpLeg`, etc.). 100STYLE uses different names (`LeftHip`, `RightHip`, `Chest`, `Chest2`, etc.).

**Symptoms**: All joints mapped to root, warnings like `joint 'L_Hip' not found in BVH`.

**Solution**: Added 100STYLE skeleton mapping to `DEFAULT_JOINT_MAP`:
```python
"Chest": "Spine1", "Chest2": "Spine2", "Chest3": "Spine3",
"LeftHip": "L_Hip", "RightHip": "R_Hip",
"LeftKnee": "L_Knee", "RightKnee": "R_Knee", ...
```

### Converted Styles
Selected 5 styles with strong visual contrast:
| Style | Rationale |
|-------|-----------|
| Zombie | Classic stylization, stiff/dragging |
| Elated | Light/bouncy (originally searched "Happy", not in dataset) |
| Old | Slow, physiological characteristics |
| Robot | Mechanical, non-human |
| Drunk | Non-linear, tests model capacity |

**Result**: 40 motions (8 per style) in `/transfer/loradataset/style_converted/`

---

## Phase 6: Training (In Progress)

### Problem: MDM Weight Loading Crash
`load_pretrained_mdm()` assumed official MDM uses Sequential for input/output, but it actually uses single Linear layers.

**Error**: `ValueError: invalid literal for int() with base 10: 'poseEmbedding'`

**Solution**: Rewrote MDM architecture to match official structure:
- `input_process`: single `nn.Linear` (was 2-layer Sequential)
- `output_process`: single `nn.Linear` (was 2-layer Sequential)
- `TransformerBlock`: post-norm (was pre-norm)
- Condition tokens: timestep + text added together as 1 token (was 2 separate tokens)

### Problem: PyTorch Too New for CUDA Driver
`torch==2.11.0` requires newer CUDA driver than the machine has (CUDA 12.9 / driver 575.57).

**Solution**: Downgraded to `torch==2.4.0` with cu121 index.

### Problem: CLIP Download Fills Home Quota
HuggingFace cache defaults to `~/.cache/huggingface`.

**Solution**: `export HF_HOME=/transfer/hf_cache`

### Problem: `src/data/` Not Tracked by Git
`.gitignore` had `data/` which matched ALL directories named `data`, including `src/data/`.

**Solution**: Changed to `/data/` (root-only match).

### Current Status
- MDM weights load successfully (138 params mapped)
- LoRA applied: 524,288 trainable params (2.83% of total)
- CLIP text encoder downloading to `/transfer/hf_cache/`
- **Next**: Re-run training command with correct cache paths

---

## Remaining Tasks

| Task | Status | Est. Time |
|------|--------|-----------|
| Complete MDM + LoRA training | In progress | 1-2 hours |
| Generate + evaluate results | Not started | 1 hour |
| MLD + LoRA training (comparison) | Not started | 2 days |
| Visualization outputs | Not started | 1 hour |
| Critical Reflective Paper | Not started | 2-3 days |
