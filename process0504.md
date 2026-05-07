# Project Progress Log - 2026.05.04 ~ 05.07

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

### First Training Run: loss=nan

Training ran to completion (2000 steps, ~6 min) but loss was `nan` throughout.

**Diagnosis**: Wrote `scripts/diagnose_data.py` to compare data ranges:
| Data | Raw range | After normalization (HumanML3D mean/std) |
|------|-----------|------------------------------------------|
| HumanML3D | [-1.0, 1.75] | [-9, +15] |
| Style (BVH-converted) | [-6.26, 6.27] | [-488, +484] |

The BVH-converted style data has completely different value distributions from HumanML3D's processed features. Using HumanML3D's Mean/Std to normalize the style data produces values 50x too large, causing numerical overflow in the diffusion loss.

**Root Cause**: Our BVH converter (`bvh_converter.py`) computes 263-dim features from joint positions, but the computation pipeline (especially rotation representations, velocity calculations, unit scaling) differs from HumanML3D's original processing scripts. The features are structurally similar but numerically incompatible.

**Solution**: Modified `StyleMotionDataset` to compute its own mean/std from the style data at initialization, rather than using HumanML3D's stats. This ensures the normalized data falls in a reasonable range regardless of the raw feature scale.

```python
# Before: used HumanML3D stats (mismatched)
self.mean = mean  # from HumanML3D Mean.npy
self.std = std    # from HumanML3D Std.npy

# After: compute from style data itself
all_data = np.concatenate([np.load(f) for f in motion_files], axis=0)
self.mean = all_data.mean(axis=0)
self.std = all_data.std(axis=0)
```

### Current Status (end of Phase 6)
- MDM weights load successfully (138 params mapped, 1 missing for pos_embedding)
- LoRA applied: 524,288 trainable params (2.83% of total)
- CLIP ViT-B/32 loaded from HuggingFace (cached on `/transfer/hf_cache/`)
- First training run completed but with nan loss (data normalization mismatch)
- **Next**: Re-run training with self-computed normalization (style_lora_v2)

---

## Phase 7: Switching to Official MDM Codebase

### Problem: Custom MDM Produces Garbage Output
Generated motions from the custom MDM implementation were garbled — random tangled lines instead of human skeletons, despite tensors having correct shapes.

**Root Cause**: Subtle architectural differences between our reimplementation and the official MDM code. Even small mismatches (normalization order, conditioning token layout, etc.) cause the pretrained weights to produce garbage.

**Solution**: Switched to the **official MDM codebase** (`/transfer/mdm_official`) with a wrapper layer:
- `src/models/mdm_official.py` — loads official MDM, patches SMPL init, replaces attention layers
- `SplitQKVAttention` — drop-in replacement for `nn.MultiheadAttention`, splits fused QKV into separate `to_q`, `to_k`, `to_v`, `to_out` Linear layers for PEFT LoRA targeting

### Problem: SMPL Body Model Not Available
Official MDM imports SMPL body model, which requires `SMPL_NEUTRAL.pkl` (not needed for our hml_vec representation).

**Solution**: Patched `SMPL.__init__` with a dummy during model creation:
```python
def _dummy_smpl_init(self, **kwargs):
    nn.Module.__init__(self)
_smpl_module.SMPL.__init__ = _dummy_smpl_init
```

### Problem: `args.json` Missing Fields
Newer MDM code expects fields not present in older checkpoints (`unconstrained`, `keyframe`, etc.).

**Solution**: Added defaults dict to fill in missing fields before creating the model.

### Problem: x₀ Prediction vs Noise Prediction
MDM uses `predict_xstart=True` — the model predicts the clean sample x₀ directly, NOT noise. Our training loss and DDIM sampler were both written for noise prediction.

**Solution**:
- Training loss changed from `(pred - noise)²` to `(pred_x0 - x_0)²`
- DDIM sampler corrected to derive noise from x₀ prediction, then step

### Problem: `model.to(device)` Returns None
Official MDM's `.to()` method doesn't return `self`.

**Solution**: `model.to(device)` → `model.to(device); return model`

### Problem: `replace_attention_layers` Also Replaced CLIP Layers
The function replaced ALL `nn.MultiheadAttention` in the model, including CLIP's (which uses float16). LoRA layers are float32 → dtype mismatch crash.

**Solution**: Skip modules with `clip_model` in the name:
```python
for name, module in model.named_modules():
    if "clip_model" in name:
        continue
```

### Problem: SplitQKVAttention Missing Attributes
PyTorch's `TransformerEncoderLayer` checks `self_attn.batch_first` and passes `is_causal` kwarg.

**Solution**: Added `self.batch_first = False` attribute and `**kwargs` to forward signature.

---

## Phase 8: First Real Training Run (style_lora_v3)

### Training with Self-Normalized Style Data
Used `StyleMotionDataset` with self-computed mean/std (not HumanML3D stats) + clip to [-5, 5].

**Result**: Training completed (2000 steps), loss converged (not NaN). But...

### Problem: Base vs LoRA Output Identical
Generation showed **no difference** between base model and LoRA model.

**Diagnosis** (`scripts/diagnose_lora.py`):
- LoRA path `/transfer/loraoutputs/models/style_lora_v3/final` **did not exist**
- The training had never actually been run! All previous runs had been debugging other issues (model loading, CLIP dtype, attention interface)
- Generation script fell back to using base motions for both outputs

### Actually Running Training
After fixing all model loading bugs, ran training for real:
- 40 motions, 2000 steps, loss converged
- LoRA weights saved successfully

### Problem: Normalization Space Mismatch
Even with successful training, LoRA effect was weak because:
- Base MDM pretrained in HumanML3D-normalized space
- LoRA trained in self-normalized style data space (different mean/std)
- At inference, diffusion operates in HumanML3D space → LoRA's learned adjustments don't transfer

**Solution**: Changed `StyleMotionDataset` to use HumanML3D mean/std + clip extreme values to [-5, 5]:
```python
self.mean = mean.copy()  # from HumanML3D
self.std = std.copy()
motion = np.clip((motion - self.mean) / self.std, -5.0, 5.0)
```

### Result After Fix
- LoRA output now visibly different from base model
- But 10.4% of values clipped → significant information loss
- Skeleton animation had foot sliding (common diffusion model issue)
- Style effect visible but weak

---

## Phase 9: Experimental Design Fix

### Problem: LoRA Value Not Demonstrated
Testing with style-specific prompts ("walking like a zombie") showed little difference because the base MDM already understands these from CLIP text conditioning.

**Key Insight**: LoRA's value is making the model **default to a style** even with generic prompts. Should test with prompts like "a person walking forward" and show that LoRA adds style implicitly.

### Attempt: Filter HumanML3D by Captions
Created `scripts/filter_style_data.py` to select style-relevant motions from HumanML3D by text:
- Searched captions for keywords (zombie, old, drunk, happy, etc.)
- Found 2,974 unique motions across 8 styles
- Data already in correct HumanML3D feature space → zero normalization mismatch

**Training (style_lora_v4)**: 2974 motions, 3000 steps, HumanML3D normalization.

**Problem**: This data comes from the same distribution the base model was trained on → LoRA learns nothing new. Base model already knows these styles from its original training.

### Correct Approach: Fix BVH Converter
For LoRA to add value, it must learn from **external style data** (100STYLE mocap) that the base model hasn't seen. The normalization mismatch must be fixed at the source.

---

## Phase 10: BVH Converter Fix (Current)

### Root Cause of Normalization Mismatch
263-dim features contain 126 dimensions of **6D rotation data** (dims 67–192). Our original BVH converter used **identity placeholders** `[1,0,0,1,0,0]` for all rotation features — constant values completely unlike real HumanML3D rotation data. This single issue caused the bulk of the normalization mismatch.

### Fix: Extract Real Rotations from BVH
BVH files contain local Euler rotations for every joint at every frame. Modified the converter to:

1. **`forward_kinematics()`** now returns both positions AND local rotation matrices
2. **`compute_humanml3d_features()`** accepts rotation matrices and converts to 6D representation using `rotation_matrix_to_6d()`
3. **`rotation_matrix_to_6d()`** fixed to produce standard format: `[col0, col1]` = first two columns of rotation matrix
4. Root angular velocity wrapped to `[-π, π]`

### Reconversion Results
Reconverted 5 styles (40 motions) with fixed converter → `style_converted_v2/`:

```
Per-group normalized ranges:
Group                      Converted            HumanML3D
root_rot_vel           [-27.2, 30.4]        [-15.3, 15.8]
root_vel_xz              [-8.5, 8.3]        [-14.2, 36.7]
root_height              [-1.1, 1.1]          [-5.8, 5.8]
joint_positions         [-11.7, 8.4]        [-11.3, 16.3]
joint_rotations_6d       [-5.1, 5.1]          [-6.6, 5.5]  ← was [-489, 489]!
joint_velocities       [-19.9, 24.7]        [-31.2, 41.8]
foot_contacts            [-2.4, 0.4]          [-2.4, 0.4]

Values outside [-5, 5]: 8.5% (down from ~50% with identity rotations)
```

**Rotation features now align with HumanML3D.** Main remaining outlier is `root_rot_vel` (some sudden orientation changes in BVH data).

### Current Status
- BVH converter fixed with real rotation data
- `style_converted_v2/` generated (40 motions, 5 styles)
- **Next**: Train style_lora_v5 with fixed data, generate with generic prompts, compare base vs LoRA

---

## Phase Summary

| Version | Data Source | Normalization | Result |
|---------|-----------|---------------|--------|
| v1 | 100STYLE BVH (identity rotations) | Self-computed | NaN loss |
| v2 | 100STYLE BVH (identity rotations) | Self-computed | Trained but no effect at inference (space mismatch) |
| v3 | 100STYLE BVH (identity rotations) | HumanML3D + clip [-5,5] | Weak effect, 10.4% clipped |
| v4 | HumanML3D filtered by captions | HumanML3D (native) | Trained but base already knows these styles |
| **v5** | **100STYLE BVH (real rotations)** | **HumanML3D + clip [-5,5]** | **Mixed 5 styles → weak effect (see Phase 11)** |
| **v6 (HML filtered)** | **HumanML3D filtered by captions** | **HumanML3D (native)** | **No effect — data already in pre-training distribution** |
| **v7 (per-style BVH)** | **100STYLE BVH per-style** | **HumanML3D + clip [-5,5]** | **Visible style differences (see Phase 13)** |

---

## Phase 11: Training v5 with Fixed BVH Data (Mixed Styles)

### Training
- Data: `/transfer/loradataset/style_converted_v2/` (40 motions, 5 styles mixed)
- Config: 3000 steps, lr=1e-4, batch_size=64, cosine schedule
- Loss converged normally

### Generation Results
- Generated with generic prompts ("a person walking forward", etc.)
- Base vs LoRA comparison: **slight difference visible** (LoRA output had different range)
- Value ranges: Base [-17.0, 10.5], LoRA [-11.3, 10.1]
- But visual difference was weak — mixing 5 styles diluted the style signal

### Skeleton Visualization Issue
- All skeletons (base and LoRA) showed an **extra bone between the legs** — a point hanging below the crotch area
- **Cause**: Root joint (pelvis, joint 0) positioned lower than hip joints (1, 2), creating a downward triangle
- **Fix**: Set root position to midpoint of hip joints in `motion_features_to_positions()`:
  ```python
  for t in range(T):
      positions[t, 0] = (positions[t, 1] + positions[t, 2]) / 2
  ```
- Result: skeleton looks correct after fix

---

## Phase 12: HumanML3D Filtered Experiment (v6) — Failed

### Hypothesis
Use HumanML3D data filtered by style keywords (e.g. "old", "angry", "drunk") — larger dataset (2974 motions), no normalization issues.

### Setup
- `scripts/filter_style_data.py` modified to output per-style directories:
  ```
  style_filtered/old/motions/ + metadata.jsonl
  style_filtered/angry/motions/ + metadata.jsonl
  ...
  style_filtered/mixed/motions/ + metadata.jsonl
  ```
- Trained 5 single-style LoRAs: old, angry, drunk, happy, mixed
- Config: 3000 steps, lr=1e-4

### Results
| Style | Value Range | Observation |
|-------|------------|-------------|
| Base | [-17.0, 10.5] | Normal |
| old | [-11.3, 10.1] | Almost identical to base |
| angry | [-72.5, 45.3] | **Exploded** — severely overfitted/diverged |
| drunk | [-11.3, 11.2] | Almost identical to base |
| happy | [-23.6, 14.7] | Slight difference |
| mixed | [-14.3, 8.6] | Almost identical to base |

### Why It Failed
**The HumanML3D filtered data is a SUBSET of MDM's pre-training data.** The base model has already been trained on ALL HumanML3D motions, including those labeled "old" or "drunk". Fine-tuning LoRA on data the model already knows teaches it nothing new.

**Analogy**: Asking someone who already read the whole book to re-read one chapter — their knowledge doesn't change.

**Key Lesson**: LoRA needs **external data** (different distribution from pre-training) to be effective. 100STYLE BVH data is genuinely new to the model.

---

## Phase 13: Per-Style 100STYLE Training (v7) — Current

### Design Changes
1. **Per-style directories**: `reconvert_and_check.py` now outputs per-style:
   ```
   style_bvh/zombie/motions/ + metadata.jsonl
   style_bvh/elated/motions/ + metadata.jsonl
   style_bvh/old/motions/ + metadata.jsonl
   style_bvh/depressed/motions/ + metadata.jsonl
   style_bvh/drunk/motions/ + metadata.jsonl
   style_bvh/mixed/motions/ + metadata.jsonl
   ```

2. **Improved hyperparameters**:
   - Steps: 3000 → **5000** (more epochs for small dataset)
   - Learning rate: 1e-4 → **2e-4** (stronger LoRA updates)
   - Cosine LR schedule with 100-step warmup

3. **Multi-LoRA generation**: `generate_and_eval.py` now loads and compares all trained LoRAs against base model in one run

### Training Results (all 6 LoRAs)

| Style | Data Size | Final Loss | Training Time |
|-------|-----------|------------|---------------|
| zombie | 8 motions | 0.170 | ~10 min |
| elated | 8 motions | 0.233 | ~10 min |
| old | 8 motions | 0.122 | ~10 min |
| depressed | 8 motions | 0.215 | ~10 min |
| drunk | 8 motions | 0.244 | ~10 min |
| mixed | 40 motions | 0.172 | ~17 min |

All losses converged smoothly. Lower loss (old: 0.122) suggests the style is closer to HumanML3D distribution; higher loss (drunk: 0.244) suggests more divergence (which is expected for extreme styles).

### Initial Visual Results (zombie)
- **Base vs LoRA-zombie shows visible difference**: LoRA skeleton has lower center of gravity, more compact posture — consistent with zombie style
- Skeleton extra bone issue resolved
- Mild foot sliding present (common in diffusion motion generation, noted as limitation)

### Evaluator Bug Fix
- `generate_and_eval.py` called `evaluator.evaluate()` which didn't exist
- **Fix**: Changed to `evaluator.evaluate_batch()` (the actual method name in `evaluator.py`)

### Awaiting
- Full multi-style generation results (all 6 LoRAs)
- Quantitative evaluation comparison
- Final visualizations

---

## Phase Summary

| Version | Data Source | Normalization | Result |
|---------|-----------|---------------|--------|
| v1 | 100STYLE BVH (identity rotations) | Self-computed | NaN loss |
| v2 | 100STYLE BVH (identity rotations) | Self-computed | Trained but no effect at inference (space mismatch) |
| v3 | 100STYLE BVH (identity rotations) | HumanML3D + clip [-5,5] | Weak effect, 10.4% clipped |
| v4 | HumanML3D filtered by captions | HumanML3D (native) | Trained but base already knows these styles |
| v5 | 100STYLE BVH (real rotations, mixed) | HumanML3D + clip [-5,5] | Slight difference, diluted by mixing 5 styles |
| v6 | HumanML3D filtered (per-style) | HumanML3D (native) | **No effect** — subset of pre-training data |
| **v7** | **100STYLE BVH (per-style)** | **HumanML3D + clip [-5,5]** | **Visible style differences with zombie** |

---

## Remaining Tasks

| Task | Status |
|------|--------|
| Generate + evaluate all 6 LoRAs | In progress |
| Quantitative metrics comparison | Not started |
| MLD + LoRA training (comparison) | Not started |
| Final visualisations for report | Not started |
| Critical Reflective Paper | Not started |
| Progress report slides + script | Done |
