# Motion Style-Adaptive LoRA Fine-Tuning Pipeline

A pipeline for fine-tuning the **Motion Diffusion Model (MDM)** with LoRA adapters to learn stylised human motion generation. Given a style-specific motion capture dataset (e.g. "zombie walk" from 100STYLE), the system trains lightweight LoRA adapters that steer the base model toward generating motions in that style — even when prompted with generic captions like *"a person walking forward"*.

This repo wraps the **official MDM codebase** (Tevet et al. 2023) and injects PEFT-compatible LoRA adapters by replacing the fused `nn.MultiheadAttention` with split `to_q / to_k / to_v / to_out` linear layers.

## Project Structure

```
FinetuningLoRA/
├── configs/default.json              # Hyperparameters and /transfer paths
├── pipeline.py                       # Legacy end-to-end orchestrator (deprecated)
├── run_train.sh                      # Batch trainer: all styles + mixed
├── process0504.md                    # Engineering log (Phase 1-13)
│
├── src/
│   ├── models/
│   │   ├── mdm_official.py           # Official MDM wrapper + SplitQKVAttention + LoRA injection
│   │   ├── mdm.py                    # Custom MDM (legacy, kept for reference only)
│   │   └── diffusion.py              # Cosine-schedule Gaussian diffusion (DDPM + DDIM)
│   ├── data/
│   │   ├── humanml_dataset.py        # HumanML3DDataset & StyleMotionDataset (HML3D-space norm + clip)
│   │   └── bvh_converter.py          # 100STYLE BVH -> 263-dim with real 6D rotations
│   ├── training/
│   │   └── train_mdm_lora.py         # LoRA fine-tuning (predicts x_0, not noise)
│   ├── evaluation/
│   │   └── evaluator.py              # FID, Diversity, Jitter metrics
│   └── visualization/
│       └── motion_viz.py             # Skeleton animation with foot-contact lock + fixed axes
│
└── scripts/
    ├── prepare_data.py               # Download/verify datasets and pretrained weights
    ├── reconvert_and_check.py        # Reconvert 100STYLE BVH per-style with fixed rotations
    ├── filter_style_data.py          # (Failed experiment v6) filter HumanML3D by captions
    ├── convert_100style.py           # Old all-in-one BVH converter
    ├── generate_and_eval.py          # Multi-LoRA vs base generation + eval + viz
    ├── diagnose_data.py              # Sanity-check feature ranges
    └── diagnose_lora.py              # Verify LoRA weights actually load & affect output
```

> A detailed phase-by-phase engineering log of every problem solved (NaN loss, normalisation mismatch, attention layer surgery, x_0 vs noise prediction, etc.) lives in [`process0504.md`](process0504.md).

## Architecture

### Base Model: Official MDM (Motion Diffusion Model)

We load Tevet et al.'s official `humanml_trans_enc_512` checkpoint (475 K steps, ~78 MB) and patch it for LoRA:

- **Input format**: `(B, 263, 1, T)` — HumanML3D motion features (root velocity/height + 21×3 joint positions + 21×6 joint rotations + 22×3 local velocities + 4 foot contacts)
- **Conditioning**: CLIP ViT-B/32 text embeddings (frozen, lives inside the official MDM module)
- **Prediction target**: `predict_xstart=True` — model predicts the clean `x_0`, **not** the noise
- **Trainable params**: ~524 K LoRA params (≈2.8% of the 18.5 M base)
- **SMPL patch**: SMPL body model init is dummied out (only needed for visualisation, not for `hml_vec` features)

### LoRA Injection Strategy

The official MDM uses `nn.MultiheadAttention` with a **fused** `in_proj_weight` of shape `(3D, D)`. PEFT cannot target this. We solve it in two steps:

1. **`SplitQKVAttention`** — drop-in replacement that copies the fused QKV weights into separate `to_q`, `to_k`, `to_v`, `to_out` `nn.Linear` modules while preserving the `nn.MultiheadAttention` forward signature (so it works inside `nn.TransformerEncoderLayer` unchanged).
2. **`replace_attention_layers()`** walks all submodules and swaps `MultiheadAttention -> SplitQKVAttention`, **skipping the CLIP encoder** (which uses fp16 and would conflict with the fp32 LoRA layers).
3. **`apply_lora()`** uses `peft.LoraConfig` to inject rank-16 adapters on `[to_q, to_k, to_v, to_out]` of all 8 transformer layers.

### Diffusion

- Cosine β-schedule with 1000 timesteps
- Training loss: `MSE(pred_x0, x_0)` masked by valid frames (NOT the standard `MSE(pred_noise, noise)`)
- Inference: DDIM (50 steps) — sampler derives noise from the predicted `x_0` and steps backward

## Setup (Training Machine)

The pipeline assumes a Linux machine with all storage on `/transfer/` (no home quota).

```bash
# 1. Clone this repo
git clone https://github.com/YunaGuo0909/FinetuningLoRA.git
cd FinetuningLoRA

# 2. Clone the official MDM repo (used by mdm_official.py)
git clone https://github.com/GuyTevet/motion-diffusion-model.git /transfer/mdm_official

# 3. Create venv on /transfer (avoids home disk quota)
uv venv /transfer/lora_venv --python 3.11
ln -s /transfer/lora_venv .venv
source .venv/bin/activate
export UV_CACHE_DIR=/transfer/uv_cache
export HF_HOME=/transfer/hf_cache          # CLIP cache
export TRANSFORMERS_CACHE=/transfer/hf_cache
uv pip install -r requirements.txt
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Download datasets and pretrained weights
python scripts/prepare_data.py
python scripts/prepare_data.py --verify
```

Expected `/transfer` layout after setup:

```
/transfer/
├── lora_venv/                              -> .venv (symlinked)
├── mdm_official/                           # cloned official MDM repo
├── loradataset/
│   ├── humanml3d/                          # 17,126 motions + Mean.npy/Std.npy (from Kaggle)
│   ├── 100STYLE/                           # 810 BVH files, 100 styles (from Zenodo)
│   ├── style_bvh/                          # output of reconvert_and_check.py
│   │   ├── zombie/{motions/, metadata.jsonl}
│   │   ├── elated/...   old/...   depressed/...   drunk/...
│   │   └── mixed/                          # all 5 styles merged
│   └── style_filtered/                     # (failed v6 experiment, unused)
├── lorapretrain/
│   └── humanml_trans_enc_512/humanml_trans_enc_512/model000475000.pt
├── loraoutputs/
│   ├── models/lora_bvh_<style>/{checkpoint-*,final}
│   └── eval/multi_style/{*.npy, viz/*.gif, evaluation_results.json}
├── uv_cache/
└── hf_cache/                               # CLIP weights
```

## Usage

### Step 1 — Convert 100STYLE BVH to per-style HumanML3D features

`reconvert_and_check.py` uses the **fixed** BVH converter (real 6D rotations, not identity placeholders) and writes per-style directories so each style can be trained as a separate LoRA.

```bash
python scripts/reconvert_and_check.py
# Output: /transfer/loradataset/style_bvh/{zombie,elated,old,depressed,drunk,mixed}/
```

The script also prints per-feature-group normalisation ranges to confirm BVH-converted data is now compatible with HumanML3D's pretrained feature space (rotation features used to span `[-489, 489]`; now `[-5.1, 5.1]` after the fix).

### Step 2 — Train one LoRA per style

```bash
# Train all 6 LoRAs (zombie, elated, old, depressed, drunk, mixed)
bash run_train.sh

# Or train specific styles
bash run_train.sh zombie depressed

# Or call directly
python src/training/train_mdm_lora.py \
    --style_data_dir /transfer/loradataset/style_bvh/zombie \
    --output_dir    /transfer/loraoutputs/models/lora_bvh_zombie \
    --max_train_steps 5000 \
    --learning_rate 2e-4 \
    --lora_rank 16
```

Each run takes ~10 min on an RTX 4080 (8 motions × 5000 steps). The `mixed` run takes ~17 min (40 motions).

### Step 3 — Generate, evaluate, visualise (all LoRAs at once)

```bash
python scripts/generate_and_eval.py
# Output: /transfer/loraoutputs/eval/multi_style/
#   base_motions.npy
#   lora_<style>_motions.npy
#   evaluation_results.json
#   viz/{<idx>_base_*.gif, <idx>_lora_<style>_*.gif, <idx>_cmp_<style>_*.gif}
```

The script generates with **generic prompts** (e.g. *"a person walking forward"*) — this is intentional. The point of LoRA is to make the model *default to a style* even without explicit style words in the prompt. Comparing base vs LoRA on the same generic prompt reveals what the adapter has actually learned.

### Diagnostics

```bash
python scripts/diagnose_data.py    # check feature ranges, NaN/Inf
python scripts/diagnose_lora.py    # verify LoRA weights load and produce different outputs
```

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| FID | Frechet Inception Distance on motion stat features (mean/std/min/max per dim) | Lower is better |
| Diversity | Average pairwise L2 distance of generated motions | Higher is better |
| Jitter | Mean acceleration magnitude on joint positions (smoothness) | Lower is better |

`MotionEvaluator.compare_base_vs_lora()` runs both base and LoRA generations under the same seed and reports the delta.

## Datasets

- **HumanML3D** (Guo et al., 2022) — 17,126 motions with 44,970 captions, 263-dim features. Used as the base model's pretraining data and as a normalisation reference for fine-tuning. Downloaded from Kaggle (`mrriandmstique/humanml3d`, ~5.8 GB) because the official repo doesn't redistribute due to AMASS licensing.
- **100STYLE** (Mason et al., 2022) — 100 locomotion styles, 810 BVH files, ~4 M frames. Used as the **external style data** that LoRA actually learns from. Downloaded directly from Zenodo (`https://zenodo.org/records/8127870`).

We focus on 5 visually distinct styles: **Zombie, Elated, Old, Depressed, Drunk** (8 motions each), plus **Mixed** (all 40).

## Key Engineering Lessons

The path from "code compiles" to "LoRA visibly changes output" took 7 versions. Documented in `process0504.md`:

1. **NaN loss (v1)** — BVH-converted data normalised with HumanML3D's mean/std produced values 50× too large. Fix: clip to `[-5, 5]`.
2. **No effect at inference (v2)** — using self-computed mean/std meant LoRA learned in one space while diffusion ran in another. Fix: always normalise with HumanML3D stats so LoRA shares the base model's input space.
3. **Garbled output from custom MDM** — subtle architectural mismatches (post-norm vs pre-norm, conditioning token layout) made pretrained weights useless. Fix: switch to official MDM with attention-layer surgery.
4. **Identity rotation placeholders** — the original BVH converter filled the 126 rotation dimensions with `[1,0,0,1,0,0]` constants, making 50% of dims totally unlike HumanML3D. Fix: extract real local rotation matrices from BVH and convert via `rotation_matrix_to_6d()`.
5. **HumanML3D-filtered data teaches LoRA nothing (v6)** — that data is a subset of the base model's pretraining, so there's no new signal. LoRA needs **external** data (100STYLE BVH).
6. **Mixing 5 styles dilutes the signal (v5)** — train one LoRA per style; use a `mixed` LoRA only as an ablation baseline.
7. **`x_0` vs noise prediction** — official MDM uses `predict_xstart=True`. Both training loss and DDIM sampler had to be rewritten to predict `x_0` directly.

## Key Dependencies

- PyTorch 2.4.0 + CUDA 12.1 (newer torch needs newer CUDA driver than the lab machine has)
- `diffusers`, `accelerate`, `peft` (Hugging Face LoRA stack)
- `open-clip-torch` (text conditioning, used inside official MDM)
- `matplotlib` (skeleton visualisation)

## Acknowledgements

- [Motion Diffusion Model (MDM)](https://github.com/GuyTevet/motion-diffusion-model) — Tevet et al., 2023
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) — Guo et al., 2022
- [100STYLE](https://www.ianmaurice.com/100style/) — Mason et al., 2022
- [PEFT](https://github.com/huggingface/peft) — Hugging Face LoRA implementation

## Coursework

Bournemouth University Level 7 — *Generative AI for Media* (Hammadi Nait-Charif). Deadline 2026-05-15. See [`Brief.txt`](Brief.txt) for the full assignment.
