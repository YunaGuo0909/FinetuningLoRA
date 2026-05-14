# Motion Style-Adaptive LoRA Fine-Tuning for MDM

A pipeline for fine-tuning the **Motion Diffusion Model (MDM)** with LoRA adapters to generate stylised human motion. Given a style-specific motion capture dataset (e.g. "zombie walk" from 100STYLE), the system trains lightweight LoRA adapters that steer the base model toward that style — even when prompted with generic text like *"a person walking forward"*.

This repo wraps the **official MDM codebase** (Tevet et al. 2023) and injects PEFT-compatible LoRA adapters by replacing the fused `nn.MultiheadAttention` with split `to_q / to_k / to_v / to_out` linear layers.

---

## Project Structure

```
FinetuningLoRA/
├── configs/default.json              # Hyperparameters and /transfer paths
│
├── src/
│   ├── models/
│   │   ├── mdm_official.py           # Official MDM wrapper + SplitQKVAttention + LoRA injection
│   │   ├── mdm.py                    # Custom MDM (legacy, kept for reference)
│   │   └── diffusion.py              # Cosine-schedule Gaussian diffusion (DDPM + DDIM)
│   ├── data/
│   │   ├── humanml_dataset.py        # HumanML3DDataset & StyleMotionDataset
│   │   └── bvh_converter.py          # 100STYLE BVH -> 263-dim with real 6D rotations
│   ├── training/
│   │   └── train_mdm_lora.py         # LoRA fine-tuning with foot + root auxiliary losses
│   ├── evaluation/
│   │   └── evaluator.py              # Diversity and Jitter metrics
│   └── visualization/
│       └── motion_viz.py             # Skeleton animation renderer
│
└── scripts/
    ├── prepare_data.py               # Download/verify datasets and pretrained weights
    ├── reconvert_and_check.py        # Convert 100STYLE BVH per-style with fixed rotations
    ├── generate_and_eval.py          # Multi-LoRA vs base generation + eval + visualisation
    ├── train_v2_foot_penalty.sh      # Batch train with foot velocity penalty
    ├── train_v3_low_alpha.sh         # Batch train with lower alpha (=8)
    ├── diagnose_data.py              # Check feature ranges, NaN/Inf
    └── diagnose_lora.py              # Verify LoRA weights load and affect output
```

---

## Architecture

### Base Model

Official MDM `humanml_trans_enc_512` checkpoint (475 K steps, ~78 MB). Input: `(B, 263, 1, T)` HumanML3D motion features. Conditioning: frozen CLIP ViT-B/32 text embeddings. Prediction target: `x_0` directly (`predict_xstart=True`), not noise.

### LoRA Injection

MDM uses `nn.MultiheadAttention` with a fused `in_proj_weight` of shape `(3D, D)` — PEFT cannot target this. Solution:

1. **`SplitQKVAttention`** copies the fused weights into separate `to_q`, `to_k`, `to_v`, `to_out` linear layers, preserving the forward signature so it works inside `nn.TransformerEncoderLayer` unchanged.
2. **`replace_attention_layers()`** swaps all `MultiheadAttention` modules in the motion transformer, skipping the CLIP encoder (fp16, would conflict with fp32 LoRA layers).
3. **`apply_lora()`** injects rank-16 adapters on `[to_q, to_k, to_v, to_out]` across all 8 transformer layers (~524 K trainable params, 2.8% of the 18.5 M base).

### Training Loss (final config — v12)

```
L = MSE(pred_x0, x0)
  + λ_foot * Σ_{j∈feet,ankles} ||v_j^pred ⊙ c_j^gt||²
  + λ_root * ||v_root_xz^pred ⊙ m_dual||²
```

- `λ_foot = 2.0` — penalises predicted foot velocity when ground-truth contact is active
- `λ_root = 1.0` — penalises root drift when both feet are grounded
- `α = 8`, `rank = 16` (scaling 0.5), `lr = 1e-4`, 5000 steps, batch 64

---

## Setup

Requires Linux, Python 3.11, CUDA 12.1, NVIDIA GPU. All large files live on `/transfer/` (no home quota).

```bash
# 1. Clone this repo and the official MDM
git clone https://github.com/YunaGuo0909/FinetuningLoRA.git && cd FinetuningLoRA
git clone https://github.com/GuyTevet/motion-diffusion-model.git /transfer/mdm_official

# 2. Create venv on /transfer
uv venv /transfer/lora_venv --python 3.11
ln -s /transfer/lora_venv .venv
source .venv/bin/activate
export UV_CACHE_DIR=/transfer/uv_cache
export HF_HOME=/transfer/hf_cache
uv pip install -r requirements.txt
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Download datasets and weights
python scripts/prepare_data.py --verify
```

Expected `/transfer` layout:

```
/transfer/
├── mdm_official/                           # official MDM repo
├── loradataset/
│   ├── humanml3d/                          # HumanML3D (17,126 motions + Mean.npy/Std.npy)
│   ├── 100STYLE/                           # 810 BVH files (Zenodo)
│   └── style_bvh/                          # output of reconvert_and_check.py
│       ├── zombie/  elated/  old/  depressed/  drunk/
│       ├── angry/   chicken/ proud/ heavyset/  bigsteps/
│       ├── stiff/   duckfoot/ highknees/ flapping/ punch/
│       ├── wildarms/ handsinpockets/ handsbetweenlegs/
│       ├── onphoneleft/ penguin/ robot/
│       └── mixed/
├── lorapretrain/
│   └── humanml_trans_enc_512/model000475000.pt
└── loraoutputs/
    ├── models/lora_bvh_<style>_v3/         # trained LoRA checkpoints
    └── eval/multi_style_v6/                # generated motions + GIFs + metrics
```

---

## Usage

### 1 — Convert 100STYLE BVH to HumanML3D features

```bash
python scripts/reconvert_and_check.py
# Output: /transfer/loradataset/style_bvh/<style>/
```

Uses real 6D rotations extracted from BVH Euler angles (not identity placeholders). Prints per-feature-group normalisation ranges to confirm compatibility with HumanML3D's feature space.

### 2 — Train LoRA adapters

```bash
# Train all 21 styles (final config: alpha=8, foot+root penalty, lr=1e-4)
bash scripts/train_v3_low_alpha.sh

# Or train a single style directly
python src/training/train_mdm_lora.py \
    --style_data_dir /transfer/loradataset/style_bvh/zombie \
    --output_dir     /transfer/loraoutputs/models/lora_bvh_zombie_v3 \
    --max_train_steps 5000 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 8 \
    --foot_vel_weight 2.0 \
    --root_stable_weight 1.0
```

Each style takes ~10 min on an RTX 4080.

### 3 — Generate, evaluate, visualise

```bash
LORA_VERSION=v3 python scripts/generate_and_eval.py
# Output: /transfer/loraoutputs/eval/multi_style_v6/
#   evaluation_results.json
#   viz/<idx>_cmp_<style>_<prompt>.gif
```

Generates with **generic prompts** (e.g. *"a person walking forward"*, *"a person stepping sideways"*). The LoRA should change the style regardless of what the prompt says.

---

## Styles Trained (21)

| Strong effect | Moderate effect | Weak/inconsistent |
|---|---|---|
| BigSteps, Chicken, Drunk, Old | WildArms, HandsInPockets | Heavyset, HighKnees, Stiff |
| Angry, DuckFoot, Elated, Zombie | OnPhoneLeft, Proud, Robot | Flapping, Punch, Penguin |
| Depressed | | HandsBetweenLegs |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Diversity** | Average pairwise L2 distance between generated sequences. Higher = more varied output. |
| **Jitter** | Mean joint acceleration magnitude across all frames. Lower = smoother motion. |

Neither metric captures style fidelity directly. High diversity may reflect erratic generation rather than rich stylistic variation. Results are compared against the base model baseline.

---

## Key Engineering Lessons

The path from "code runs" to "LoRA visibly changes output" went through 12 training versions. Core lessons:

1. **NaN loss** — BVH data normalised with HumanML3D stats was 50× out of range. Fix: clip to `[-5, 5]`.
2. **No inference effect** — training in self-computed normalisation space but running inference in HumanML3D space. Fix: always use HumanML3D mean/std.
3. **Garbled output from custom MDM** — subtle architectural mismatches made pretrained weights useless. Fix: use official MDM with attention-layer surgery.
4. **Identity rotation placeholders** — 126 of 263 dims were constants. Fix: extract real rotation matrices from BVH and convert to 6D representation.
5. **HumanML3D-filtered data teaches nothing** — it is a subset of the base model's pretraining. LoRA needs external data (100STYLE).
6. **Mixing styles dilutes the signal** — train one LoRA per style.
7. **`x_0` vs noise prediction** — MDM uses `predict_xstart=True`. Training loss and DDIM sampler both had to target `x_0`, not noise.
8. **Skeleton shrinking** — `alpha/rank = 1.0` was too aggressive. Lowering to `alpha=8` (scaling 0.5) reduced the artefact.
9. **Foot sliding** — post-processing fixes broke root-relative coordinates. Training-level foot velocity and root stability penalties were more effective.

---

## Dependencies

- Python 3.11, PyTorch 2.4.0 + CUDA 12.1
- `peft`, `accelerate` (HuggingFace LoRA stack)
- `open-clip-torch` (text conditioning inside official MDM)
- `matplotlib` (skeleton visualisation)

## Acknowledgements

- [Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model) — Tevet et al., 2023
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) — Guo et al., 2022
- [100STYLE](https://www.ianmaurice.com/100style/) — Mason et al., 2022
- [PEFT](https://github.com/huggingface/peft) — HuggingFace

