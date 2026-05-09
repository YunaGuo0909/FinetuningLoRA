#!/bin/bash
# Train v4 LoRA models:
#   - alpha=8 (same as v3)
#   - foot_vel_weight=2.0 (same as v3)
#   - root_stable_weight=1.0 (NEW: penalize root drift during dual contact)
#   - lr=1e-4 (halved from v3's 2e-4, reduce jitter)
#
# IMPORTANT: before running, reconvert BVH data with improved foot contacts:
#   python scripts/reconvert_and_check.py
# The new compute_foot_contacts uses height+velocity (not height-only).
#
# Usage: bash scripts/train_v4_improved.sh

CHECKPOINT_DIR="/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
HUMANML3D_DIR="/transfer/loradataset/humanml3d"
BASE_STYLE_DIR="/transfer/loradataset/style_bvh"
OUTPUT_BASE="/transfer/loraoutputs/models"

STYLES="zombie elated old depressed drunk mixed"

for STYLE in $STYLES; do
    echo "============================================================"
    echo "Training v4 LoRA: $STYLE (alpha=8, lr=1e-4, foot+root penalty)"
    echo "============================================================"

    accelerate launch src/training/train_mdm_lora.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --humanml3d_dir "$HUMANML3D_DIR" \
        --style_data_dir "$BASE_STYLE_DIR/$STYLE" \
        --output_dir "$OUTPUT_BASE/lora_bvh_${STYLE}_v4" \
        --lora_rank 16 \
        --lora_alpha 8 \
        --learning_rate 1e-4 \
        --max_train_steps 5000 \
        --batch_size 64 \
        --lr_scheduler cosine \
        --lr_warmup_steps 100 \
        --checkpointing_steps 1000 \
        --foot_vel_weight 2.0 \
        --root_stable_weight 1.0

    echo "Done: $STYLE"
    echo ""
done

echo "All v4 training complete."
echo "Changes vs v3:"
echo "  - BVH foot contacts: height+velocity (was height-only)"
echo "  - root_stable_weight=1.0 (NEW)"
echo "  - lr=1e-4 (was 2e-4)"
