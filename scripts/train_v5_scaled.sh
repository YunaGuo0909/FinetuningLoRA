#!/bin/bash
# Train v5 LoRA models:
#   - BVH skeleton scaled to match HumanML3D body proportions (NEW - fixes shrinking)
#   - Improved foot contacts (height + velocity)
#   - alpha=8, foot_vel=2.0, root_stable=1.0, lr=1e-4 (same as v4)
#   - 20 styles (expanded from 5)
#
# IMPORTANT: reconvert BVH data FIRST with the new height-matching converter:
#   python scripts/reconvert_and_check.py
#
# Usage: bash scripts/train_v5_scaled.sh

CHECKPOINT_DIR="/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
HUMANML3D_DIR="/transfer/loradataset/humanml3d"
BASE_STYLE_DIR="/transfer/loradataset/style_bvh"
OUTPUT_BASE="/transfer/loraoutputs/models"

STYLES="zombie elated old depressed drunk angry chicken cat dinosaur heavyset bentknees crouched bigsteps highknees fairysteps flapping karatechop dragleftleg handsinpockets sneaky mixed"

for STYLE in $STYLES; do
    STYLE_DIR="$BASE_STYLE_DIR/$STYLE"
    if [ ! -d "$STYLE_DIR" ]; then
        echo "SKIP: $STYLE (directory not found)"
        continue
    fi

    echo "============================================================"
    echo "Training v5 LoRA: $STYLE"
    echo "============================================================"

    accelerate launch src/training/train_mdm_lora.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --humanml3d_dir "$HUMANML3D_DIR" \
        --style_data_dir "$STYLE_DIR" \
        --output_dir "$OUTPUT_BASE/lora_bvh_${STYLE}_v5" \
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

echo "All v5 training complete."
echo "Key change vs v4: BVH skeleton scaled to match HumanML3D height"
echo "Styles: 20 (was 5)"
