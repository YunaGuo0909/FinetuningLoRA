#!/bin/bash
# Train v2 LoRA models with foot velocity penalty.
# Outputs go to lora_bvh_<style>_v2/ to preserve existing v1 checkpoints.
#
# Usage: bash scripts/train_v2_foot_penalty.sh

CHECKPOINT_DIR="/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
HUMANML3D_DIR="/transfer/loradataset/humanml3d"
BASE_STYLE_DIR="/transfer/loradataset/style_bvh"
OUTPUT_BASE="/transfer/loraoutputs/models"

STYLES="zombie elated old depressed drunk mixed"
FOOT_VEL_WEIGHT=2.0

for STYLE in $STYLES; do
    echo "============================================================"
    echo "Training v2 LoRA: $STYLE (foot_vel_weight=$FOOT_VEL_WEIGHT)"
    echo "============================================================"

    accelerate launch src/training/train_mdm_lora.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --humanml3d_dir "$HUMANML3D_DIR" \
        --style_data_dir "$BASE_STYLE_DIR/$STYLE" \
        --output_dir "$OUTPUT_BASE/lora_bvh_${STYLE}_v2" \
        --lora_rank 16 \
        --lora_alpha 16 \
        --learning_rate 2e-4 \
        --max_train_steps 5000 \
        --batch_size 64 \
        --lr_scheduler cosine \
        --lr_warmup_steps 100 \
        --checkpointing_steps 1000 \
        --foot_vel_weight $FOOT_VEL_WEIGHT

    echo "Done: $STYLE"
    echo ""
done

echo "All v2 training complete."
echo "V1 models preserved in: $OUTPUT_BASE/lora_bvh_<style>/"
echo "V2 models saved to:     $OUTPUT_BASE/lora_bvh_<style>_v2/"
