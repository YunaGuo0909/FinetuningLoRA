#!/bin/bash
# Train v3 LoRA models: lower lora_alpha=8 (scaling=0.5) + foot velocity penalty.
# Key change vs v2: lora_alpha 16->8 to reduce skeleton shrinking.
# Outputs go to lora_bvh_<style>_v3/ to preserve v1 and v2 checkpoints.
#
# Usage: bash scripts/train_v3_low_alpha.sh

CHECKPOINT_DIR="/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
HUMANML3D_DIR="/transfer/loradataset/humanml3d"
BASE_STYLE_DIR="/transfer/loradataset/style_bvh"
OUTPUT_BASE="/transfer/loraoutputs/models"

STYLES="zombie elated old depressed drunk mixed"
FOOT_VEL_WEIGHT=2.0

for STYLE in $STYLES; do
    echo "============================================================"
    echo "Training v3 LoRA: $STYLE (alpha=8, foot_vel=$FOOT_VEL_WEIGHT)"
    echo "============================================================"

    accelerate launch src/training/train_mdm_lora.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --humanml3d_dir "$HUMANML3D_DIR" \
        --style_data_dir "$BASE_STYLE_DIR/$STYLE" \
        --output_dir "$OUTPUT_BASE/lora_bvh_${STYLE}_v3" \
        --lora_rank 16 \
        --lora_alpha 8 \
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

echo "All v3 training complete."
echo "V1: $OUTPUT_BASE/lora_bvh_<style>/        (alpha=16, no foot penalty)"
echo "V2: $OUTPUT_BASE/lora_bvh_<style>_v2/     (alpha=16, foot_vel=2.0)"
echo "V3: $OUTPUT_BASE/lora_bvh_<style>_v3/     (alpha=8,  foot_vel=2.0)"
