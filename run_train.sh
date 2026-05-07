#!/bin/bash
# Train single-style LoRAs using 100STYLE BVH data
#
# Usage:
#   bash run_train.sh           # train all styles + mixed
#   bash run_train.sh zombie    # train single style
#   bash run_train.sh zombie depressed  # train specific styles

STYLE_DATA_ROOT="/transfer/loradataset/style_bvh"
OUTPUT_ROOT="/transfer/loraoutputs/models"
STEPS=5000
LR=2e-4

if [ $# -gt 0 ]; then
    STYLES=("$@")
else
    STYLES=("zombie" "elated" "old" "depressed" "drunk" "mixed")
fi

for STYLE in "${STYLES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Training LoRA: ${STYLE} (steps=${STEPS}, lr=${LR})"
    echo "=============================================="

    DATA_DIR="${STYLE_DATA_ROOT}/${STYLE}"
    OUT_DIR="${OUTPUT_ROOT}/lora_bvh_${STYLE}"

    if [ ! -d "${DATA_DIR}" ]; then
        echo "  SKIP: ${DATA_DIR} does not exist"
        echo "  Run first: python scripts/reconvert_and_check.py"
        continue
    fi

    python src/training/train_mdm_lora.py \
        --style_data_dir "${DATA_DIR}" \
        --output_dir "${OUT_DIR}" \
        --max_train_steps ${STEPS} \
        --learning_rate ${LR}

    echo "  Done: ${OUT_DIR}"
done

echo ""
echo "All training complete!"
