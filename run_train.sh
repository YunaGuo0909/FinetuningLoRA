#!/bin/bash
# Train multiple single-style LoRAs + one mixed-style LoRA
#
# Usage:
#   bash run_train.sh           # train all
#   bash run_train.sh old       # train single style
#   bash run_train.sh old angry # train specific styles

STYLE_DATA_ROOT="/transfer/loradataset/style_filtered"
OUTPUT_ROOT="/transfer/loraoutputs/models"
STEPS=3000

# If args given, use those styles; otherwise train key styles + mixed
if [ $# -gt 0 ]; then
    STYLES=("$@")
else
    STYLES=("old" "angry" "drunk" "happy" "mixed")
fi

for STYLE in "${STYLES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Training LoRA: ${STYLE}"
    echo "=============================================="

    DATA_DIR="${STYLE_DATA_ROOT}/${STYLE}"
    OUT_DIR="${OUTPUT_ROOT}/lora_${STYLE}"

    if [ ! -d "${DATA_DIR}" ]; then
        echo "  SKIP: ${DATA_DIR} does not exist"
        continue
    fi

    python src/training/train_mdm_lora.py \
        --style_data_dir "${DATA_DIR}" \
        --output_dir "${OUT_DIR}" \
        --max_train_steps ${STEPS}

    echo "  Done: ${OUT_DIR}"
done

echo ""
echo "All training complete!"
