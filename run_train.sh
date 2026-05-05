#!/bin/bash
python src/training/train_mdm_lora.py \
  --style_data_dir /transfer/loradataset/style_converted_v2 \
  --output_dir /transfer/loraoutputs/models/style_lora_v5 \
  --max_train_steps 3000
