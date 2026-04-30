#!/bin/bash
# Example: run the full pipeline on the training machine
#
# Usage:
#   bash scripts/run_pipeline.sh <reference_image> <style_description> <run_name>
#
# Example:
#   bash scripts/run_pipeline.sh /transfer/ref/ink_wash.jpg "Japanese ink wash painting" ink_wash

set -e

REFERENCE="$1"
STYLE="$2"
RUN_NAME="$3"

if [ -z "$REFERENCE" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: bash scripts/run_pipeline.sh <reference_image> <style_description> <run_name>"
    exit 1
fi

echo "=== Step 1: Build index (skip if already exists) ==="
if [ ! -f /transfer/embeddings/wikiart_clip.pt ]; then
    python scripts/build_index.py
else
    echo "Index already exists, skipping."
fi

echo ""
echo "=== Step 2: Run full pipeline ==="
python pipeline.py --reference "$REFERENCE" --style "$STYLE" --run-name "$RUN_NAME"
