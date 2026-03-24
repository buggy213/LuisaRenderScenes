#!/bin/bash
set -euo pipefail

BASE_URL="https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes"
OUT_DIR="$(dirname "$0")/scenes"

SCENES=(
    bathroom
    bathroom2
    bathroom-interior
    bedroom
    camera
    classroom
    dining-room
    glass-of-water
    living-room-2
    lone-monk
    staircase
    staircase2
)

mkdir -p "$OUT_DIR"

for scene in "${SCENES[@]}"; do
    zip_file="$OUT_DIR/$scene.zip"
    echo "Downloading $scene..."
    curl -L -o "$zip_file" "$BASE_URL/$scene.zip"
    echo "Extracting $scene..."
    # dining-room zip lacks a top-level folder, so extract into one
    if [[ "$scene" == "dining-room" ]]; then
        mkdir -p "$OUT_DIR/$scene"
        unzip -o -q "$zip_file" -d "$OUT_DIR/$scene"
    else
        unzip -o -q "$zip_file" -d "$OUT_DIR"
    fi
    rm "$zip_file"
done

# Clean up macOS junk
find "$OUT_DIR" -name '__MACOSX' -type d -exec rm -rf {} + 2>/dev/null || true
find "$OUT_DIR" -name '.DS_Store' -delete 2>/dev/null || true

echo "Done. All scenes downloaded to $OUT_DIR"
