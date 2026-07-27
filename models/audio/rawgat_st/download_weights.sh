#!/bin/bash
# Download pretrained RawGAT-ST model weights
# Note: Weights are included in the GitHub repository

echo "Checking for RawGAT-ST pretrained weights..."

mkdir -p models

WEIGHT_PATH="temp_repo/Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth"

if [ -f "$WEIGHT_PATH" ]; then
    cp "$WEIGHT_PATH" models/Best_epoch.pth
    echo "Model weights copied successfully!"
    echo "Location: $(pwd)/models/Best_epoch.pth"
else
    echo "Weights not found in cloned repo. Attempting to clone and download..."
    git clone https://github.com/eurecom-asp/RawGAT-ST-antispoofing.git temp_repo 2>/dev/null
    if [ -f "$WEIGHT_PATH" ]; then
        cp "$WEIGHT_PATH" models/Best_epoch.pth
        echo "Model weights downloaded successfully!"
    else
        echo "Download failed. Please download manually from:"
        echo "https://github.com/eurecom-asp/RawGAT-ST-antispoofing"
        exit 1
    fi
fi
