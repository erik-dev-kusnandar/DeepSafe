#!/bin/bash
# Download pretrained model weights for Vocoder Artifacts
# Place this in models/audio/vocoder_artifacts/

echo "🔽 Downloading Vocoder Artifacts pretrained weights..."
echo "⚠️  Note: This requires ~100MB download"

# Create models directory
mkdir -p models

# Download using gdown (requires pip install gdown)
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Google Drive file ID
FILE_ID="15qOi26czvZddIbKP_SOR8SLQFZK8cf8E"

# Download
gdown $FILE_ID -O models/librifake_pretrained_lambda0.5_epoch_25.pth

if [ -f "models/librifake_pretrained_lambda0.5_epoch_25.pth" ]; then
    echo "✓ Model weights downloaded successfully!"
    echo "  Location: $(pwd)/models/librifake_pretrained_lambda0.5_epoch_25.pth"
else
    echo "❌ Download failed. Please download manually from:"
    echo "   https://drive.google.com/file/d/15qOi26czvZddIbKP_SOR8SLQFZK8cf8E/view"
    exit 1
fi
