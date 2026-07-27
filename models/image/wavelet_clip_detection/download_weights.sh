#!/bin/bash
# Download pretrained Wavelet-CLIP model weights from MEGA
#
# Due to MEGA download limitations, manual download may be required.
# Follow the instructions below.

WEIGHT_DIR="model_code/weights"
WEIGHT_FILE="$WEIGHT_DIR/clip_wavelet_best.pth"

mkdir -p "$WEIGHT_DIR"

if [ -f "$WEIGHT_FILE" ]; then
    echo "Model weights already exist at: $WEIGHT_FILE"
    exit 0
fi

echo "Wavelet-CLIP pretrained weights not found."
echo ""
echo "Please download manually:"
echo "  1. Visit: https://mega.nz/folder/2BMQ0RJK#h_0M09W5GWXKWZvEE7fxOg"
echo "  2. Download clip_wavelet_best.pth"
echo "  3. Place it at: $WEIGHT_FILE"
echo ""
echo "Or use mega-cmd (install from https://mega.io/cmd):"
echo "  megadl 'https://mega.nz/folder/2BMQ0RJK#h_0M09W5GWXKWZvEE7fxOg' --path /tmp/mega_weights"
echo "  cp /tmp/mega_weights/clip_wavelet_best.pth $WEIGHT_FILE"
echo ""

exit 1
