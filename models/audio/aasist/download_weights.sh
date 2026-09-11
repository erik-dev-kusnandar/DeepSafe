#!/bin/bash
# Download pretrained AASIST model weights

echo "Downloading AASIST pretrained weights..."

mkdir -p models

# Download AASIST.pth from GitHub (path: models/weights/AASIST.pth).
# Also verify the file is non-empty (a failed wget leaves a 0-byte file).
if wget -O models/AASIST.pth \
    "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth"; then
    size=$(stat -c%s models/AASIST.pth 2>/dev/null || stat -f%z models/AASIST.pth)
    if [ -n "$size" ] && [ "$size" -gt 1000000 ]; then
        echo "Model weights downloaded successfully!"
        echo "Location: $(pwd)/models/AASIST.pth ($((size / 1024 / 1024)) MB)"
        exit 0
    fi
fi

echo "Download failed. Please download manually from:"
echo "https://github.com/clovaai/aasist/tree/main/models/weights"
exit 1
