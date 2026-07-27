#!/bin/bash
# Download pretrained AASIST model weights

echo "Downloading AASIST pretrained weights..."

mkdir -p models

# Download AASIST.pth from GitHub releases or direct link
wget -O models/AASIST.pth \
    "https://github.com/clovaai/aasist/raw/main/weights/AASIST.pth" 2>/dev/null || \
    gdown "1FhIWkisD4Q8VQo8a5SFG%H0tB5R0hUj" -O models/AASIST.pth

if [ -f "models/AASIST.pth" ]; then
    echo "Model weights downloaded successfully!"
    echo "Location: $(pwd)/models/AASIST.pth"
else
    echo "Download failed. Please download manually from:"
    echo "https://github.com/clovaai/aasist"
    exit 1
fi
