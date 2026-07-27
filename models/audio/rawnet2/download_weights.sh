#!/bin/bash
# Download pretrained RawNet2 model weights from ASVspoof 2021

echo "Downloading RawNet2 pretrained weights..."

mkdir -p models

wget -O models/pre_trained_DF_RawNet2.zip \
    "https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip"

if [ -f "models/pre_trained_DF_RawNet2.zip" ]; then
    unzip -o models/pre_trained_DF_RawNet2.zip -d models/
    rm models/pre_trained_DF_RawNet2.zip
    echo "Model weights downloaded and extracted successfully!"
else
    echo "Download failed. Please download manually from:"
    echo "https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip"
    exit 1
fi
