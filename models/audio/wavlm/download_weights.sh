#!/bin/bash
# Download pretrained WavLM deepfake detection model from HuggingFace

echo "Downloading WavLM deepfake detection model..."

mkdir -p models

# The model will be downloaded automatically by HuggingFace transformers
# on first run. This script pre-downloads it to the models directory.

python3 -c "
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import os
os.environ['TRANSFORMERS_CACHE'] = 'models'
print('Downloading feature extractor...')
AutoFeatureExtractor.from_pretrained('0xmola/wavlm-deepfake-audio-forensics')
print('Downloading model...')
AutoModelForAudioClassification.from_pretrained('0xmola/wavlm-deepfake-audio-forensics')
print('Model downloaded successfully!')
"

if [ $? -eq 0 ]; then
    echo "Model downloaded and cached in models/"
else
    echo "Download failed. The model will be downloaded on first API call."
    echo "Alternatively, download manually from:"
    echo "https://huggingface.co/0xmola/wavlm-deepfake-audio-forensics"
fi
