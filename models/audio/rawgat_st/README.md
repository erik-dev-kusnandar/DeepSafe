# RawGAT-ST Audio Detection Model

## Overview
This directory contains the integration of the **RawGAT-ST** audio deepfake detection model into DeepSafe.

**Source**: [eurecom-asp/RawGAT-ST-antispoofing](https://github.com/eurecom-asp/RawGAT-ST-antispoofing)

**Paper**: [RawGAT-ST: Raw Graph Attention Neural Network with Spatial-Temporal Feature Learning for Voice Anti-Spoofing](https://arxiv.org/abs/2310.04746)

## Model Architecture
- **Base**: Graph Attention Network on raw waveform with spatial-temporal features
- **Input**: Audio waveform (16kHz, ~4 seconds, 64600 samples)
- **Output**: Binary classification (Bonafide/Spoof)
- **Class mapping**: Class 0 = bonafide (real), Class 1 = spoof (fake)
- **Performance**: State-of-the-art on ASVspoof 2019 and 2021

## Directory Structure
```
models/audio/rawgat_st/
├── Dockerfile          # Container definition
├── app.py              # Flask API wrapper
├── requirements.txt    # Python dependencies
├── download_weights.sh # Weight download script
├── temp_repo/          # Cloned original repository (model code)
├── models/             # Model weights directory
│   └── Best_epoch.pth  # Pretrained weights (copied from repo)
└── README.md           # This file
```

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "model": "RawGAT-ST", "version": "1.0.0", "device": "cpu"}
```

### Prediction
```
POST /predict
Body: {
  "audio_data": "base64_encoded_wav_file",
  "threshold": 0.5
}

Response: {
  "probability": 0.87,
  "prediction": 1,
  "class": "fake",
  "inference_time": 0.12,
  "sample_rate": 16000,
  "audio_duration_seconds": 4.0
}
```

## Testing
```bash
# Build and start the service
docker build -t rawgat-st .
docker run -p 7005:7005 rawgat-st

# Check health
curl http://localhost:7005/health

# Test prediction
python deepsafe_test.py test --media-type audio --input test_samples/sample_audio.wav
```
