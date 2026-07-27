# RawNet2 Audio Deepfake Detection Model

## Overview
This directory contains the integration of the **RawNet2** audio deepfake detection model into DeepSafe.

**Source**: [asvspoof-challenge/2021 - Baseline-RawNet2](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)

**Paper**: [End-to-End Anti-Spoofing with RawNet2 (ICASSP 2022)](https://ieeexplore.ieee.org/document/9746865)

## Model Architecture
- **Base**: RawNet2 with GRU-based sequence modeling
- **Input**: Audio waveform (16kHz, 64600 samples ~4 seconds)
- **Output**: Binary classification (Real/Fake)
- **Performance**: Baseline for ASVspoof 2021 LA task

## Downloading Model Weights
Model weights are NOT included in the Docker image by default.

```bash
./download_weights.sh
```

Or download manually from:
https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip

Extract to: `models/audio/rawnet2/models/pre_trained_DF_RawNet2.pth`

## Directory Structure
```
models/audio/rawnet2/
├── Dockerfile          # Container definition
├── app.py              # Flask API wrapper
├── requirements.txt    # Python dependencies
├── download_weights.sh # Weight download script
├── models/             # Model weights directory
│   └── pre_trained_DF_RawNet2.pth
└── README.md           # This file
```

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "model": "RawNet2", "version": "1.0.0", "device": "cpu"}
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
  "audio_duration_seconds": 4.0375
}
```

## Testing
```bash
# Build and start the service
docker build -t rawnet2-service .
docker run -d -p 7004:7004 -v $(pwd)/models:/app/models rawnet2-service

# Check health
curl http://localhost:7004/health

# Test prediction
python deepsafe_test.py test --media-type audio --input test_samples/sample_audio.wav
```
