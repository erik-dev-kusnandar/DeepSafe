# AASIST Audio Deepfake Detection Model

## Overview
This directory contains the integration of the **AASIST** (Anti-Spoofing with Integrated Spectro-Temporal Graph Attention Networks) audio deepfake detection model into DeepSafe.

**Source**: [clovaai/aasist](https://github.com/clovaai/aasist)

**Paper**: [AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (ICASSP 2022)](https://ieeexplore.ieee.org/document/9747513)

## Model Architecture
- **Base**: Graph Attention Network (GAT) with spectro-temporal features
- **Input**: Audio waveform (16kHz, 64600 samples ≈ 4 seconds)
- **Output**: Binary classification (Real/Fake)
- **Performance**: 0.83% HTER on ASVspoof 2021

## Downloading Model Weights
Due to file size, model weights are NOT included in the Docker image by default.

You must download them separately:

```bash
# Option 1: Run the download script
./download_weights.sh

# Option 2: Manual download
# Visit: https://github.com/clovaai/aasist
# Download AASIST.pth and place in: models/audio/aasist/models/AASIST.pth
```

## Directory Structure
```
models/audio/aasist/
├── Dockerfile          # Container definition
├── app.py              # Flask API wrapper
├── requirements.txt    # Python dependencies
├── temp_repo/          # Cloned AASIST repository (model code)
├── models/             # Model weights directory
│   └── AASIST.pth      # Pretrained weights (download separately)
└── README.md           # This file
```

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "model": "AASIST", "version": "1.0.0", "device": "cpu"}
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
  "audio_duration_seconds": 4.04
}
```

## Testing
```bash
# Start the service
docker-compose up -d aasist

# Check health
curl http://localhost:7003/health

# Test prediction (requires sample audio)
python deepsafe_test.py test --media-type audio --input test_samples/sample_audio.wav
```
