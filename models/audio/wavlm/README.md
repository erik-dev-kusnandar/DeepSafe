# WavLM Audio Deepfake Detection Model

## Overview
This directory contains the integration of the **WavLM** audio deepfake detection model into DeepSafe.

**Model**: [0xmola/wavlm-deepfake-audio-forensics](https://huggingface.co/0xmola/wavlm-deepfake-audio-forensics)

## Model Architecture
- **Base**: WavLM (pretrained speech representation model) with classification head
- **Input**: Audio waveform (16kHz, mono)
- **Output**: Binary classification (Real/Fake) with softmax probabilities
- **Model Size**: ~360MB
- **Framework**: HuggingFace Transformers (auto-downloaded on first run)

## Downloading Model Weights
Unlike other models, the WavLM weights are hosted on HuggingFace and downloaded automatically by `transformers` on first run. No manual download is needed.

To pre-download into a local cache:
```bash
bash download_weights.sh
```

## Directory Structure
```
models/audio/wavlm/
├── Dockerfile          # Container definition
├── app.py              # Flask API wrapper
├── requirements.txt    # Python dependencies
├── download_weights.sh # Optional pre-download script
└── README.md           # This file
```

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "model": "WavLM", "version": "1.0.0", "device": "cpu"}
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
  "inference_time": 0.34,
  "sample_rate": 16000,
  "audio_duration_seconds": 5.0
}
```

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `MODEL_PORT` | `7006` | Port the Flask API listens on |
| `TRANSFORMERS_CACHE` | `/app/models` | Where HuggingFace models are cached |

## Testing
```bash
# Start the service
docker-compose up -d wavlm

# Check health
curl http://localhost:7006/health

# Test prediction (requires sample audio)
python deepsafe_test.py test --media-type audio --input test_samples/sample_audio.wav
```
