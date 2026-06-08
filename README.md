# DeepSafe: Enterprise-Grade Deepfake Detection Platform

Modular, containerized ensemble platform for deepfake detection in images, video, and audio.

## Architecture

```
                         ┌─────────────────────┐
                         │   DeepSafe API       │
                         │   (port 8003)        │
                         └──────┬──────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                      │
    ┌─────┴──────┐      ┌──────┴──────┐       ┌──────┴──────┐
    │   Image     │      │    Video    │       │    Audio    │
    │  Models     │      │   Models    │       │   Models    │
    └──┬──┬──┬──┬─┘      └──────┬──────┘       └──────┬──────┘
       │  │  │  │               │                      │
       │  │  │  │     cross_efficient_vit     vocoder_artifacts
       │  │  │  │           (7001)                 (7002)
       │  │  │  │
  npr  │  │  │  │
 (5001)│  │  │  │
       │  │  │  │
  yermandy_clip  universalfakedetect
  (5002)         (5004)
       │
  wavelet_clip   spsl    ucf
  (5003)         (5006)  (5007)
```

## Models

| Model | Type | Port | Preload | Description |
| :--- | :--- | :--- | :--- | :--- |
| **NPR Deepfake** | Image | 5001 | ❌ | Neural Pattern Recognition |
| **Yermandy CLIP** | Image | 5002 | ✅ | CLIP-based detection |
| **Wavelet CLIP** | Image | 5003 | ✅ | Wavelet + CLIP hybrid |
| **Universal Fake Detect** | Image | 5004 | ❌ | Generalizable detection |
| **SPSL** | Image | 5006 | ❌ | Self-supervised learning |
| **UCF** | Image | 5007 | ❌ | Unconvolutional features |
| **Cross Efficient ViT** | Video | 7001 | ❌ | Video Vision Transformer |
| **Vocoder Artifacts** | Audio | 7002 | ✅ | Audio artifact detection |

## Prerequisites

- Docker & Docker Compose with Compose V2
- NVIDIA Docker runtime (for GPU support)
- Git LFS (optional, for model weight files)

## Model Weights

Some models require weight files that are **not tracked in git** (see `.gitignore`):

```
models/image/yermandy_clip_detection/model_code/weights/model.ckpt
models/image/wavelet_clip_detection/model_code/weights/clip_wavelet_best.pth
models/audio/vocoder_artifacts/models/librifake_pretrained_lambda0.5_epoch_25.pth
```

Transfer these separately (rsync/scp/USB) to the server before building.

## Quick Start

```bash
# 1. Clone
git clone <your-repo-url>
cd DeepSafe

# 2. (Optional) GPU mode — enabled by default
export USE_GPU=true

# 3. Build and start all services
docker compose build
docker compose up -d

# 4. Check all services are healthy
curl http://localhost:8003/health
```

### GPU / CPU Mode

```bash
# GPU (default) — requires nvidia-container-toolkit
docker compose up -d

# CPU only
USE_GPU=false docker compose up -d
```

### Port Configuration

Override ports via environment variables:

```bash
NPR_PORT=5001 YERMANDY_PORT=5002 WAVELET_PORT=5003 \
UNIVERSAL_PORT=5004 SPSL_PORT=5006 UCF_PORT=5007 \
VIDEO_PORT=7001 AUDIO_PORT=7002 \
docker compose up -d
```

## Configuration

Edit `config/deepsafe_config.json` to register/unregister model endpoints and adjust ensemble settings:

```json
{
  "default_threshold": 0.6,
  "default_ensemble_method": "average",
  "default_api_timeout_seconds": 1200
}
```

### Ensemble Methods

- `average` — mean of all model probabilities (recommended)
- `voting` — majority vote
- `stacking` — meta-learner (requires pre-trained artifacts)

## API

### Predict

```bash
# Base64-encoded image
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "media_type": "image",
    "image_data": "<base64>",
    "threshold": 0.6,
    "ensemble_method": "average"
  }'
```

### Health

```bash
curl http://localhost:8003/health
```

## Integration with XPose

DeepSafe runs alongside the XPose backend. In `xpose-backend/.env`:

```
DEEPSAFE_BASE_URL=http://host.docker.internal:8003
```

The XPose worker sends media files to DeepSafe's `/predict` endpoint and aggregates per-model results.

## Preloading

Models with `PRELOAD_MODEL=true` load their weights at container startup (~5 min on CPU). This avoids cold-start latency on first request but increases startup time.

Models with `PRELOAD_MODEL=false` load weights on first request (faster startup, slower first inference).

## Development

```bash
# Rebuild a single model
docker compose build wavelet_clip_detection

# Restart without rebuilding
docker compose up -d wavelet_clip_detection

# View logs
docker compose logs -f api
docker compose logs -f wavelet_clip_detection

# Stop all
docker compose down
```

## Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| Model returns HTTP 500 | Missing dependency | Check container logs |
| `libtorch_cpu.so` execstack error | Kernel security policy | Fixed in wavelet Dockerfile via ELF patching |
| `pkg_resources` not found | setuptools too new | Pinned to `setuptools<72` |
| CLIP download hangs at runtime | HuggingFace blocked | Pre-downloaded at build time |
| Container exits immediately | OOM or GPU error | Set `USE_GPU=false` or increase memory |
