import os
import io
import base64
import time
import logging
import sys
import threading
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import open_clip
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UnivFD (UniversalFakeDetect) Video Deepfake Detection Service",
    description="UnivFD: CLIP ViT-L/14 + linear head (fc_weights.pth) deepfake image detector, averaged over sampled video frames. Higher score = FAKE.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "univfd_video_service"
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.55"))
FRAMES_PER_VIDEO = int(os.environ.get("FRAMES_PER_VIDEO", "15"))
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"

FC_WEIGHTS_PATH = os.environ.get(
    "FC_WEIGHTS_PATH", "/app/weights/fc_weights.pth"
)
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "ViT-L-14")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")
CLIP_FEAT_DIM = 768

IMAGENET_STATS = {
    "mean": (0.48145466, 0.4578275, 0.40821073),
    "std": (0.26862954, 0.26130258, 0.27577711),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_lock = threading.Lock()
_model: Optional[nn.Module] = None
_fc: Optional[nn.Linear] = None
_transform: Optional[T.Compose] = None
_model_loaded = False


class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64 encoded video data string")
    threshold: float = Field(
        DEFAULT_THRESHOLD, ge=0.0, le=1.0,
        description="Classification threshold for final video score",
    )


def _ensure_loaded() -> None:
    global _model, _fc, _transform, _model_loaded
    if _model_loaded:
        return
    with _model_lock:
        if _model_loaded:
            return
        logger.info(
            f"Loading open_clip '{CLIP_MODEL_NAME}' (pretrained='{CLIP_PRETRAINED}') on {DEVICE.type}..."
        )
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model.eval()

        logger.info(f"Loading UnivFD linear head from {FC_WEIGHTS_PATH} ...")
        fc = nn.Linear(CLIP_FEAT_DIM, 1)
        fc.load_state_dict(torch.load(FC_WEIGHTS_PATH, map_location="cpu"))
        fc.eval()
        if DEVICE.type == "cuda":
            model = model.to(DEVICE)
            fc = fc.to(DEVICE)
            logger.info("UnivFD model moved to CUDA")

        # Official repo eval transform: CenterCrop(224) only (no prior resize),
        # CLIP normalization. Do NOT use open_clip's preprocess (it resizes first).
        _transform = T.Compose(
            [
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMAGENET_STATS["mean"], IMAGENET_STATS["std"]),
            ]
        )
        _model, _fc = model, fc
        _model_loaded = True
        logger.info("UnivFD model loaded.")


def _extract_frames(video_bytes: bytes) -> (List[np.ndarray], int):
    temp_path = f"/tmp/temp_video_{os.urandom(8).hex()}.mp4"
    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture(temp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return [], 0
        indices = np.linspace(0, total - 1, min(FRAMES_PER_VIDEO, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames, total
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _score_frame(frame_rgb: np.ndarray) -> float:
    x = _transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = _model.encode_image(x)  # projected, un-normalized 768-dim
        logit = _fc(feats)
        p = torch.sigmoid(logit).squeeze().item()
    return p


def process_video_and_predict(video_bytes: bytes, threshold: float) -> Dict[str, Any]:
    _ensure_loaded()
    frames, total_frames = _extract_frames(video_bytes)
    if not frames:
        logger.warning("No frames extracted from video.")
        return {
            "probability": 0.5,
            "prediction": 0,
            "class": "real",
            "inference_time": 0.0,
            "details": {"total_frames_in_video": 0, "frames_analyzed": 0},
        }

    scores = [_score_frame(fr) for fr in frames]
    prob = float(np.mean(scores))
    prediction = 1 if prob >= threshold else 0
    label = "fake" if prediction == 1 else "real"

    return {
        "probability": prob,
        "prediction": prediction,
        "class": label,
        "inference_time": 0.0,
        "details": {
            "total_frames_in_video": total_frames,
            "frames_analyzed": len(frames),
            "frame_scores": [round(s, 4) for s in scores],
            "clip_model": CLIP_MODEL_NAME,
        },
    }


@app.on_event("startup")
async def startup_event_handler():
    if PRELOAD_MODEL:
        try:
            _ensure_loaded()
        except Exception as e:
            logger.error(f"Preload failed: {e}", exc_info=True)


@app.get("/")
async def root():
    return {
        "service_name": MODEL_NAME_DISPLAY,
        "status": "online",
        "frames_analyzed": FRAMES_PER_VIDEO,
        "clip_model": CLIP_MODEL_NAME,
        "model_loaded": _model_loaded,
    }


@app.get("/health")
async def health():
    model_ok = os.path.exists(FC_WEIGHTS_PATH)
    status = "healthy"
    message = "Service healthy."
    if not model_ok:
        status = "error_missing_files"
        message = f"Linear head weights not found at {FC_WEIGHTS_PATH}."
    elif not _model_loaded and not PRELOAD_MODEL:
        status = "degraded_not_loaded"
        message = "Ready for lazy loading."
    return {
        "status": status,
        "model_name": MODEL_NAME_DISPLAY,
        "weights_found": model_ok,
        "model_currently_loaded": _model_loaded,
        "message": message,
    }


@app.post("/predict")
async def predict_video(input_data: VideoInput):
    req_start_time = time.time()
    try:
        video_bytes = base64.b64decode(input_data.video_data)
        result = process_video_and_predict(video_bytes, input_data.threshold)
        result["inference_time"] = time.time() - req_start_time
        result["model"] = MODEL_NAME_DISPLAY
        logger.info(
            f"UnivFD video prediction completed in {result['inference_time']:.2f}s. "
            f"Prob Fake: {result['probability']:.4f}, Class: {result['class']}"
        )
        return result
    except Exception as e:
        logger.exception(f"Error during UnivFD prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7011))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)