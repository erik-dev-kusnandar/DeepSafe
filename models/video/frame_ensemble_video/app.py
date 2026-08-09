import os
import io
import sys
import base64
import time
import logging
import gc
import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Frame Ensemble Video Deepfake Detection Service",
    description="Aggregates predictions from image models across video frames for video-level deepfake detection.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "frame_ensemble_video_service"

IMAGE_MODEL_ENDPOINTS = {
    "npr_deepfakedetection": os.environ.get(
        "NPR_ENDPOINT", "http://npr_deepfakedetection:5001/predict"
    ),
    "yermandy_clip_detection": os.environ.get(
        "YERMANDY_ENDPOINT", "http://yermandy_clip_detection:5002/predict"
    ),
    "wavelet_clip_detection": os.environ.get(
        "WAVELET_ENDPOINT", "http://wavelet_clip_detection:5003/predict"
    ),
    "universalfakedetect": os.environ.get(
        "UNIVERSAL_ENDPOINT", "http://universalfakedetect:5004/predict"
    ),
    "spsl_deepfake_detection": os.environ.get(
        "SPSL_ENDPOINT", "http://spsl_deepfake_detection:5006/predict"
    ),
    "ucf_deepfake_detection": os.environ.get(
        "UCF_ENDPOINT", "http://ucf_deepfake_detection:5007/predict"
    ),
}

FRAMES_PER_VIDEO = int(os.environ.get("FRAMES_PER_VIDEO", "15"))
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "300"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
FACE_REQUIRED = os.environ.get("FACE_REQUIRED", "false").lower() == "true"


class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64 encoded video data string")
    threshold: Optional[float] = Field(
        0.5, ge=0.0, le=1.0,
        description="Classification threshold for final video score",
    )


def extract_frames_from_video_bytes(
    video_bytes: bytes, num_frames_to_sample: int
) -> List[np.ndarray]:
    temp_video_path = f"/tmp/temp_video_{os.urandom(8).hex()}.mp4"
    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        frames = []
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open temporary video file: {temp_video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return frames

        frame_indices = np.linspace(
            0, total_frames - 1, num_frames_to_sample, dtype=int
        )

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

        cap.release()
        return frames
    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError as e:
                logger.warning(f"Could not remove temp video file: {e}")


def detect_faces(frame_rgb: np.ndarray) -> bool:
    face_cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(face_cascade_path)
    if face_detector.empty():
        return True
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80)
    )
    return len(faces) > 0


def frame_to_base64(frame_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def query_image_model(
    model_name: str, endpoint_url: str, frame_b64: str, threshold: float
) -> Tuple[str, Optional[float], Optional[int], Optional[str]]:
    try:
        payload = {"image_data": frame_b64, "threshold": threshold}
        resp = requests.post(endpoint_url, json=payload, timeout=MODEL_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        prob = result.get("probability")
        pred = result.get("prediction")
        label = result.get("class")
        return model_name, prob, pred, label
    except Exception as e:
        logger.warning(f"Frame query to '{model_name}' failed: {e}")
        return model_name, None, None, None


def process_video_and_predict(
    video_bytes: bytes, input_threshold: float
) -> Dict[str, Any]:
    frames_rgb = extract_frames_from_video_bytes(video_bytes, FRAMES_PER_VIDEO)
    if not frames_rgb:
        logger.warning("No frames extracted from video.")
        return {
            "probability": 0.5,
            "prediction": 0,
            "class": "real",
            "details": "No frames extracted",
        }

    if FACE_REQUIRED:
        any_face = any(detect_faces(f) for f in frames_rgb)
        if not any_face:
            logger.info("No faces detected in video, returning default.")
            return {
                "probability": 0.5,
                "prediction": 0,
                "class": "real",
                "details": "No faces detected",
            }

    frame_b64_list = [frame_to_base64(f) for f in frames_rgb]

    per_model_scores: Dict[str, List[float]] = {m: [] for m in IMAGE_MODEL_ENDPOINTS}

    for frame_idx, frame_b64 in enumerate(frame_b64_list):
        frame_results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    query_image_model, m, url, frame_b64, input_threshold
                ): m
                for m, url in IMAGE_MODEL_ENDPOINTS.items()
            }
            for future in as_completed(future_map):
                m_name, prob, pred, label = future.result()
                if prob is not None:
                    per_model_scores[m_name].append(prob)

    per_model_mean: Dict[str, float] = {}
    per_model_std: Dict[str, float] = {}
    for m_name, scores in per_model_scores.items():
        if scores:
            per_model_mean[m_name] = float(np.mean(scores))
            per_model_std[m_name] = float(np.std(scores)) if len(scores) > 1 else 0.0
        else:
            per_model_mean[m_name] = 0.5
            per_model_std[m_name] = 0.0

    ensemble_probs = list(per_model_mean.values())
    if not ensemble_probs:
        return {
            "probability": 0.5,
            "prediction": 0,
            "class": "real",
            "details": "No model results",
        }

    final_prob = float(np.mean(ensemble_probs))
    final_prediction = 1 if final_prob >= input_threshold else 0
    final_class_label = "fake" if final_prediction == 1 else "real"

    return {
        "probability": final_prob,
        "prediction": final_prediction,
        "class": final_class_label,
        "inference_time": 0.0,
        "details": {
            "frame_count": len(frames_rgb),
            "frames_with_faces": None,
            "per_model_frame_averages": {
                m: {
                    "mean_probability": round(per_model_mean[m], 4),
                    "std_probability": round(per_model_std[m], 4),
                    "frames_processed": len(per_model_scores[m]),
                }
                for m in sorted(per_model_mean.keys())
            },
        },
    }


@app.get("/")
async def root():
    return {
        "service_name": MODEL_NAME_DISPLAY,
        "status": "online",
        "image_models_configured": list(IMAGE_MODEL_ENDPOINTS.keys()),
        "frames_per_video": FRAMES_PER_VIDEO,
    }


@app.get("/health")
async def health():
    all_ok = True
    model_statuses = {}
    for m_name, url in IMAGE_MODEL_ENDPOINTS.items():
        try:
            health_url = url.replace("/predict", "/health")
            resp = requests.get(health_url, timeout=5)
            model_statuses[m_name] = "healthy" if resp.ok else "unhealthy"
            if not resp.ok:
                all_ok = False
        except Exception:
            model_statuses[m_name] = "unreachable"
            all_ok = False

    return {
        "status": "healthy" if all_ok else "degraded",
        "model_name": MODEL_NAME_DISPLAY,
        "image_models_status": model_statuses,
        "frames_per_video": FRAMES_PER_VIDEO,
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
            f"Frame ensemble video prediction completed in {result['inference_time']:.2f}s. "
            f"Prob Fake: {result['probability']:.4f}, Class: {result['class']}"
        )
        return result
    except Exception as e:
        logger.exception(f"Error during frame ensemble video prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {e}",
        )


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7008))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
