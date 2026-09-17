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


def detect_face_boxes(
    frame_rgb: np.ndarray, min_size: int = 40
) -> List[Tuple[int, int, int, int]]:
    detector = _get_face_detector()
    if detector.empty():
        return []
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    scale = 2 if gray.shape[1] < 900 else 1
    boxes = []
    if scale > 1:
        up = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale))
        dets = detector.detectMultiScale(
            up, scaleFactor=1.1, minNeighbors=6, minSize=(min_size, min_size)
        )
        for (x, y, w, h) in dets:
            boxes.append((x // scale, y // scale, w // scale, h // scale))
    else:
        dets = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_size, min_size)
        )
        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in dets]
    return boxes


_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
    return _face_detector


def crop_or_full_frame(
    frame_rgb: np.ndarray, max_dim: int = 512
) -> Tuple[np.ndarray, bool]:
    """Return the largest detected face crop (with margin) for model inference.

    Falls back to the full frame when no face is detected so non-face video
    still gets processed. ``has_face`` tells callers whether a face was used.
    """
    boxes = detect_face_boxes(frame_rgb)
    if not boxes:
        return frame_rgb, False
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    fx, fy = 0.35, 0.35
    x0 = max(0, int(x - w * fx))
    y0 = max(0, int(y - h * fy))
    x1 = min(frame_rgb.shape[1], int(x + w + w * fx))
    y1 = min(frame_rgb.shape[0], int(y + h + h * fy))
    crop = frame_rgb[y0:y1, x0:x1]
    hc, wc = crop.shape[:2]
    if max(hc, wc) > max_dim:
        r = max_dim / max(hc, wc)
        crop = cv2.resize(crop, (int(wc * r), int(hc * r)), interpolation=cv2.INTER_AREA)
    return crop, True


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
        any_face = any(len(detect_face_boxes(f)) > 0 for f in frames_rgb)
        if not any_face:
            logger.info("No faces detected in video, returning default.")
            return {
                "probability": 0.5,
                "prediction": 0,
                "class": "real",
                "details": "No faces detected",
            }

    frames_with_faces = 0
    frame_b64_list = []
    for f in frames_rgb:
        # Full frame (essential for full AI generative video artifacts like Kling/Sora)
        h_f, w_f = f.shape[:2]
        max_d = 512
        full_f = f
        if max(h_f, w_f) > max_d:
            r_scale = max_d / max(h_f, w_f)
            full_f = cv2.resize(f, (int(w_f * r_scale), int(h_f * r_scale)), interpolation=cv2.INTER_AREA)
        frame_b64_list.append(frame_to_base64(full_f))

        # Face crop (if face detected)
        crop_f, has_face = crop_or_full_frame(f)
        if has_face:
            frames_with_faces += 1
            frame_b64_list.append(frame_to_base64(crop_f))

    logger.info(
        f"Frames analyzed: {len(frames_rgb)}, with cropped faces: {frames_with_faces}/{len(frames_rgb)}"
    )

    per_model_scores: Dict[str, List[float]] = {m: [] for m in IMAGE_MODEL_ENDPOINTS}

    with ThreadPoolExecutor(max_workers=max(MAX_WORKERS * 2, 16)) as executor:
        future_map = {
            executor.submit(
                query_image_model, m, url, frame_b64, input_threshold
            ): m
            for frame_b64 in frame_b64_list
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
            "frames_with_faces": frames_with_faces,
            "face_crop_applied": frames_with_faces > 0,
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
