import os
import io
import base64
import time
import logging
import sys
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Temporal Consistency Video Deepfake Detection Service",
    description="Detects deepfakes by analyzing temporal inconsistencies across video frames using optical flow, motion patterns, and face landmark stability.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "temporal_consistency_video_service"
FRAMES_PER_VIDEO = int(os.environ.get("FRAMES_PER_VIDEO", "30"))
FLOW_CONSISTENCY_WEIGHT = float(os.environ.get("FLOW_CONSISTENCY_WEIGHT", "0.35"))
FACE_MOTION_WEIGHT = float(os.environ.get("FACE_MOTION_WEIGHT", "0.35"))
NOISE_WEIGHT = float(os.environ.get("NOISE_WEIGHT", "0.15"))
BLINK_WEIGHT = float(os.environ.get("BLINK_WEIGHT", "0.15"))


class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64 encoded video data string")
    threshold: Optional[float] = Field(
        0.5, ge=0.0, le=1.0,
        description="Classification threshold for final video score",
    )


def extract_frames(
    video_bytes: bytes, num_frames: int
) -> Tuple[List[np.ndarray], int]:
    temp_path = f"/tmp/temp_video_{os.urandom(8).hex()}.mp4"
    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture(temp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return [], 0
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames, total
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def compute_optical_flow_consistency(frames: List[np.ndarray]) -> float:
    if len(frames) < 3:
        return 0.5

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flow_magnitudes = []

    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        flow_magnitudes.append((mean_mag, std_mag))

    if not flow_magnitudes:
        return 0.5

    mean_mags = [m for m, _ in flow_magnitudes]
    std_mags = [s for _, s in flow_magnitudes]

    mag_variance = float(np.var(mean_mags))
    mean_std = float(np.mean(std_mags))

    deepfake_flow_score = min(1.0, mag_variance / 50.0) * 0.5 + min(1.0, mean_std / 15.0) * 0.5
    return float(np.clip(deepfake_flow_score, 0.0, 1.0))


def analyze_face_motion_consistency(frames: List[np.ndarray]) -> float:
    if len(frames) < 3:
        return 0.5

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    if face_cascade.empty():
        return 0.5

    face_centers = []
    face_sizes = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_centers.append((x + w // 2, y + h // 2))
            face_sizes.append((w, h))

    if len(face_centers) < 3:
        return 0.5

    center_diffs = []
    for i in range(1, len(face_centers)):
        dx = face_centers[i][0] - face_centers[i - 1][0]
        dy = face_centers[i][1] - face_centers[i - 1][1]
        center_diffs.append(np.sqrt(dx ** 2 + dy ** 2))

    size_diffs = []
    for i in range(1, len(face_sizes)):
        sw = abs(face_sizes[i][0] - face_sizes[i - 1][0])
        sh = abs(face_sizes[i][1] - face_sizes[i - 1][1])
        size_diffs.append(np.sqrt(sw ** 2 + sh ** 2))

    motion_jerkiness = float(np.var(center_diffs)) if center_diffs else 0.0
    size_jerkiness = float(np.var(size_diffs)) if size_diffs else 0.0

    temporal_score = min(1.0, motion_jerkiness / 100.0) * 0.6 + min(1.0, size_jerkiness / 50.0) * 0.4
    return float(np.clip(temporal_score, 0.0, 1.0))


def analyze_temporal_noise(frames: List[np.ndarray]) -> float:
    if len(frames) < 3:
        return 0.5

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    diffs = []
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        diffs.append(float(np.mean(diff)))

    if not diffs:
        return 0.5

    diff_variance = float(np.var(diffs))
    noise_score = min(1.0, diff_variance / 30.0)
    return float(np.clip(noise_score, 0.0, 1.0))


def analyze_blink_patterns(frames: List[np.ndarray]) -> float:
    if len(frames) < 10:
        return 0.5

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    )
    if face_cascade.empty() or eye_cascade.empty():
        return 0.5

    eye_visible = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15)
            )
            eye_visible.append(len(eyes))
        else:
            eye_visible.append(0)

    if not eye_visible or sum(eye_visible) == 0:
        return 0.5

    blink_changes = sum(
        1 for i in range(1, len(eye_visible)) if eye_visible[i] != eye_visible[i - 1]
    )
    blink_rate = blink_changes / len(eye_visible)

    expected_blink_rate = 0.05
    blink_anomaly = abs(blink_rate - expected_blink_rate) / expected_blink_rate
    blink_score = min(1.0, blink_anomaly / 3.0)
    return float(np.clip(blink_score, 0.0, 1.0))


def process_video_and_predict(video_bytes: bytes, threshold: float) -> Dict[str, Any]:
    frames, total_frames = extract_frames(video_bytes, FRAMES_PER_VIDEO)
    if len(frames) < 3:
        logger.warning(f"Too few frames extracted: {len(frames)}")
        return {
            "probability": 0.5,
            "prediction": 0,
            "class": "real",
            "details": f"Only {len(frames)} frames extracted",
        }

    flow_score = compute_optical_flow_consistency(frames)
    face_motion_score = analyze_face_motion_consistency(frames)
    noise_score = analyze_temporal_noise(frames)
    blink_score = analyze_blink_patterns(frames)

    final_prob = (
        flow_score * FLOW_CONSISTENCY_WEIGHT
        + face_motion_score * FACE_MOTION_WEIGHT
        + noise_score * NOISE_WEIGHT
        + blink_score * BLINK_WEIGHT
    )
    final_prob = float(np.clip(final_prob, 0.0, 1.0))

    final_prediction = 1 if final_prob >= threshold else 0
    final_class_label = "fake" if final_prediction == 1 else "real"

    return {
        "probability": final_prob,
        "prediction": final_prediction,
        "class": final_class_label,
        "inference_time": 0.0,
        "details": {
            "total_frames_in_video": total_frames,
            "frames_analyzed": len(frames),
            "subscores": {
                "optical_flow_inconsistency": round(flow_score, 4),
                "face_motion_jerkiness": round(face_motion_score, 4),
                "temporal_noise_anomaly": round(noise_score, 4),
                "blink_anomaly": round(blink_score, 4),
            },
        },
    }


@app.get("/")
async def root():
    return {
        "service_name": MODEL_NAME_DISPLAY,
        "status": "online",
        "frames_analyzed": FRAMES_PER_VIDEO,
        "weights": {
            "flow_consistency": FLOW_CONSISTENCY_WEIGHT,
            "face_motion": FACE_MOTION_WEIGHT,
            "noise": NOISE_WEIGHT,
            "blink": BLINK_WEIGHT,
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME_DISPLAY,
        "frames_analyzed": FRAMES_PER_VIDEO,
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
            f"Temporal consistency video prediction completed in {result['inference_time']:.2f}s. "
            f"Prob Fake: {result['probability']:.4f}, Class: {result['class']}"
        )
        return result
    except Exception as e:
        logger.exception(f"Error during temporal consistency prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7009))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
