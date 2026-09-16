import os
import io
import base64
import time
import logging
import sys
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import lip_core

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LipForensics Video Deepfake Detection Service",
    description="LipForensics: temporal lip-motion anomaly detector (3D frontend + ResNet18 + MS-TCN on 96x96 mouth crops). Higher score = FAKE.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "lipforensics_video_service"
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.55"))


class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64 encoded video data string")
    threshold: float = Field(
        DEFAULT_THRESHOLD, ge=0.0, le=1.0,
        description="Classification threshold for final video score",
    )


def process_video_and_predict(video_bytes: bytes, threshold: float) -> dict:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_name = tmp.name
    try:
        tmp.write(video_bytes)
        tmp.close()
        result = lip_core.run_inference(tmp_name)
    except ValueError as ve:
        err_str = str(ve)
        if any(msg in err_str for msg in ["No faces detected", "No face tracks found", "No frames could be read"]):
            logger.warning(f"LipForensics note: {err_str}")
            return {
                "probability": 0.5,
                "prediction": 0,
                "class": "real",
                "inference_time": 0.0,
                "details": {
                    "total_frames_in_video": 0,
                    "frames_analyzed": 0,
                    "clips_evaluated": 0,
                    "note": err_str,
                    "no_faces_detected": True,
                },
            }
        raise
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass

    prob = float(result["probability"])
    prediction = 1 if prob >= threshold else 0
    label = "fake" if prediction == 1 else "real"
    return {
        "probability": prob,
        "prediction": prediction,
        "class": label,
        "inference_time": 0.0,
        "details": result["details"],
    }


@app.get("/")
async def root():
    return {
        "service_name": MODEL_NAME_DISPLAY,
        "status": "online",
        "max_frame_cap": lip_core.MAX_FRAMES,
        "clip_size": lip_core.FRAMES_PER_CLIP,
        "threads": lip_core.NUM_THREADS,
        "default_threshold": DEFAULT_THRESHOLD,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME_DISPLAY,
        "max_frame_cap": lip_core.MAX_FRAMES,
        "clip_size": lip_core.FRAMES_PER_CLIP,
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
            f"LipForensics video prediction completed in {result['inference_time']:.2f}s. "
            f"Prob Fake: {result['probability']:.4f}, Class: {result['class']}"
        )
        return result
    except Exception as e:
        logger.exception(f"Error during LipForensics prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7012))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)