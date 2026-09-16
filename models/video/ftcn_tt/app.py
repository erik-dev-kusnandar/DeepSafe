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

import ftcn_core

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FTCN+TT Video Deepfake Detection Service",
    description="FTCN+TT (Swin temporal transformer on face clips) deepfake video detector. Detects faces and landmarks on video frames, tracks them across time, and classifies 32-frame face clips.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "ftcn_tt_video_service"
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
        result = ftcn_core.run_inference(tmp_name)
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
        "max_frame_cap": ftcn_core.MAX_FRAME,
        "threads": ftcn_core.NUM_THREADS,
        "default_threshold": DEFAULT_THRESHOLD,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME_DISPLAY,
        "max_frame_cap": ftcn_core.MAX_FRAME,
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
            f"FTCN+TT video prediction completed in {result['inference_time']:.2f}s. "
            f"Prob Fake: {result['probability']:.4f}, Class: {result['class']}"
        )
        return result
    except Exception as e:
        logger.exception(f"Error during FTCN+TT prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7010))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)