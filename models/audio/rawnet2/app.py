"""
RawNet2 Audio Deepfake Detection API
Based on: https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2
"""

# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import soundfile as sf
import io
import base64
import time
import os
import sys

sys.path.append("/app/temp_repo/LA/Baseline-RawNet2")

# pyrefly: ignore [missing-import]
from model import RawNet

app = Flask(__name__)

model = None
device = None


def load_model():
    """Load the pretrained RawNet2 detection model."""
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RawNet(
        d_args={
            "nb_samp": 64600,
            "first_conv": 1024,
            "in_channels": 1,
            "filts": [20, [20, 20], [20, 128], [128, 128]],
            "blocks": [2, 4],
            "nb_fc_node": 1024,
            "gru_node": 1024,
            "nb_gru_layer": 3,
            "nb_classes": 2,
        },
        device=device,
    )

    model_path = "/app/models/pre_trained_DF_RawNet2.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    else:
        print(f"WARNING: Model weights not found at {model_path}")
        print("  Please ensure model is mounted or copied to /app/models/")

    model.to(device)
    model.eval()
    return model


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "model": "RawNet2",
            "version": "1.0.0",
            "device": str(device) if device else "not loaded",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict if audio is fake or real.
    Expects JSON: {"audio_data": "base64_encoded_wav_file", "threshold": 0.5}
    """
    try:
        data = request.json
        audio_b64 = data.get("audio_data")
        threshold = data.get("threshold", 0.5)

        if not audio_b64:
            return jsonify({"error": "Missing audio_data"}), 400

        start_time = time.time()

        audio_bytes = base64.b64decode(audio_b64)
        audio_file = io.BytesIO(audio_bytes)

        audio, sample_rate = sf.read(audio_file)

        if sample_rate != 16000:
            # pyrefly: ignore [missing-import]
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        target_len = 64600
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            n_tiles = (target_len // max(len(audio), 1)) + 1
            audio = np.tile(audio, n_tiles)[:target_len]

        audio_duration = len(audio) / 16000
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(audio_tensor)
            probs = torch.exp(log_probs)
            prob_fake = probs[0, 0].item()

        prediction = 1 if prob_fake >= threshold else 0
        verdict = "fake" if prediction == 1 else "real"

        inference_time = time.time() - start_time

        return jsonify(
            {
                "probability": float(prob_fake),
                "prediction": int(prediction),
                "class": verdict,
                "inference_time": inference_time,
                "sample_rate": 16000,
                "audio_duration_seconds": float(audio_duration),
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading RawNet2 Detection Model...")
    load_model()
    print("Starting Flask API...")
    app.run(host="0.0.0.0", port=int(os.getenv("MODEL_PORT", "7004")))
