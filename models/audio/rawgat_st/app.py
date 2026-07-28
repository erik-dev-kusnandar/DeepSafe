"""
RawGAT-ST Audio Deepfake Detection API
Based on: https://github.com/eurecom-asp/RawGAT-ST-antispoofing
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import soundfile as sf
import io
import base64
import time
import os
import sys

sys.path.append("/app/temp_repo")

from model import RawGAT_ST

app = Flask(__name__)

model = None
device = None


def load_model():
    global model, device

    use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    d_args = {
        "nb_samp": 64600,
        "out_channels": 70,
        "first_conv": 128,
        "in_channels": 1,
        "filts": [32, [32, 32], [32, 64], [64, 64]],
        "blocks": [2, 4],
        "nb_classes": 2,
    }

    model = RawGAT_ST(d_args=d_args, device=device)

    model_path = "/app/models/Best_epoch.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
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
    return jsonify(
        {
            "status": "healthy",
            "model": "RawGAT-ST",
            "version": "1.0.0",
            "device": str(device) if device else "not loaded",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
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
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        original_duration = len(audio) / 16000

        target_len = 64600
        if len(audio) < target_len:
            repeats = target_len // len(audio) + 1
            audio = np.tile(audio, repeats)[:target_len]
        elif len(audio) > target_len:
            audio = audio[:target_len]

        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(audio_tensor, Freq_aug=False)
            prob_fake = torch.softmax(output, dim=1)[:, 1].item()

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
                "audio_duration_seconds": original_duration,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading RawGAT-ST Detection Model...")
    load_model()
    print("Starting Flask API...")
    app.run(host="0.0.0.0", port=int(os.getenv("MODEL_PORT", 7005)))
