"""
AASIST Audio Deepfake Detection API
Based on: https://github.com/clovaai/aasist
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

# Add the cloned AASIST repo to path for model imports
sys.path.append("/app/temp_repo")

# pyrefly: ignore [missing-import]
from models.AASIST import Model

app = Flask(__name__)

# Global model variable
model = None
device = None


def load_model():
    """Load the pretrained AASIST detection model."""
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model architecture with AASIST config
    d_args = {
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }

    model = Model(d_args=d_args)

    # Load pretrained weights
    model_path = "/app/models/AASIST.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠ WARNING: Model weights not found at {model_path}")
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
            "model": "AASIST",
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

        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64)
        audio_file = io.BytesIO(audio_bytes)

        # Load audio
        audio, sample_rate = sf.read(audio_file)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # pyrefly: ignore [missing-import]
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Pad/trim to expected length (64600 samples = ~4.04 seconds at 16kHz)
        # For short audio: tile-repeat to fill the target length
        # For long audio: truncate
        target_len = 64600
        if len(audio) < target_len:
            repeats = int(np.ceil(target_len / len(audio)))
            audio = np.tile(audio, repeats)[:target_len]
        else:
            audio = audio[:target_len]

        # Convert to tensor (expecting shape: [batch, samples])
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            last_hidden, output = model(audio_tensor)
            # output shape: (B, 2) — [spoof_logit, bonafide_logit]
            # Class 0 = spoof, Class 1 = bonafide
            prob_fake = torch.softmax(output, dim=1)[:, 0].item()

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
                "audio_duration_seconds": len(audio) / 16000,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading AASIST Audio Deepfake Detection Model...")
    load_model()
    print("Starting Flask API...")
    app.run(host="0.0.0.0", port=int(os.getenv("MODEL_PORT", 7003)))
