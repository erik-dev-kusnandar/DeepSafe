"""
Vocoder Artifacts Audio Deepfake Detection API
Based on: https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts
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

# Add the model directory to path
sys.path.append("/app/temp_repo")

# pyrefly: ignore [missing-import]
from model import RawNet

app = Flask(__name__)

# Global model variable
model = None
device = None


def load_model():
    """Load the pretrained Vocoder Artifacts detection model."""
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA is available. Disabling cuDNN to prevent CUDNN_STATUS_INTERNAL_ERROR.")
        torch.backends.cudnn.enabled = False

    # Initialize model architecture
    model = RawNet(
        d_args={
            "nb_samp": 64600,  # Sample length
            "first_conv": 251,
            "in_channels": 1,
            "filts": [20, [20, 20], [20, 128], [128, 128]],
            "nb_fc_node": 1024,
            "gru_node": 1024,
            "nb_gru_layer": 3,
            "nb_classes": 2,
        },
        device=device,
    )

    # Load pretrained weights
    model_path = "/app/models/librifake_pretrained_lambda0.5_epoch_25.pth"
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
            "model": "Vocoder Artifacts Detection",
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

        # Load audio (expecting WAV format, 16kHz)
        audio, sample_rate = sf.read(audio_file)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # pyrefly: ignore [missing-import]
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Pad/trim to expected length (64600 samples = ~4 seconds at 16kHz)
        target_len = 64600
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)), "constant")

        # Convert to tensor (expecting shape: [batch, samples])
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

        # Run inference
        try:
            with torch.no_grad():
                output_binary, _ = model(audio_tensor)
                # Model returns log-softmax, convert to probability
                prob_fake = torch.exp(output_binary)[0, 1].item()
        except Exception as e:
            if ("cuda" in str(e).lower() or "cudnn" in str(e).lower()) and device.type == "cuda":
                print(f"⚠ CUDA/cuDNN error: {e}. Falling back to CPU for execution.")
                device = torch.device("cpu")
                model.to(device)
                audio_tensor = audio_tensor.to(device)
                with torch.no_grad():
                    output_binary, _ = model(audio_tensor)
                    prob_fake = torch.exp(output_binary)[0, 1].item()
            else:
                raise e

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
    print("Loading Vocoder Artifacts Detection Model...")
    load_model()
    print("Starting Flask API...")
    app.run(host="0.0.0.0", port=int(os.getenv("MODEL_PORT", 8001)))
