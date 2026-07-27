"""
WavLM Audio Deepfake Detection API
Based on: https://huggingface.co/DavidCombei/wavLM-base-Deepfake_V2
"""

# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import soundfile as sf
# pyrefly: ignore [missing-import]
import librosa
# pyrefly: ignore [missing-import]
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import io
import base64
import time
import os

app = Flask(__name__)

# Global model variables
model = None
feature_extractor = None
device = None
spoof_idx = None


def load_model():
    """Load the pretrained WavLM deepfake detection model from HuggingFace."""
    global model, feature_extractor, device, spoof_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "DavidCombei/wavLM-base-Deepfake_V2"

    print(f"Loading WavLM feature extractor from {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    print(f"Loading WavLM model from {model_name}...")
    model = AutoModelForAudioClassification.from_pretrained(model_name)

    model.to(device)
    model.eval()

    # Determine spoof class index
    label2id = model.config.label2id
    if "spoof" in label2id:
        spoof_idx = label2id["spoof"]
    elif "fake" in label2id:
        spoof_idx = label2id["fake"]
    else:
        spoof_idx = 1

    print(f"Model loaded on {device}. Label mapping: {label2id}, spoof_idx={spoof_idx}")
    return model


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "model": "WavLM",
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

        # Load audio (expecting WAV format, 16kHz)
        audio, sample_rate = sf.read(audio_file)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Cast to float32
        audio = audio.astype(np.float32)

        audio_duration = len(audio) / 16000

        # Run inference through the transformer pipeline
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_fake = probs[0, spoof_idx].item()

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
                "audio_duration_seconds": audio_duration,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading WavLM Deepfake Detection Model...")
    load_model()
    print("Starting Flask API...")
    app.run(host="0.0.0.0", port=int(os.getenv("MODEL_PORT", 7006)))
