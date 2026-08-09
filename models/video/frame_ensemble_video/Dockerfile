FROM python:3.9-slim-bullseye

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV QT_QPA_PLATFORM=offscreen

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==1.12.1 \
    torchvision==0.13.1 \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MODEL_PORT=7008
ENV FRAMES_PER_VIDEO=15
ENV MODEL_TIMEOUT=300
ENV MAX_WORKERS=8
ENV FACE_REQUIRED=false

ENV NPR_ENDPOINT=http://npr_deepfakedetection:5001/predict
ENV YERMANDY_ENDPOINT=http://yermandy_clip_detection:5002/predict
ENV WAVELET_ENDPOINT=http://wavelet_clip_detection:5003/predict
ENV UNIVERSAL_ENDPOINT=http://universalfakedetect:5004/predict
ENV SPSL_ENDPOINT=http://spsl_deepfake_detection:5006/predict
ENV UCF_ENDPOINT=http://ucf_deepfake_detection:5007/predict

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]
