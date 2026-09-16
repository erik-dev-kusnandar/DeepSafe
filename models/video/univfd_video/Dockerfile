FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV QT_QPA_PLATFORM=offscreen

WORKDIR /app

# torch + torchvision CUDA (cu126, sm_120/Blackwell compatible)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.7.1+cu126 torchvision==0.21.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# UnivFD linear head weights (official repo, ~4 KB)
RUN mkdir -p /app/weights && \
    curl -fL --retry 3 -o /app/weights/fc_weights.pth \
      https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth

COPY app.py .

ENV MODEL_PORT=7011
ENV FRAMES_PER_VIDEO=15
ENV PRELOAD_MODEL=${PRELOAD_MODEL:-false}
ENV DEFAULT_THRESHOLD=0.55

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]