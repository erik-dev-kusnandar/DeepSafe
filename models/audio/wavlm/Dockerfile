FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# torch>=2.8.0+cu128 gives sm_120 (RTX 50-series / Blackwell) support.
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 \
    torchaudio==2.8.0
RUN uv pip install --system --no-cache \
    flask==2.3.0 \
    soundfile==0.12.1 \
    librosa==0.10.0 \
    numpy==1.24.0 \
    transformers==4.48.3 \
    huggingface_hub==0.27.0

RUN mkdir -p /app/models

COPY app.py .

EXPOSE 7006

ENV MODEL_PORT=7006
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models

CMD ["python", "app.py"]
