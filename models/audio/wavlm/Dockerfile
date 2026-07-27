FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

RUN uv pip install --system --no-cache \
    torch==1.13.1 \
    torchaudio==0.13.1 \
    flask==2.3.0 \
    soundfile==0.12.1 \
    librosa==0.10.0 \
    numpy==1.24.0 \
    transformers==4.30.0 \
    huggingface_hub==0.15.0

RUN mkdir -p /app/models

COPY app.py .

EXPOSE 7006

ENV MODEL_PORT=7006
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models

CMD ["python", "app.py"]
