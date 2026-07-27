FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

RUN git clone https://github.com/eurecom-asp/RawGAT-ST-antispoofing.git temp_repo

RUN uv pip install --system --no-cache \
    torch==1.13.1 \
    torchaudio==0.13.1 \
    flask==2.3.0 \
    soundfile==0.12.1 \
    librosa==0.10.0 \
    numpy==1.24.0 \
    pyyaml==6.0

RUN mkdir -p /app/models

RUN cp temp_repo/Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth /app/models/Best_epoch.pth

COPY app.py .

EXPOSE 7005

ENV MODEL_PORT=7005
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
