FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV QT_QPA_PLATFORM=offscreen

WORKDIR /app

# LipForensics repo (model code + preprocessing + mean face + configs)
RUN git clone --depth 1 https://github.com/ahaliassos/LipForensics.git /app/model

# torch + torchvision CUDA (cu128, sm_120/Blackwell compatible)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.7.1+cu128 torchvision==0.22.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LipForensics FF++ weights (lipforensics_ff.pth, Google Drive)
RUN pip install --no-cache-dir gdown && \
    mkdir -p /app/weights && \
    gdown "https://drive.google.com/uc?id=1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7" -O /app/weights/lipforensics_ff.pth

COPY app.py lip_core.py ./

ENV PYTHONPATH=/app/model:/app
ENV MODEL_PORT=7012
ENV LIP_MAX_FRAMES=110
ENV LIP_CLIP_SIZE=25
ENV LIP_NUM_THREADS=8
ENV DEFAULT_THRESHOLD=0.55

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]