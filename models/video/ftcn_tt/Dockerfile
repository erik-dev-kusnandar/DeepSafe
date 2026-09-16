FROM python:3.7-slim-buster

# buster is EOL: redirect apt to archive.debian.org
RUN sed -i 's|deb.debian.org|archive.debian.org|g' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true; \
    sed -i '/-security/d; /-updates/d' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
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

# FTCN repo (weights: ftcn_tt.pth + test_tools / model code)
RUN git clone --depth 1 https://github.com/yinglinzheng/FTCN.git /app/model

# torch 1.4 CPU (matches FTCN codebase)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==1.4.0 torchvision==0.5.0 \
        -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# FTCN+TT classifier weights
RUN mkdir -p /app/model/checkpoints && \
    curl -fL --retry 3 -o /app/model/checkpoints/ftcn_tt.pth \
      https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth

COPY app.py ftcn_core.py ./

ENV PYTHONPATH=/app/model:/app
ENV MODEL_PORT=7010
ENV FTCN_MAX_FRAME=240
ENV FTCN_NUM_THREADS=8
ENV DEFAULT_THRESHOLD=0.55

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]