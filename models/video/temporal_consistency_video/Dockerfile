FROM python:3.9-slim-bullseye

WORKDIR /app

# bullseye is EOL: redirect apt to archive.debian.org (repo moved off deb.debian.org)
RUN sed -i 's|deb.debian.org|archive.debian.org|g' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true; \
    sed -i '/-security/d; /-updates/d' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true

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
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MODEL_PORT=7009
ENV FRAMES_PER_VIDEO=30
ENV FLOW_CONSISTENCY_WEIGHT=0.35
ENV FACE_MOTION_WEIGHT=0.35
ENV NOISE_WEIGHT=0.15
ENV BLINK_WEIGHT=0.15

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]
