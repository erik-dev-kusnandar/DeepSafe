# Use a specific Python version
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y git wget build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install pip first
RUN pip install --no-cache-dir --upgrade pip "setuptools<72" wheel

# Install numpy with compatible version for scipy
RUN pip install --no-cache-dir numpy==1.22.0

# Install PyTorch
RUN pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Patch libtorch_cpu.so to clear executable stack flag (old torch bug on modern kernels)
RUN python3 << 'PYEOF'
import struct, os
lib = '/usr/local/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so'
with open(lib, 'rb') as f:
    d = bytearray(f.read())
if d[:4] != b'\x7fELF' or d[4] != 2 or d[5] != 1:
    print('Not 64-bit LE ELF')
    exit()
e_phoff = struct.unpack_from('<Q', d, 32)[0]
e_phentsize = struct.unpack_from('<H', d, 54)[0]
e_phnum = struct.unpack_from('<H', d, 56)[0]
found = False
for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type = struct.unpack_from('<I', d, off)[0]
    if p_type == 0x6474e551:
        p_flags = struct.unpack_from('<I', d, off + 4)[0]
        if p_flags & 1:
            struct.pack_into('<I', d, off + 4, p_flags & ~1)
            with open(lib, 'wb') as f:
                f.write(d)
            print('Fixed PT_GNU_STACK exec flag')
        else:
            print('PT_GNU_STACK already not executable')
        found = True
        break
if not found:
    print('PT_GNU_STACK not found')
PYEOF

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone the Wavelet-CLIP repository
RUN git clone https://github.com/lalithbharadwajbaru/Wavelet-CLIP.git model_code

# Create directory for model weights
RUN mkdir -p model_code/weights

# Create __init__.py files to make directories proper Python packages
RUN touch model_code/__init__.py && \
    touch model_code/training/__init__.py && \
    touch model_code/training/detectors/__init__.py && \
    touch model_code/training/metrics/__init__.py && \
    touch model_code/training/loss/__init__.py

# Copy the application file
COPY app.py .

# Set Hugging Face home to cache pre-trained models inside the container
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
ENV HF_HUB_CACHE=/app/huggingface_cache
RUN mkdir -p $HF_HOME

# Pre-download CLIP model needed by Wavelet-CLIP detector
RUN python3 << 'PYEOF'
from transformers import AutoProcessor, CLIPModel
AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
PYEOF

# Environment Variables
ENV MODEL_PORT=5003
ENV MODEL_PATH=model_code/weights/clip_wavelet_best.pth
ENV PRELOAD_MODEL="true"
ENV MODEL_TIMEOUT="600"
ENV PYTHONPATH=/app/model_code:/app/model_code/training:$PYTHONPATH

# Expose port
EXPOSE ${MODEL_PORT}

# Command to run the application
CMD ["python", "app.py"]