# Clash Royale GNN - Dockerfile for AMD ROCm (Strix Halo / Radeon GPUs)
# Optimized for AMD Ryzen AI MAX+ 395 with Radeon 8060S (gfx1151)

# Use latest ROCm PyTorch image
FROM rocm/pytorch:latest

# Set working directory
WORKDIR /app

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set Strix Halo GPU environment variables BEFORE any GPU operations
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTORCH_ROCM_ARCH=gfx1100
ENV HIP_VISIBLE_DEVICES=0
ENV ROCR_VISIBLE_DEVICES=0

# ROCm environment variables
ENV ROCM_HOME=/opt/rocm
ENV PATH=$ROCM_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Reinstall PyTorch nightly with ROCm support (may have gfx1151 support)
RUN pip uninstall -y torch torchvision torchaudio || true
RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3

# Copy requirements
COPY requirements.txt .

# Install PyTorch Geometric
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv || true

# Install other requirements (skip torch as it's already installed)
RUN grep -v "torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin || true

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/01-raw data/02-preprocessed data/03-features data/04-predictions \
    models models_transfer logs

# Default command - Transfer Learning
CMD ["python", "entrypoint/train_transfer_learning.py"]
