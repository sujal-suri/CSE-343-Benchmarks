# base image
from python:3.9-slim

# working directory
workdir /app

# install system dependencies needed for building python packages
# this adds build tools and common image format libraries
run apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# dependencies
copy requirements.txt ./
run pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu # cpu torch/torchvision

# copy application code
copy app.py ./
copy model/ ./model/
copy data/ ./data/

# expose port
expose 8501

# healthcheck (optional)
healthcheck cmd curl --fail http://localhost:8501/_stcore/health

# run command
cmd ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]