# Dockerfile

# use python runtime as parent image (matching your example)
FROM python:3.10-slim

# set working directory
WORKDIR /app

# install system dependencies needed by librosa/soundfile for audio project
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy requirements file (for the *audio* project)
COPY requirements.txt .

# install packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of application code, model (audio version)
# Ensure app.py, best_epass_simmatch_model.pth are present
COPY . .

# expose streamlit's port (matching CMD below)
EXPOSE 8501

# tell streamlit not to open a browser (from your example)
ENV STREAMLIT_SERVER_HEADLESS=true

# update healthcheck to use port 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# run app.py when container launches using explicit args (matching your example)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]