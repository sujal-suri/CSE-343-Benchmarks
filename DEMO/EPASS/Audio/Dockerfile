# base image
FROM python:3.10-slim

# set workdir
WORKDIR /app

# install system deps for audio (libsndfile, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt .

# install python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy app code and model
COPY . .

# expose streamlit port
EXPOSE 8501

# run streamlit headless
ENV STREAMLIT_SERVER_HEADLESS=true

# healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]