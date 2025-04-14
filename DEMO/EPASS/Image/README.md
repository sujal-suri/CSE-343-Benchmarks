# EPASS Tiny ImageNet Classification Demo

Minimal instructions to run the Streamlit/Docker application.

## Prerequisites

*   Docker installed and running.

## Setup

1.  Create a directory (e.g., `epass_deploy`).
2.  Place the following files in the directory:
    *   `app.py` (Streamlit app code)
    *   `requirements.txt` (Dependencies)
    *   `Dockerfile`
    *   `best_teacher_model.pth` (Your trained model)
    *   `wnids.txt` (Tiny ImageNet metadata)
    *   `words.txt` (Tiny ImageNet metadata)

## Build Docker Image

Navigate to the project directory in your terminal and run:

```bash
docker build -t epass-tinyimagenet-app .
```

running the docker container
```bash
docker run -p 8501:8501 --name epass-img epass-tinyimagenet-app
```
Open your web browser and go to:
http://localhost:8501
Upload an image for classification.