# Freesound Audio Tagging Demo (JointMatch) - Deployment

This project provides a web-based demo and Docker container for the trained JointMatch model for the Freesound Audio Tagging task.

## Prerequisites

*   **Docker:** You need Docker installed on your system. [Install Docker](https://docs.docker.com/get-docker/)
*   **Trained Model:** You need the `best_model_f.pth` file obtained from training the `jointmatch-audio.ipynb` notebook.

## Setup

1.  **Get Project Files:** Place `app.py`, `Dockerfile`, `requirements.txt`, and `labels.txt` in a single project directory.
2.  **Place Model File:** Copy your trained `best_model_f.pth` file into the same directory.
3.  **Verify Labels:** Ensure the `labels.txt` file contains the correct 41 labels in alphabetical order, one per line.

## Build the Docker Image

Open a terminal or command prompt in the project directory (where the `Dockerfile` is located) and run:

```bash
docker build -t freesound-jointmatch-demo .
```
This command builds the Docker image and tags it as freesound-jointmatch-demo. The . indicates that the build context (including the Dockerfile and other needed files) is the current directory. This might take a few minutes the first time.
Run the Docker Container
Once the image is built successfully, run the container using:
```bash
    docker run -p 8501:8501 --name joint-audio freesound-jointmatch-demo
```
