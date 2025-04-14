# EPASS+SimMatch Freesound Audio Tagging Demo

This project provides a web-based demonstration using Streamlit for the EPASS+SimMatch audio tagging model trained on the Freesound General-Purpose Audio Tagging Challenge dataset (FSDKaggle2018). The application allows users to upload an audio file and receive a prediction for the primary sound event present.

The deployment is containerized using Docker for easy setup and execution.

## Features

*   Upload audio files (WAV, MP3, OGG, FLAC).
*   Automatic resampling to the model's required sample rate (32kHz).
*   Predicts the most likely sound event from 41 classes.
*   Displays the top predicted label and its confidence score.
*   Shows the top 5 predictions with their respective confidence scores.

## Technology Stack

*   Python 3.10 (as per Dockerfile base)
*   Streamlit (Web Framework)
*   PyTorch (Deep Learning Framework)
*   Librosa, SoundFile (Audio Processing)
*   Docker (Containerization)

## Prerequisites

Before you begin, ensure you have Docker installed on your system.
*   **Docker Engine:** Follow the official installation guide for your operating system: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

## Setup and Running Instructions

1.  **Get the Code:**
    *   Clone this repository or download the source code files (`app.py`, `requirements.txt`, `Dockerfile`).
    *   **Crucially:** Place your trained **audio** model weights file, `best_epass_simmatch_model.pth`, into the same directory as the other files.

    Your project directory should look like this:
    ```
    epass_deployment/
    ├── best_epass_simmatch_model.pth
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
    ```

2.  **Build the Docker Image:**
    *   Open a terminal or command prompt.
    *   Navigate (`cd`) into the `epass_deployment` directory.
    *   Run the following command to build the Docker image using the `Dockerfile` (configured for port `8501` via CMD). Name the image `epass-audio`. **Ensure this completes successfully.**
        ```bash
        docker build -t epass-audio .
        ```
        *(Image name: `epass-audio`)*

3.  **Run the Docker Container:**
    *   Once the image is successfully built, run the container using the following command:
        ```bash
        docker run -p 8501:8501 --name epass-audio-container epass-audio
        ```
    *   **Explanation:**
        *   `-p 8501:8501`: Maps port `8501` on your host machine to port `8501` inside the container (where Streamlit is listening, as specified in the `CMD`).
        *   `--name epass-audio-container`: Assigns the name `epass-audio-container` to the running container instance.
        *   `epass-audio`: Specifies that the container should be created from the image named `epass-audio`.
        *   *(Add `--rm` if you want the container automatically removed on stop)*

## Usage

1.  After running the `docker run` command, open your web browser.
2.  Navigate to `http://localhost:8501`.
3.  You should see the Streamlit application interface running.
4.  Use the file uploader and classification button as described before.

## Managing the Container

*   **Stop:** `docker stop epass-audio-container`
*   **Start (if stopped without --rm):** `docker start epass-audio-container`
*   **View Logs:** `docker logs epass-audio-container` *(Crucial for debugging)*
*   **Remove (if stopped without --rm):** `docker rm epass-audio-container`

## Project Structure
├── best_epass_simmatch_model.pth # Trained PyTorch model weights
├── app.py # The Streamlit application script
├── requirements.txt # Python dependencies
├── Dockerfile # Instructions for building the Docker image (uses explicit CMD args for 8501)
└── README.md # This file (updated image/container names)


## Troubleshooting

*   **ERR_EMPTY_RESPONSE or Crashes?** Check the logs first: `docker logs epass-audio-container`. Look for Python errors (especially related to audio processing or model loading). Ensure the image was rebuilt correctly.
*   **Error building?** Check the output of the `docker build` command for errors. Ensure `best_epass_simmatch_model.pth` is present.
*   **"address already in use" / "port is already allocated"?** Another application is using port 8501 on your host machine. Stop that application or map to a different host port (e.g., `docker run -p 8888:8501 --name epass-audio-container epass-audio` and access via `http://localhost:8888`).