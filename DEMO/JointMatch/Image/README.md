# JointMatch Tiny ImageNet Classifier Demo

This project provides a web-based demonstration using Streamlit for classifying images from the Tiny ImageNet dataset using a pre-trained EfficientNet model trained with the JointMatch approach (as described in the notebook). It also includes instructions for dockerizing the application.

## Project Structure
jointmatch-deployment/
├── app.py # Streamlit application script
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration
├── README.md # This file
├── model/
│ └── best_model.pth # Trained model weights 
└── data/
├── wnids.txt # Tiny ImageNet class IDs 
└── words.txt # Tiny ImageNet ID-to-label mapping 

## Setup

1.  **Model File:** Place your trained `best_model.pth` file inside the `model/` directory. This file should contain the state dictionary of the trained `model_f` from the notebook.
2.  **Data Files:**
    *   Obtain `wnids.txt` and `words.txt` from the original Tiny ImageNet dataset (`tiny-imagenet-200`).
    *   Place these two files inside the `data/` directory.

## Running Locally (without Docker)

1.  **Install Dependencies:**
    Open a terminal in the `jointmatch-deployment` directory and run:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```
3.  **Access:** Open your web browser and navigate to the local URL provided (usually `http://localhost:8501`).

## Running with Docker

1.  **Ensure Docker is running.**
2.  **Build the Docker Image:**
    Open a terminal in the `jointmatch-deployment` directory and run:
    ```bash
    docker build -t jointmatch-tinyimagenet-app .
    ```
    (You can replace `jointmatch-tinyimagenet-app` with your preferred image name).
3.  **Run the Docker Container:**
    ```bash
    docker run -p 8501:8501 --name joint-image jointmatch-tinyimagenet-app
    ```
    This maps the container's port 8501 to your local machine's port 8501.
4.  **Access:** Open your web browser and navigate to `http://localhost:8501`.

To stop the container, press `Ctrl+C` in the terminal where it's running, or use `docker ps` to find the container ID and `docker stop <container_id>`.