# JointMatch Text Classifier App

This project provides a web-based demonstration using Streamlit for the JointMatch text classification model trained on the DBpedia Ontology dataset.

## Features

*   Simple web interface to classify input text.
*   Uses a pre-trained JointMatch (dual-model) ensemble loaded from `jointmatch_dbpedia_best_100pct.pth`.
*   Predicts one of 14 DBpedia Ontology classes.
*   Includes a Dockerfile for easy containerized deployment.

## Project Structure

Ensure your folder is set up as follows:
Text/
├── jointmatch_dbpedia_best_100pct.pth <-- Place your trained model file here
├── app.py <-- Streamlit application code
├── requirements.txt <-- Python dependencies
├── Dockerfile <-- Docker build instructions
└── README.md <-- This file


## Setup

*   Ensure you have Python 3 (preferably 3.11 or compatible with your `torch` version) and `pip` installed.
*   For Docker deployment, ensure Docker is installed and running.

## Running the Application

You can run the application in two ways:

**Method 1: Running Directly (using Python/Streamlit)**

1.  **Navigate:**
    Open your terminal or command prompt and navigate to the project directory (`deployment_folder`).

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   On macOS/Linux: `source venv/bin/activate`
    *   On Windows: `venv\Scripts\activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Streamlit App:**
    ```bash
    streamlit run app.py
    ```

**Method 2: Running with Docker**

1.  **Navigate:**
    Open your terminal or command prompt and navigate to the project directory (`deployment_folder`).

2.  **Build the Docker Image:**
    ```bash
    docker build -t joint-text .
    ```
    *(The `.` indicates the build context is the current directory)*

3.  **Run the Docker Container:**
    ```bash
    docker run -p 8501:8501 joint-text
    ```
    *(This maps port 8501 on your host to port 8501 in the container)*

## Accessing the Demo

Once the application is running (using either method), open your web browser and navigate to:

`http://localhost:8501`

You should see the Streamlit interface. Paste some text and click "Classify Text" to get a prediction.

## Notes

*   The specific model file used in this setup is `jointmatch_dbpedia_best_100pct.pth`.
*   The `app.py` script loads the PyTorch model onto the CPU (`map_location='cpu'`) by default for better compatibility across different machines. Adjust if deploying to a GPU environment.