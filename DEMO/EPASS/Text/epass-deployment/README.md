
## Setup

1.  **Clone/Download:** Get the project files (`app.py`, `prepare_artifacts.py`, `Dockerfile`, `requirements.txt`).
2.  **Place Model:** Copy your trained model file (`epass_dbpedia_best_100pct.pth`) into the `epass-deployment` directory.
3.  **Prepare Artifacts:** Open a terminal in the `epass-deployment` directory and run the preparation script. This will download the DBpedia dataset (if necessary) and create the `vocab.pkl` file required by the application.
    ```bash
    python prepare_artifacts.py
    ```

## Running the Application (Docker - Recommended)

1.  **Build the Docker Image:** Open a terminal in the `epass-deployment` directory and run:
    ```bash
    docker build -t epass-demo .
    ```
    (This might take a few minutes).

2.  **Run the Docker Container:**
    ```bash
    docker run -p 8501:8501 --name epass-text-container epass-demo
    ```

3.  **Access the Demo:** Open your web browser and navigate to `http://localhost:8501`.

## Running Locally (Without Docker)

If you prefer to run without Docker for testing:

1.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Streamlit:** Ensure `vocab.pkl` has been created by `prepare_artifacts.py`.
    ```bash
    streamlit run app.py
    ```
4.  Access the demo (usually `http://localhost:8501`).

## Stopping the Docker Container

1.  If running interactively in the terminal, press `Ctrl + C`.
2.  Alternatively, open a new terminal and run:
    ```bash
    docker stop epass-container
    ```
3.  To remove the stopped container:
    ```bash
    docker rm epass-container
    ```

## Notes

*   Ensure the constants (hyperparameters, vocab settings) in `prepare_artifacts.py` and `app.py` match the ones used during the training run that produced the `.pth` file.
*   The `prepare_artifacts.py` script needs internet access to download the dataset the first time it's run.
*   The Streamlit app might show a warning about `BatchNorm1d` if you classify only one piece of text at a time (batch size 1). This is often expected during inference and the fallback/simplified projection in the code handles it, but results might be slightly less precise than with larger batches.