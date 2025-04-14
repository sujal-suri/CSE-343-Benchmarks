# base python image
from python:3.11-slim

# working directory
workdir /app

# copy requirements and install
copy requirements.txt .
run pip install --no-cache-dir -r requirements.txt

# copy application code
copy . .

# expose streamlit port
expose 8501

# tell streamlit to run headless
env streamlit_server_headless=true

# run the app
cmd ["streamlit", "run", "app.py"]