import streamlit as st
import tempfile
from predict import get_prediction
import asyncio
import sys

if sys.platform == "darwin" and sys.version_info >= (3, 8):
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        print(f"Error setting event loop policy: {e}")

st.title("Audio Classifier with PyTorch")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Classifying..."):
        label = get_prediction(tmp_path)
    
    st.success(f"Predicted Label: {label}")

