import streamlit as st
from PIL import Image
from predict import get_prediction

st.title("Image Classifier with PyTorch Model")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        label = get_prediction(image)
    
    st.success(f"Predicted Label: {label}")

