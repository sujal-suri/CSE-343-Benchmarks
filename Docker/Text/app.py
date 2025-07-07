import streamlit as st
from predict import get_prediction

st.title("Text Classifier with PyTorch")

# Input form
text_input = st.text_area("Enter your text below:", height=200)

if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            label = get_prediction(text_input)
        st.success(f"Predicted Label: {label}")

