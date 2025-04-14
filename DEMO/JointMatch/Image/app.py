import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.datasets.folder import IMG_EXTENSIONS
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import io
import os

# configuration
MODEL_PATH = "model/best_model.pth"
WNIDS_PATH = "data/wnids.txt"
WORDS_PATH = "data/words.txt"
NUM_CLASSES = 200
IMG_SIZE = 64
MODEL_ARCH = 'efficientnet-b0'
DEVICE = torch.device("cpu") # cpu for deployment

# preprocessing (matches validation transform)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.Resize(IMG_SIZE + 8),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

# model and class loading functions
@st.cache_resource
def load_model(model_path, model_arch, num_classes, device):
    try:
        model = EfficientNet.from_pretrained(model_arch, num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)

        if 'model_f_state_dict' in checkpoint:
            state_dict = checkpoint['model_f_state_dict']
        elif 'state_dict' in checkpoint:
             state_dict = checkpoint['state_dict']
        else:
             state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        st.info(f"Model loaded. Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_class_mappings(wnids_path, words_path):
    try:
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]

        wnid_to_label = {}
        with open(words_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid_to_label[parts[0]] = parts[1].split(',')[0]

        idx_to_label = {i: wnid_to_label.get(wnid, 'Unknown') for i, wnid in enumerate(wnids)}
        return idx_to_label, wnids
    except Exception as e:
        st.error(f"Error loading class mappings: {e}")
        return None, None

# load resources
model = load_model(MODEL_PATH, MODEL_ARCH, NUM_CLASSES, DEVICE)
idx_to_label, wnids = load_class_mappings(WNIDS_PATH, WORDS_PATH)

# streamlit ui
st.title("JointMatch Tiny ImageNet Classifier")

supported_extensions = [ext.replace(".", "") for ext in IMG_EXTENSIONS]

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=supported_extensions
    )

if uploaded_file is not None and model is not None and idx_to_label is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # preprocess and predict
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)

        with torch.no_grad():
            outputs = model(batch_t)
            probabilities = torch.softmax(outputs, dim=1)[0]
            scores, indices = torch.topk(probabilities, 5)

        # display results
        st.subheader("Top 5 Predictions:")
        for i in range(scores.size(0)):
            idx = indices[i].item()
            label = idx_to_label.get(idx, f"Unknown Index {idx}")
            wnid = wnids[idx] if wnids and idx < len(wnids) else "N/A"
            score = scores[i].item()
            st.write(f"{i+1}. **{label}** (WNID: {wnid}) - Confidence: {score:.4f}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

elif uploaded_file is not None:
    st.warning("Model or class mappings failed to load.")