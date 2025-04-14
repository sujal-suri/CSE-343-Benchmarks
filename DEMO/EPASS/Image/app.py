# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError
import io
import os
import numpy as np

# --- configuration ---
num_classes = 200
num_projectors = 3
projection_dim = 128
model_path = 'best_teacher_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- epass model definition ---
class EPASSModel(nn.Module):
    def __init__(self, num_classes=num_classes, num_projectors=num_projectors, projection_dim=projection_dim, pretrained=False):
        super(EPASSModel, self).__init__()
        self.encoder = models.resnet18(weights=None) # structure only
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.projectors = nn.ModuleList()
        for _ in range(num_projectors):
            projector = nn.Sequential(
                nn.Linear(num_ftrs, num_ftrs),
                nn.BatchNorm1d(num_ftrs),
                nn.ReLU(inplace=True),
                nn.Linear(num_ftrs, projection_dim)
            )
            self.projectors.append(projector)
        self.num_projectors = num_projectors

    def forward(self, x, return_features=False, return_proj_only=False):
        features = self.encoder(x)
        if return_proj_only:
             projected = [proj(features) for proj in self.projectors]
             avg_projection = torch.stack(projected, dim=0).mean(dim=0)
             return avg_projection
        logits = self.fc(features)
        if return_features:
            projected = [proj(features) for proj in self.projectors]
            avg_projection = torch.stack(projected, dim=0).mean(dim=0)
            return logits, features, avg_projection
        else:
            return logits


# --- data preprocessing ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_val = transforms.Compose([
    transforms.Resize(70),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize
])

# --- helper functions ---
@st.cache_resource
def load_class_names(wnids_path='wnids.txt', words_path='words.txt'):
    # loads wnid to human-readable name mapping
    try:
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        wnid_to_name = {}
        with open(words_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid, name = parts[0], parts[1]
                    wnid_to_name[wnid] = name.split(',')[0] # simple cleaning
        # assumes imagefolder sorts classes alphabetically by wnid folder name during training
        # if the val loader was built differently, this needs adjustment
        sorted_wnids = sorted(wnids)
        class_idx_to_name = {i: wnid_to_name.get(wnid, wnid) for i, wnid in enumerate(sorted_wnids)}
        if len(class_idx_to_name) != num_classes:
             st.warning(f"warning: found {len(class_idx_to_name)} classes, expected {num_classes}.")
             # fill missing if needed
             for i in range(num_classes):
                  if i not in class_idx_to_name:
                       class_idx_to_name[i] = f"unknown class {i}"
        return class_idx_to_name
    except FileNotFoundError:
        st.error(f"error: `wnids.txt` or `words.txt` not found. please place them in the same directory as `app.py`.")
        return None
    except Exception as e:
        st.error(f"error loading class names: {e}")
        return None

@st.cache_resource
def load_model(model_path, device):
    # loads the pre-trained epass model
    try:
        model = EPASSModel(num_classes=num_classes, num_projectors=num_projectors, projection_dim=projection_dim, pretrained=False).to(device)
        state_dict = torch.load(model_path, map_location=device)
        if any(k.startswith('module.') for k in state_dict.keys()): # handle dataparallel prefix if present
             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"error: model file '{model_path}' not found.")
        return None
    except Exception as e:
        st.error(f"error loading model: {e}")
        return None

def predict(image, model, transform, class_names, device):
    """preprocesses image and makes prediction. returns none, none if error occurs."""
    # check image mode and convert only if necessary
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        st.error(f"failed to convert image to rgb: {e}")
        return None, None # return none on error

    # apply transforms
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        st.error(f"failed during image transformation: {e}")
        return None, None # return none on error

    # predict
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_idx = predicted_idx.item()
        predicted_class_name = class_names.get(predicted_class_idx, f"unknown class {predicted_class_idx}")
        confidence_score = confidence.item()
        return predicted_class_name, confidence_score
    except Exception as e:
        st.error(f"failed during model prediction: {e}")
        return None, None # return none on error

# --- streamlit app ui ---
st.set_page_config(page_title="tiny imagenet classification")
st.title("epass tiny imagenet classifier")
st.write("upload an image for classification.")

model = load_model(model_path, device)
class_names = load_class_names()

if model is None or class_names is None:
    st.stop()

uploaded_file = st.file_uploader("choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))

        st.image(image, caption='uploaded image.', use_column_width=True)
        st.write("")
        st.write("classifying...")

        # make prediction
        predicted_class, confidence = predict(image, model, transform_val, class_names, device)

        # display the result only if prediction was successful
        if predicted_class is not None and confidence is not None:
            st.success(f"predicted class: {predicted_class}")
            st.info(f"confidence: {confidence:.2%}")
        else:
            # error messages are shown in the predict function
            st.error("classification failed.")

    except UnidentifiedImageError:
         st.error("error: cannot identify image file. it might be corrupted or in an unsupported format.")
    except Exception as e:
        st.error(f"error processing uploaded file: {e}")