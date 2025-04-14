import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio # kept for consistency if needed later
import librosa
import numpy as np
import os
import pandas as pd
from io import BytesIO

# --- configuration ---
class AppConfig:
    SR = 32000
    DURATION = 5
    N_MELS = 128
    NUM_CLASSES = 41
    MODEL_PATH = "best_model_f.pth"
    LABELS_PATH = "labels.txt"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = AppConfig()

# --- preprocessing ---
def preprocess_audio(y, sr=config.SR, duration=config.DURATION, n_mels=config.N_MELS):
    """preprocesses waveform y into a mel spectrogram tensor."""
    try:
        max_len = sr * duration
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max == mel_min:
             time_steps = int(librosa.time_to_frames(duration, sr=sr, hop_length=512)) + 1
             return np.zeros((n_mels, time_steps), dtype=np.float32) # handle silent clips

        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        return mel_norm.astype(np.float32)

    except Exception as e:
        st.error(f"error during preprocessing: {e}")
        time_steps = int(librosa.time_to_frames(duration, sr=sr, hop_length=512)) + 1
        return np.zeros((n_mels, time_steps), dtype=np.float32) # return zeros on error

# --- model definition ---
def create_model(num_classes, pretrained=False):
    """creates the resnet18-based model structure."""
    base_model = models.resnet18(weights=None)
    base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    encoder = nn.Sequential(*list(base_model.children())[:-1])
    encoder_output_dim = base_model.fc.in_features
    classifier = nn.Linear(encoder_output_dim, num_classes)
    return nn.Sequential(encoder, nn.Flatten(), classifier)

# --- load model and labels ---
@st.cache_resource
def load_model_and_labels():
    """loads the model and label list."""
    try:
        model = create_model(config.NUM_CLASSES)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
    except FileNotFoundError:
        st.error(f"model file not found at {config.MODEL_PATH}.")
        return None, None
    except Exception as e:
        st.error(f"error loading model: {e}")
        return None, None

    try:
        with open(config.LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != config.NUM_CLASSES:
             st.warning(f"label count mismatch: file ({len(labels)}), config ({config.NUM_CLASSES}).")
    except FileNotFoundError:
        st.error(f"labels file not found at {config.LABELS_PATH}.")
        return model, None
    except Exception as e:
        st.error(f"error loading labels: {e}")
        return model, None

    return model, labels

# --- prediction ---
@st.cache_data
def predict(waveform, sr, _model, _labels):
    """runs preprocessing and inference."""
    if _model is None or _labels is None:
        return None

    mel_spec = preprocess_audio(waveform, sr=sr, duration=config.DURATION, n_mels=config.N_MELS)
    mel_tensor = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(config.DEVICE) # add batch and channel dims

    with torch.no_grad():
        output = _model(mel_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_p, top_class_indices = torch.topk(probabilities, 5, dim=1)

    top_p = top_p.cpu().numpy().flatten()
    top_class_indices = top_class_indices.cpu().numpy().flatten()
    predictions = [{"label": _labels[idx], "probability": prob} for idx, prob in zip(top_class_indices, top_p)]

    return predictions


# --- streamlit ui ---
st.set_page_config(layout="wide")
st.title("🔊 Freesound Audio Tagging Demo (JointMatch)")
st.write(f"Using model: `{config.MODEL_PATH}` on device: `{config.DEVICE}`")

model, labels = load_model_and_labels()

if model and labels:
    uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3, .ogg)...", type=['wav', 'mp3', 'ogg', 'flac'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
        try:
            audio_bytes = uploaded_file.read()
            y, sr = librosa.load(BytesIO(audio_bytes), sr=config.SR, mono=True)
            st.write("Preprocessing and predicting...")
            predictions = predict(y, sr, model, labels)

            if predictions:
                st.success("Top 5 Predictions:")
                cols = st.columns(5)
                for i, p in enumerate(predictions):
                     with cols[i]:
                         st.metric(label=f"Rank {i+1}", value=p['label'], delta=f"{p['probability']:.2%}", delta_color="off")
                st.write("---")
                df = pd.DataFrame(predictions)
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"error loading or processing audio file: {e}")

else:
    st.error("model or labels could not be loaded.")

st.sidebar.header("About")
st.sidebar.info("JointMatch model for Freesound Audio Tagging.")