# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import librosa
import numpy as np
import os
from io import BytesIO
import soundfile as sf
import traceback # for error logging

# --- configuration ---
sr = 32000 # sample rate
duration = 5 # seconds
n_mels = 128 # mel bins
n_fft = 1024 # fft window size
hop_length = 512 # fft hop length
max_len = sr * duration # max audio length in samples

# --- model params ---
num_classes = 41 # output classes
embedding_dim = 128
num_projectors = 3

# --- label map ---
# sorted list of class names
labels = sorted([
    'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 'Burping_or_eructation',
    'Bus', 'Cello', 'Chime', 'Clarinet', 'Computer_keyboard', 'Cough',
    'Cowbell', 'Double_bass', 'Drawer_open_or_close', 'Electric_piano',
    'Fart', 'Finger_snapping', 'Fireworks', 'Flute', 'Glockenspiel', 'Gong',
    'Gunshot_or_gunfire', 'Harmonica', 'Hi-hat', 'Keys_jangling', 'Knock',
    'Laughter', 'Meow', 'Microwave_oven', 'Oboe', 'Saxophone', 'Scissors',
    'Shatter', 'Snare_drum', 'Squeak', 'Tambourine', 'Tearing', 'Telephone',
    'Trumpet', 'Violin_or_fiddle', 'Writing'
])
idx_to_label = {idx: label for idx, label in enumerate(labels)}

# --- model definition ---
class EpassSimMatchNet(nn.Module):
    def __init__(self, num_classes, embedding_dim, num_projectors, pretrained=True):
        super().__init__()
        # encoder (resnet18 base)
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # adapt for 1 channel (spectrogram)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1]) # use up to pool layer
        encoder_output_dim = base_model.fc.in_features # get feature dim (512)

        # classifier head
        self.fc = nn.Linear(encoder_output_dim, num_classes)

        # projector heads (mlps)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_output_dim, encoder_output_dim), # intermediate layer
                nn.ReLU(),
                nn.Linear(encoder_output_dim, embedding_dim)
            ) for _ in range(num_projectors)
        ])
        self.num_projectors = num_projectors

    def forward(self, x):
        features = self.encoder(x)
        flat_features = torch.flatten(features, 1)

        # classification logits
        logits = self.fc(flat_features)

        # embeddings not used in this simple app, return only logits
        # embeddings = [proj(flat_features) for proj in self.projectors]
        # ensembled_embedding = torch.mean(torch.stack(embeddings, dim=0), dim=0)

        return logits

# --- audio preprocessing ---
def preprocess_audio(y, sr, max_len, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length):
    # ensure float32 input
    y = y.astype(np.float32)

    # pad or truncate
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    elif len(y) > max_len:
        y = y[:max_len]

    # compute mel spectrogram
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # normalize to [0, 1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max == mel_min: # handle silent clips
             st.warning("audio silent or constant, prediction unreliable.")
             time_steps = int(np.ceil(max_len / hop_length)) # calc expected steps
             return np.zeros((n_mels, time_steps), dtype=np.float32)

        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        return mel_norm.astype(np.float32) # shape: (n_mels, time)

    except Exception as e:
        st.error(f"error in mel spectrogram calc: {e}")
        time_steps = int(np.ceil(max_len / hop_length)) # calc expected steps
        return np.zeros((n_mels, time_steps), dtype=np.float32)

# --- model loading ---
@st.cache_resource # cache model
def load_model(model_path="best_epass_simmatch_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EpassSimMatchNet(num_classes, embedding_dim, num_projectors, pretrained=True)
    try:
        # load weights to correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # set to evaluation mode
        st.success(f"model loaded successfully onto {device}.")
        return model, device
    except FileNotFoundError:
        st.error(f"model file not found at '{os.path.abspath(model_path)}'. check path.")
        return None, None
    except Exception as e:
        st.error(f"error loading model: {e}")
        st.error(f"traceback: {traceback.format_exc()}")
        return None, None

# --- prediction function ---
def predict(model, device, audio_data, sr):
    if model is None or device is None:
        return "model not loaded", 0.0, []

    # preprocess audio
    mel_spec = preprocess_audio(y=audio_data.astype(np.float32), sr=sr, max_len=max_len)

    # check preproc output shape
    expected_time_steps = int(np.ceil(max_len / hop_length))
    if mel_spec.shape[1] == 0 or mel_spec.shape != (n_mels, expected_time_steps):
         st.warning(f"preproc shape {mel_spec.shape} != expected ({n_mels}, {expected_time_steps}). using zeros.")
         mel_spec = np.zeros((n_mels, expected_time_steps), dtype=np.float32) # ensure correct shape

    # convert to tensor (batch, channel, mels, time)
    mel_tensor = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

    # inference
    try:
        with torch.no_grad():
            logits = model(mel_tensor)
            probabilities = F.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probabilities, dim=1)

            # get top 5 preds
            top_k_prob, top_k_indices = torch.topk(probabilities, 5, dim=1)
            top_k_preds = [(idx_to_label.get(i.item(), "unknown"), p.item()) for i, p in zip(top_k_indices[0], top_k_prob[0])]

        predicted_idx = top_idx.item()
        predicted_label = idx_to_label.get(predicted_idx, "unknown")
        confidence = top_prob.item()

        return predicted_label, confidence, top_k_preds

    except Exception as e:
        st.error(f"error during inference: {e}")
        st.error(f"traceback: {traceback.format_exc()}")
        return "inference error", 0.0, []

# --- streamlit ui ---
st.set_page_config(layout="wide")
st.title("🔊 freesound audio tagging (epass+simmatch)")
st.write("upload audio (wav, mp3, ogg, flac) for sound event classification.")
st.write(f"model expects audio @ {sr} hz, {duration} sec duration.")

# load model
model, device = load_model()

uploaded_file = st.file_uploader("choose audio file...", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None and model is not None:
    st.write(f"processing: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} kb)")
    # read audio
    try:
        file_bytes = BytesIO(uploaded_file.read()) # read in-memory
        audio_data, sr_orig = sf.read(file_bytes)

        # fix: convert to float32 early
        audio_data = audio_data.astype(np.float32)

        # convert to mono if needed
        if audio_data.ndim > 1:
            st.write("stereo detected, converting to mono...")
            audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32) # ensure float32 after mean

        st.write(f"original sr: {sr_orig} hz, duration: {len(audio_data)/sr_orig:.2f} sec")

        # resample if needed
        if sr_orig != sr:
            st.write(f"resampling from {sr_orig} hz to {sr} hz...")
            audio_data = librosa.resample(y=audio_data, orig_sr=sr_orig, target_sr=sr)
            current_sr = sr # update sr variable
            st.write("resampling complete.")
        else:
            current_sr = sr_orig # use original sr
            st.write(f"sr matches target ({sr} hz).")

        # display audio player (preview)
        st.write("audio preview (processed segment):")
        preview_audio = audio_data[:max_len] # show only the processed part
        st.audio(np.ascontiguousarray(preview_audio), format='audio/wav', sample_rate=current_sr)

        # classify button
        if st.button("classify sound"):
            with st.spinner('preprocessing & predicting...'):
                # ensure float32 input to predict
                predicted_label, confidence, top_k_preds = predict(model, device, audio_data.astype(np.float32), current_sr)

            if predicted_label != "inference error" and predicted_label != "model not loaded":
                st.subheader("prediction:")
                st.success(f"**{predicted_label}** (confidence: {confidence:.2%})")

                st.subheader("top 5 predictions:")
                for label, prob in top_k_preds:
                    st.write(f"- {label}: {prob:.2%}")
            # errors already shown by predict/load_model

    except Exception as e:
        st.error(f"error reading/processing audio: {e}")
        st.error(f"traceback: {traceback.format_exc()}")
        st.error("check file format (wav often best).")

elif uploaded_file is not None and model is None:
    st.error("cannot process file, model failed to load.")
elif model is None:
     st.warning("model not loaded. check file path/logs.")

st.sidebar.info(
    """
    **model:** epass + simmatch base (resnet18)
    **dataset:** fsdkaggle2018 (subset)
    **classes:** 41 sound events.
    **input:** audio -> 32khz -> 5 sec segments.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    *app based on audio analysis techniques.*
    *requires `best_epass_simmatch_model.pth`.*
    """
)