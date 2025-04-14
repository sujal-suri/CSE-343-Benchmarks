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

# --- configuration (from notebook page 2) ---
SR = 32000
DURATION = 5
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
MAX_LEN = SR * DURATION

# model params (from notebook page 2 & 9)
NUM_CLASSES = 41 # updated based on labels found in notebook output
EMBEDDING_DIM = 128
NUM_PROJECTORS = 3

# --- label map (from notebook page 8 output) ---
# extracted exactly from notebook output, sorted alphabetically
LABELS = sorted([
    'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 'Burping_or_eructation',
    'Bus', 'Cello', 'Chime', 'Clarinet', 'Computer_keyboard', 'Cough',
    'Cowbell', 'Double_bass', 'Drawer_open_or_close', 'Electric_piano',
    'Fart', 'Finger_snapping', 'Fireworks', 'Flute', 'Glockenspiel', 'Gong',
    'Gunshot_or_gunfire', 'Harmonica', 'Hi-hat', 'Keys_jangling', 'Knock',
    'Laughter', 'Meow', 'Microwave_oven', 'Oboe', 'Saxophone', 'Scissors',
    'Shatter', 'Snare_drum', 'Squeak', 'Tambourine', 'Tearing', 'Telephone',
    'Trumpet', 'Violin_or_fiddle', 'Writing'
])
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# --- model definition (from notebook page 9-10) ---
class EpassSimMatchNet(nn.Module):
    def __init__(self, num_classes, embedding_dim, num_projectors, pretrained=True):
        super().__init__()
        # encoder (resnet18 base) - notebook used default weights, so pretrained=true is correct
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # use sequential up to layer before fc (includes adaptive avg pool)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        encoder_output_dim = base_model.fc.in_features # 512 for resnet18

        # classifier head
        self.fc = nn.Linear(encoder_output_dim, num_classes)

        # epass projector heads (multiple mlps) - needed for architecture matching
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_output_dim, encoder_output_dim), # optional intermediate layer from notebook
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

        # embeddings are calculated but only logits needed for basic classification app
        # returning only logits to simplify inference step for this app
        # if embeddings were needed, they'd be calculated and returned here
        # embeddings = [proj(flat_features) for proj in self.projectors]
        # ensembled_embedding = torch.mean(torch.stack(embeddings, dim=0), dim=0)

        return logits # return only logits for this app


# --- audio preprocessing (from notebook page 3-4) ---
def preprocess_audio(y, sr, max_len, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # pad or truncate to fixed length
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    elif len(y) > max_len:
        y = y[:max_len]

    # compute mel spectrogram
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # normalize to [0, 1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max == mel_min: # avoid division by zero for silent clips
             st.warning("audio seems silent, prediction might be unreliable.")
             # calculate expected time steps even for zeros
             time_steps = int(max_len / hop_length) + 1
             return np.zeros((n_mels, time_steps), dtype=np.float32)

        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        return mel_norm.astype(np.float32) # shape: (n_mels, time)

    except Exception as e:
        st.error(f"error processing audio: {e}")
        # calculate expected time steps even for error
        time_steps = int(max_len / hop_length) + 1
        return np.zeros((n_mels, time_steps), dtype=np.float32)


# --- model loading ---
@st.cache_resource # cache model loading
def load_model(model_path="best_epass_simmatch_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pretrained=true matches notebook's use of default resnet weights
    model = EpassSimMatchNet(NUM_CLASSES, EMBEDDING_DIM, NUM_PROJECTORS, pretrained=True)
    try:
        # load weights onto the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # set model to evaluation mode
        st.success("model loaded successfully.")
        return model, device
    except FileNotFoundError:
        st.error(f"model file not found at {model_path}. place it in the same directory as app.py.")
        return None, None
    except Exception as e:
        st.error(f"error loading model: {e}")
        return None, None

# --- prediction function ---
def predict(model, device, audio_data, sr):
    if model is None or device is None:
        return "model not loaded", 0.0, []

    # preprocess the audio
    mel_spec = preprocess_audio(y=audio_data, sr=sr, max_len=MAX_LEN)

    # convert to tensor, add batch & channel dimensions -> (1, 1, n_mels, time)
    mel_tensor = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

    # perform inference
    with torch.no_grad():
        logits = model(mel_tensor) # get logits from model
        probabilities = F.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probabilities, dim=1)

        # get top 5 predictions
        top_k_prob, top_k_indices = torch.topk(probabilities, 5, dim=1)
        top_k_preds = [(IDX_TO_LABEL.get(i.item(), "unknown"), p.item()) for i, p in zip(top_k_indices[0], top_k_prob[0])]

    predicted_idx = top_idx.item()
    predicted_label = IDX_TO_LABEL.get(predicted_idx, "unknown")
    confidence = top_prob.item()

    return predicted_label, confidence, top_k_preds


# --- streamlit ui ---
st.set_page_config(layout="wide")
st.title("🔊 freesound audio tagging (epass+simmatch)")
st.write("upload an audio file (e.g., wav, mp3) to classify its sound event.")

# load the model
model, device = load_model()

uploaded_file = st.file_uploader("choose an audio file...", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None and model is not None:
    # read audio file
    try:
        # use soundfile to read, handles more formats potentially
        audio_data, sr_orig = sf.read(BytesIO(uploaded_file.read()))
        # convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        st.write(f"original sample rate: {sr_orig} hz, duration: {len(audio_data)/sr_orig:.2f}s")

        # resample if necessary to match model's expected SR
        if sr_orig != SR:
            st.write(f"resampling from {sr_orig} hz to {SR} hz...")
            audio_data = librosa.resample(y=audio_data, orig_sr=sr_orig, target_sr=SR)
            sr = SR # update sample rate variable
            st.write("resampling complete.")
        else:
            sr = sr_orig # use original if already correct

        # display audio player
        st.audio(audio_data, format='audio/wav', sample_rate=SR) # use potentially resampled data

        # classify button
        if st.button("classify sound"):
            with st.spinner('processing and predicting...'):
                predicted_label, confidence, top_k_preds = predict(model, device, audio_data, sr)

            st.subheader("prediction:")
            st.success(f"**{predicted_label}** (confidence: {confidence:.2%})")

            st.subheader("top 5 predictions:")
            for label, prob in top_k_preds:
                st.write(f"- {label}: {prob:.2%}")

    except Exception as e:
        st.error(f"error reading or processing audio file: {e}")
        st.error("please try a different file or format (wav often works best).")

elif uploaded_file is not None and model is None:
    st.error("cannot process file because the model failed to load.")

st.sidebar.info(
    """
    **model:** epass + simmatch base (resnet18 encoder)
    **dataset:** fsdkaggle2018
    **classes:** 41 sound events.
    """
)