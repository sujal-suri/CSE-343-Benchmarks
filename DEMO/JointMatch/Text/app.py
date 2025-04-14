import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
from collections import Counter
import random
import numpy as np

# --- configuration ---
MODEL_PATH = "jointmatch_dbpedia_best_100pct.pth"
CLASS_NAMES = [
    'Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder',
    'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal',
    'Plant', 'Album', 'Film', 'WrittenWork'
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu") # uncomment to force cpu

# --- helper functions and classes ---

def simple_tokenizer(text):
    # simple tokenizer splitting by space and lowercasing
    return text.lower().split()

class TextEncoder(nn.Module):
    # encodes text using embedding, fc layers, and average pooling
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.pad_idx = pad_idx

    def forward(self, text, lengths):
        embedded = self.embed_dropout(self.embedding(text))
        mask = (text != self.pad_idx).unsqueeze(-1).float()
        non_zero_lengths = lengths.unsqueeze(1).float().clamp(min=1)
        summed_embeddings = torch.sum(embedded * mask, dim=1)
        avg_pooled = summed_embeddings / non_zero_lengths
        out = self.fc1(avg_pooled)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class TextClassifier(nn.Module):
    # simple classifier using the textencoder
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout, pad_idx):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim, hidden_dim, dropout, pad_idx)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, lengths):
        features = self.encoder(text, lengths)
        logits = self.classifier(features)
        return logits, None # only need logits for inference

# --- model loading ---
@st.cache_resource # cache the loaded model and vocab
def load_model_and_vocab(model_path):
    # loads the jointmatch models and vocabulary from the checkpoint
    if not os.path.exists(model_path):
        st.error(f"model file not found at: {model_path}")
        st.stop()

    try:
        # load to cpu for broader deployment compatibility
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # st.info(f"checkpoint loaded. keys: {list(checkpoint.keys())}") # uncomment for debugging

        config = checkpoint.get('config')
        if not config:
            st.error("model config not found in checkpoint.")
            st.stop()

        vocab = checkpoint.get('vocab')
        if not vocab:
            st.error("vocabulary not found in checkpoint.")
            st.stop()

        model_f_state_dict = checkpoint.get('model_f_state_dict')
        model_g_state_dict = checkpoint.get('model_g_state_dict')
        if not model_f_state_dict or not model_g_state_dict:
             st.error("model state dicts not found.")
             st.stop()

        # recreate models using saved config
        model_f = TextClassifier(
            vocab_size=config['VOCAB_SIZE'],
            embed_dim=config['EMBED_DIM'],
            hidden_dim=config['HIDDEN_DIM'],
            num_classes=config['NUM_CLASSES'],
            dropout=config['DROPOUT'],
            pad_idx=config['PAD_IDX']
        ).to(DEVICE)

        model_g = TextClassifier(
             vocab_size=config['VOCAB_SIZE'],
             embed_dim=config['EMBED_DIM'],
             hidden_dim=config['HIDDEN_DIM'],
             num_classes=config['NUM_CLASSES'],
             dropout=config['DROPOUT'],
             pad_idx=config['PAD_IDX']
         ).to(DEVICE)

        model_f.load_state_dict(model_f_state_dict)
        model_g.load_state_dict(model_g_state_dict)

        model_f.eval() # set to evaluation mode
        model_g.eval() # set to evaluation mode

        pad_idx = config['PAD_IDX']
        unk_idx = vocab.get('<unk>', 0)

        st.success(f"models loaded successfully from epoch {checkpoint.get('epoch', 'n/a')}.")
        return model_f, model_g, vocab, pad_idx, unk_idx

    except Exception as e:
        st.error(f"error loading model: {e}")
        st.stop()


# --- prediction logic ---
def preprocess_text(text, vocab, tokenizer, unk_idx):
    # tokenizes and numericalizes text
    tokens = tokenizer(text)
    indices = [vocab.get(token, unk_idx) for token in tokens]
    return torch.tensor(indices, dtype=torch.long).to(DEVICE)

def predict(text, model_f, model_g, vocab, tokenizer, unk_idx, pad_idx, class_names):
    # makes an ensemble prediction for the input text
    model_f.eval()
    model_g.eval()
    with torch.no_grad():
        tensor_indices = preprocess_text(text, vocab, tokenizer, unk_idx)
        if tensor_indices.numel() == 0:
             return "n/a", 0.0

        # model expects batch dimension: [batch_size, seq_len]
        tensor_batch = tensor_indices.unsqueeze(0)
        lengths = torch.tensor([tensor_indices.size(0)], dtype=torch.long).to(DEVICE)

        logits_f, _ = model_f(tensor_batch, lengths)
        logits_g, _ = model_g(tensor_batch, lengths)

        avg_logits = (logits_f + logits_g) / 2.0
        probs = F.softmax(avg_logits, dim=1)
        confidence, pred_idx_tensor = torch.max(probs, dim=1)
        pred_idx = pred_idx_tensor.item()
        confidence_score = confidence.item()
        predicted_class = class_names[pred_idx]

        return predicted_class, confidence_score

# --- streamlit ui ---
st.set_page_config(page_title="jointmatch-text app", layout="wide") # updated name
st.title("📚 JointMatch Text Classifier (DBpedia Demo)") # updated name
st.markdown("enter text (e.g., from a wikipedia abstract) to classify it into one of the 14 dbpedia ontology classes.")

# load models and vocabulary
model_f, model_g, vocab, pad_idx, unk_idx = load_model_and_vocab(MODEL_PATH)

# user input
st.header("input text")
user_input = st.text_area("paste your text here:", height=200, placeholder="example: the eiffel tower is a wrought-iron lattice tower...")

# prediction button
if st.button("classify text", type="primary"):
    if user_input:
        with st.spinner('classifying...'):
            predicted_class, confidence = predict(
                user_input, model_f, model_g, vocab, simple_tokenizer, unk_idx, pad_idx, CLASS_NAMES
            )
        st.subheader("prediction result")
        st.success(f"predicted class: **{predicted_class}**")
        st.info(f"confidence: **{confidence:.2%}**")
    else:
        st.warning("please enter some text to classify.")

st.markdown("---")
st.markdown("model based on jointmatch trained on the dbpedia ontology dataset.")