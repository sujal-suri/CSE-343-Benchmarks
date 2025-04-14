# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np

# --- constants and hyperparameters (match training) ---
EMBED_DIM = 128
HIDDEN_DIM = 256
PROJ_DIM = 128
NUM_PROJECTORS = 3
NUM_CLASSES = 14
DROPOUT = 0.4

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

MODEL_PATH = "epass_dbpedia_best_100pct.pth"
VOCAB_PATH = "vocab.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder',
    'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal',
    'Plant', 'Album', 'Film', 'WrittenWork'
]
if len(CLASS_NAMES) != NUM_CLASSES:
    st.error(f"class names length ({len(CLASS_NAMES)}) != NUM_CLASSES ({NUM_CLASSES}).")
    CLASS_NAMES = [f'Class {i}' for i in range(NUM_CLASSES)] # fallback

# --- model definition (copy from notebook, corrected EnsembleProjectors) ---
class TextEncoder(nn.Module):
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
        lengths = lengths.to(text.device)
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

# --- Corrected EnsembleProjectors ---
class EnsembleProjectors(nn.Module):
    def __init__(self, input_dim, proj_dim, num_projectors, dropout):
        super().__init__()
        self.num_projectors = num_projectors
        self.projectors = nn.ModuleList()
        for _ in range(num_projectors):
            self.projectors.append(
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.BatchNorm1d(input_dim // 2), # this layer might still cause runtime errors with batch_size=1
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim // 2, proj_dim)
                )
            )
        # fallback_projector removed from here

    def forward(self, x):
        # reverted to original forward method
        # note: batchnorm1d error might still occur if batch size is 1 during eval
        proj_outputs = [proj(x) for proj in self.projectors]
        ensemble_output = torch.stack(proj_outputs, dim=0).mean(dim=0)
        return ensemble_output

class EPASS_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, proj_dim, num_classes,
                 num_projectors, dropout, pad_idx):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim, hidden_dim, dropout, pad_idx)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = EnsembleProjectors(hidden_dim, proj_dim, num_projectors, dropout)

    def forward(self, text, lengths):
        features = self.encoder(text, lengths)
        logits = self.classifier(features)
        projection = self.projector(features) # projection not used for inference
        return logits, projection

# --- preprocessing function (copy from notebook) ---
def simple_tokenizer(text):
    return text.lower().split()

# --- load model and vocabulary (cached) ---
@st.cache_resource
def load_model_and_vocab(model_path, vocab_path, device):
    if not os.path.exists(vocab_path):
        st.error(f"vocabulary file not found at {vocab_path}. please run prepare_artifacts.py.")
        return None, None, None, None
    if not os.path.exists(model_path):
        st.error(f"model file not found at {model_path}.")
        return None, None, None, None

    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        st.success("vocabulary loaded.")

        if PAD_TOKEN not in vocab:
             st.error(f"'{PAD_TOKEN}' not found in vocabulary.")
             return None, None, None, None
        if UNK_TOKEN not in vocab:
             st.error(f"'{UNK_TOKEN}' not found in vocabulary.")
             return None, None, None, None

        PAD_IDX = vocab[PAD_TOKEN]
        UNK_IDX = vocab[UNK_TOKEN]
        VOCAB_SIZE = len(vocab)

        model = EPASS_Model(
            vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
            proj_dim=PROJ_DIM, num_classes=NUM_CLASSES, num_projectors=NUM_PROJECTORS,
            dropout=DROPOUT, pad_idx=PAD_IDX
        )
        # load state dict - map_location for cpu if needed
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # set model to evaluation mode
        st.success(f"model loaded onto {device}.")
        return model, vocab, PAD_IDX, UNK_IDX
    except Exception as e:
        st.error(f"error loading model or vocabulary: {e}")
        return None, None, None, None

# --- prediction function ---
def predict(text, model, vocab, tokenizer, pad_idx, unk_idx, device, class_names):
    if model is None or vocab is None:
        return "error: model/vocab not loaded.", 0.0

    model.eval()
    with torch.no_grad():
        tokens = tokenizer(str(text))
        indices = [vocab.get(token, unk_idx) for token in tokens]
        length = torch.tensor([len(indices)], dtype=torch.long)
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

        tensor = tensor.to(device)
        length = length.to(device)

        # --- Enhanced try-except block ---
        try:
            logits, _ = model(tensor, length)
            probabilities = F.softmax(logits, dim=1)
            top_prob, top_cat = probabilities.topk(1, dim=1)
            pred_idx = top_cat.item()
            confidence = top_prob.item()

            if 0 <= pred_idx < len(class_names):
                return class_names[pred_idx], confidence
            else:
                return f"error: predicted index {pred_idx} out of range.", 0.0
        except RuntimeError as e: # catch runtimeerror specifically
            # check if it's the common batchnorm1d error with batch size 1
            if "batchnorm1d" in str(e).lower() and "expected more than 1 value per channel" in str(e).lower():
                 st.error(f"prediction failed due to a batchnorm1d issue when processing a single input: {e}. this is a known limitation during evaluation with a batch size of 1.")
                 return f"prediction error (batchnorm)", 0.0
            else:
                 # re-raise other runtime errors or handle them generically
                 st.error(f"a runtime error occurred during prediction: {e}")
                 return f"prediction runtime error", 0.0
        except Exception as e: # catch other general exceptions
            st.error(f"an unexpected error occurred during prediction: {e}")
            return f"prediction error", 0.0

# --- streamlit ui ---
st.title("EPASS text classification demo (DBpedia)")
st.write(f"using device: {DEVICE}")

# load resources
model, vocab, PAD_IDX, UNK_IDX = load_model_and_vocab(MODEL_PATH, VOCAB_PATH, DEVICE)

# input text area
input_text = st.text_area("enter text to classify:", height=150, placeholder="example: the film tells the story of...")

# predict button
if st.button("classify text"):
    if model and vocab and input_text:
        with st.spinner("classifying..."):
            try:
                prediction, confidence = predict(input_text, model, vocab, simple_tokenizer, PAD_IDX, UNK_IDX, DEVICE, CLASS_NAMES)
                # check if prediction indicates an error occurred
                if "error" in prediction.lower():
                     # error message already displayed by predict function, do nothing more here
                     pass
                else:
                    st.success(f"predicted class: **{prediction}**")
                    st.metric(label="confidence", value=f"{confidence:.2%}")
            except Exception as e: # catch any unexpected error from predict itself (should be handled within)
                st.error(f"classification process failed: {e}")
    elif not input_text:
        st.warning("please enter some text to classify.")
    else:
         st.error("model or vocabulary failed to load. cannot classify.")