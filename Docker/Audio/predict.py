"Required Imports"
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from collections import defaultdict
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import os
from itertools import compress
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

"Taken from the code given by the authors... "
def beta_pdf(x, alpha, beta, loc=0.0, scale=1.0):
    x = (x - loc) / scale
    device = x.device
    alpha = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta = torch.as_tensor(beta, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

    def xlogy(a, b):
        return torch.where(a == 0, torch.zeros_like(b), a * torch.log(b + 1e-10))

    def xlog1py(a, y):
        return torch.where(a == 0, torch.zeros_like(y), a * torch.log1p(y + 1e-10))

    log_unnormalized = xlogy(alpha - 1.0, x) + xlog1py(beta - 1.0, -x)
    log_normalization = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    log_prob = log_unnormalized - log_normalization
    log_prob = torch.where((x >= 0) & (x <= 1), log_prob, torch.tensor(float('-inf'), device=device))

    return torch.exp(log_prob) / scale

def subspace_score(Z, batch_means):
    """
    Z: (batch_size, feature_dim)
    batch_means: (num_classes, feature_dim)
    Returns:
        cosine similarity between Z and its projection on the subspace
        spanned by batch_means — shape (batch_size, 1)
    """
    Q, _ = torch.linalg.qr(batch_means.T)
    proj_Z = Z @ Q @ Q.T
    cos_sim = F.cosine_similarity(Z, proj_Z, dim=1).clamp(0, 1)
    return cos_sim.unsqueeze(1)

class Classifier(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.classifier = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.classifier(x)
    "Taken from the code given by the authors... "
def beta_pdf(x, alpha, beta, loc=0.0, scale=1.0):
    x = (x - loc) / scale
    alpha = torch.as_tensor(alpha, dtype=torch.float32, device=x.device)
    beta = torch.as_tensor(beta, dtype=torch.float32, device=x.device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    def xlogy(a, b):
        return torch.where(a == 0, torch.zeros_like(b), a * torch.log(b + 1e-10))
    def xlog1py(a, y):
        return torch.where(a == 0, torch.zeros_like(y), a * torch.log1p(y + 1e-10))

    log_unnormalized = xlogy(alpha - 1.0, x) + xlog1py(beta - 1.0, -x)
    log_normalization = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    log_prob = log_unnormalized - log_normalization
    log_prob = torch.where((x >= 0) & (x <= 1), log_prob, torch.tensor(float('-inf'), device=x.device))

    return torch.exp(log_prob) / scale

def get_p_id(s, alpha_id, beta_id, alpha_ood, beta_ood, pi=1.0):
    """
    Compute the ID probability using Beta PDFs instead of torch.distributions.Beta,
    following the formulation in the paper (with loc/scale support implicitly assumed as 0/1).
    """
    beta_pdf_id = beta_pdf(s, alpha_id, beta_id)
    beta_pdf_ood = beta_pdf(s, alpha_ood, beta_ood)
    numerator = beta_pdf_id * pi
    denominator = numerator + beta_pdf_ood * (1 - pi)
    p_id = numerator / (denominator + 1e-8)
    return p_id

class InferenceModelAudio:
    def __init__(self, backbone_pth, classifier_pth, batch_mean_pth, beta_parameters_pth):
        self.feature_dim = 128
        self.num_classes = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.num_ftrs, self.feature_dim)
        self.backbone.load_state_dict(torch.load(backbone_pth, map_location=self.device))
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        self.classifier = Classifier(self.feature_dim, self.num_classes)
        self.classifier.load_state_dict(torch.load(classifier_pth, map_location=self.device))
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        self.batch_means = torch.load(batch_mean_pth, map_location=self.device).to(self.device)
        beta_parameters = torch.load(beta_parameters_pth, map_location=self.device).to(self.device)
        self.alpha_id, self.beta_id, self.alpha_ood, self.beta_ood = beta_parameters.chunk(4)

        self.weak_aug = T.Vol(gain=torch.FloatTensor(1).uniform_(0.9, 1.1).item(), gain_type="amplitude")
        self.sample_rate = 16000
        self.target_length = 200
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        ).to(self.device)
        self.db_transform = T.AmplitudeToDB().to(self.device)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate).to(self.device)
            waveform = resampler(waveform.to(self.device))
        else:
            waveform = waveform.to(self.device)
        return waveform

    def _to_log_mel(self, x):
        mel = self.mel_transform(x)
        log_mel = self.db_transform(mel)
        return log_mel

    def _resize_spectrogram(self, x):
        current_length = x.size(2)
        if current_length < self.target_length:
            padding = self.target_length - current_length
            x = F.pad(x, (0, padding))
        elif current_length > self.target_length:
            x = x[:, :, :self.target_length]
        return x

    def get_spectogram(self, audio_path):
        audio = self._load_audio(audio_path)
        log_mel = self._to_log_mel(audio)
        log_mel = self._resize_spectrogram(log_mel)
        return log_mel


    def __call__(self, audio_wav):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        spectogram = self.get_spectogram(audio_wav).unsqueeze(0)
        spectogram = spectogram.to(self.device)
        Z = self.backbone(spectogram)
        s_score = subspace_score(Z, self.batch_means)
        p_id = get_p_id(s_score, self.alpha_id, self.beta_id, self.alpha_ood, self.beta_ood)
        uniform = torch.rand(1).to(device)
        if(p_id[0] < uniform[0]):
            return -1
        y = self.classifier(Z)
        return torch.argmax(y, dim=1)


def get_prediction(audio_wav):
    backbone = "weights/backbone_fsd_kaggle.pth"
    batch_mean = "weights/batch_means_fsd_kaggle.pth"
    beta_parameters = "weights/beta_parameters_fsd_kaggle.pth"
    classifier = "weights/classifier_epoch_fsd_kaggle.pth"
    model = InferenceModelAudio(backbone, classifier, batch_mean, beta_parameters)
    output = model(audio_wav)
    if(output == -1):
        return output
    else:
        return output.item()




