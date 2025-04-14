import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_batch_mean(Z, y, num_classes, feature_dim, batch_size, device):
    sums = torch.zeros(num_classes, feature_dim, dtype=Z_l.dtype, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float, device=device)
    sums = sums.index_add(0, y_l, Z_l, alpha=1)
    ones = torch.ones(batch_size, dtype=torch.float, device=device)
    counts = counts.index_add(0, y_l, ones)
    counts[counts == 0] = 1.0
    avg_values = sums / counts.unsqueeze(1)
    return avg_values


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

def update_beta_parameters(w_id, w_ood, s_l, s_u ,alpha_id, beta_id, alpha_ood, beta_ood, l):
    # Detaching from computational graph so that it won't cause some bullshit error
    w_id = w_id.squeeze(1).detach(); w_ood = w_ood.squeeze(1).detach()
    s_l = s_l.squeeze(1).detach(); s_u = s_u.squeeze(1).detach()
    nu_id = torch.sum(s_l) + torch.dot(w_id, s_u)/(s_l.size(0) + torch.sum(w_id))
    sigma2_id = (torch.sum(s_l - nu_id)**2 + torch.dot(w_id, (s_u - nu_id)**2))/(s_l.size(0) + torch.sum(w_id))

    nu_ood = torch.dot(w_ood, s_u)/(torch.sum(w_ood))
    sigma2_ood = torch.dot(w_ood, (s_u - nu_id)**2)/((torch.sum(w_ood)))

    _alpha_id = nu_id*((nu_id*(1-nu_id))/sigma2_id - 1)
    _alpha_ood = nu_ood*((nu_ood*(1-nu_ood))/sigma2_ood - 1)
    _beta_id = (1 - nu_id)*((nu_id*(1-nu_id))/sigma2_id - 1)
    _beta_ood = (1 - nu_ood)*((nu_ood*(1-nu_ood))/sigma2_ood - 1)

    return alpha_id*l + (1 - l)*_alpha_id, beta_id*l + (1 - l)*_beta_id, alpha_ood*l + (1 - l)*_alpha_ood, beta_ood*l + (1 - l)*_beta_ood

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


class Classifier(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.classifier = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.classifier(x)

class InferenceModelText:
    def __init__(self, backbone_pth, classifier_pth, batch_mean_pth, beta_parameters_pth):
        self.feature_dim = 32
        self.num_classes = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')

        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            'prajjwal1/bert-tiny',
            num_labels=self.feature_dim
        )
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

    def _preprocess(self, text):
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def __call__(self, text):
        inputs = self._preprocess(text)
        with torch.no_grad():
            outputs = self.backbone(**inputs)
            Z = outputs.logits 

            s_score = subspace_score(Z, self.batch_means)
            p_id = get_p_id(
                s_score,
                self.alpha_id.to(self.device),
                self.beta_id.to(self.device),
                self.alpha_ood.to(self.device),
                self.beta_ood.to(self.device)
            )
            if p_id[0] < torch.rand(1, device=self.device):
                return -1
            y = self.classifier(Z)
            return torch.argmax(y, dim=1)

def get_prediction(text):
    backbone = "weights/backbone_epoch_9.pth" 
    class_mean = "weights/batch_means.pth"
    classifier = "weights/classifier_epoch_9.pth"
    beta_parameters = "weights/beta_parameters.pth"
    model = InferenceModelText(backbone, classifier, class_mean, beta_parameters)
    output = model(text)
    if(output == -1):
        return -1
    else: 
        return output.item()
    



