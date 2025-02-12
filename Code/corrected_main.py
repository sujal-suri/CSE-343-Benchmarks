
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define EPASS: Ensemble Projectors for Contrastive Learning
class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class EPASS(nn.Module):
    def __init__(self, base_model, num_projectors=3, output_dim=128):
        super(EPASS, self).__init__()
        self.base_model = base_model
        self.projectors = nn.ModuleList([Projector(2048, output_dim) for _ in range(num_projectors)])
        self.memory_bank = []  # Store embeddings

    def forward(self, x):
        features = self.base_model(x)
        ensemble_embeddings = torch.stack([proj(features) for proj in self.projectors], dim=0).mean(dim=0)
        self.memory_bank.append(ensemble_embeddings.detach())
        return ensemble_embeddings

# Define JointMatch: Adaptive Thresholding & Cross-Labeling
class JointMatch(nn.Module):
    def __init__(self, base_model, num_classes):
        super(JointMatch, self).__init__()
        self.model1 = base_model
        self.model2 = base_model  # Two different models for cross-labeling
        self.num_classes = num_classes
        self.thresholds = torch.ones(num_classes) * 0.95  # Adaptive thresholds

    def update_thresholds(self, pseudo_labels):
        # Adjust class-wise thresholds based on learning progress
        class_counts = torch.bincount(pseudo_labels, minlength=self.num_classes)
        class_probs = class_counts.float() / class_counts.sum()
        self.thresholds = 0.95 * (1 - class_probs) + 0.05

    def forward(self, x):
        outputs1 = self.model1(x)
        outputs2 = self.model2(x)

        # Cross-labeling: Use high-confidence predictions from one model to train the other
        pseudo_labels1 = torch.argmax(outputs1, dim=1)
        pseudo_labels2 = torch.argmax(outputs2, dim=1)

        self.update_thresholds(pseudo_labels1)

        high_confidence_mask1 = torch.max(outputs1, dim=1).values > self.thresholds[pseudo_labels1]
        high_confidence_mask2 = torch.max(outputs2, dim=1).values > self.thresholds[pseudo_labels2]

        return outputs1[high_confidence_mask2], outputs2[high_confidence_mask1]

# Load dataset (Tiny ImageNet or similar dataset)
dataset_path = "/home/momo/Documents/Datasets/tiny-imagenet-200"
train_path = dataset_path + "/train"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.ImageFolder(train_path, transform=transform)
train_data, val_data, test_data = random_split(dataset, [80000, 10000, 10000])

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Define Base Model (ResNet or any CNN backbone)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Instantiate models
base_model = SimpleCNN(num_classes=200)
epass_model = EPASS(base_model)
jointmatch_model = JointMatch(base_model, num_classes=200)

# Training Function
def train_model(model, train_loader, val_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Train models
train_model(epass_model, train_loader, val_loader, epochs=3)
train_model(jointmatch_model, train_loader, val_loader, epochs=3)

# Compare results
def compare_results(model1, model2, data_loader):
    model1.eval()
    model2.eval()
    correct1, correct2, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs1 = model1(images)
            outputs2 = model2(images)

            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            correct1 += (predicted1 == labels).sum().item()
            correct2 += (predicted2 == labels).sum().item()
            total += labels.size(0)

    acc1 = 100 * correct1 / total
    acc2 = 100 * correct2 / total
    print(f"EPASS Accuracy: {acc1:.2f}%, JointMatch Accuracy: {acc2:.2f}%")

# Evaluate
compare_results(epass_model, jointmatch_model, val_loader)
