import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import create_binary_datasets
from torchvision import datasets, transforms
import numpy as np

# set seed
torch.manual_seed(0)

# Data preprocessing
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.nn.functional.max_pool2d(
                x.unsqueeze(0), kernel_size=(14, 7), stride=(14, 7)
            ).squeeze()
        ),
        transforms.Lambda(lambda x: x.flatten()),
    ]
)

# Load datasets
full_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
train_dataset, val_dataset, test_dataset = create_binary_datasets(
    full_dataset, class_1=4, class_2=6, train_size=200, val_size=50, test_size=50
)

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 128),  # Input: 8 features (2x4 pooled)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output: binary classification
        )

    def forward(self, x):
        return self.layers(x.flatten(1))


def train_mlp(model, train_loader, val_loader, num_epochs=400):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    best_acc = 0.0
    best_model = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Convert targets to 0/1 for BCE loss
            targets_01 = ((targets + 1) / 2).float()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets_01)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.sign(outputs)  # Convert to -1/1
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        # Validation
        val_loss, val_acc = evaluate_mlp(model, val_loader)

        # Save best model
        if val_acc > best_acc:
            best_loss = val_loss
            best_acc = val_acc
            best_model = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss/total:.4f} | "
                f"Train Acc: {correct/total:.4f} | "
                f"Val Loss: {val_loss:.4f} |"
                f"Val Acc: {val_acc:.4f}"
            )

    # Load best model for testing
    model.load_state_dict(best_model)
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best Validation Loss: {best_loss:.4f}")

    return model


def evaluate_mlp(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, targets in loader:
            targets_01 = ((targets + 1) / 2).float()
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets_01)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.sign(outputs)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total


def test_mlp(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets_01 = ((targets + 1) / 2).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets_01)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.sign(outputs)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return (correct / total), total_loss / total


# Usage
mlp = SimpleMLP()
trained_mlp = train_mlp(mlp, train_loader, val_loader)
test_acc, test_loss = test_mlp(trained_mlp, test_loader)
print(f"Test Accuracy: {test_acc:.4f}" "|" f" Test Loss: {test_loss:.4f}")
