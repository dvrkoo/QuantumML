import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data_loader import create_binary_datasets


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


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)  # 8 input features

    def forward(self, x):
        return self.linear(x.flatten(1)).squeeze()


def train_logreg(model, train_loader, val_loader, num_epochs=600):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    best_val_acc = 0.0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Convert targets to 0/1
            targets_01 = ((targets + 1) / 2).float()

            outputs = model(inputs)
            loss = criterion(outputs, targets_01)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.sign(outputs)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        # Validation
        val_loss, val_acc = evaluate_logreg(model, val_loader)

        # Save best model
        if val_acc > best_val_acc:
            best_loss = val_loss
            best_val_acc = val_acc
            best_model = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss/total:.4f} | "
                f"Train Acc: {correct/total:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    # Load best model for testing
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    model.load_state_dict(best_model)
    return model


def evaluate_logreg(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, targets in loader:
            targets_01 = ((targets + 1) / 2).float()
            outputs = model(inputs)

            loss = criterion(outputs, targets_01)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.sign(outputs)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total


def test_logreg(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets_01 = ((targets + 1) / 2).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets_01)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.sign(outputs)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total, total_loss / total


# Usage
logreg = LogisticRegression()
trained_logreg = train_logreg(logreg, train_loader, val_loader)
test_acc, test_loss = test_logreg(trained_logreg, test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
