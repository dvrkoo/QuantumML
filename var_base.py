import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pennylane as qml
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from data_loader import create_binary_datasets

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
        transforms.Lambda(lambda x: x * (2 * np.pi)),  # Scale to [0, 2π)
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

# Quantum setup
n_qubits = 8  # Matches 2x4 pooled features
dev = qml.device("lightning.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def quantum_circuit(params, features):
    # Feature embedding
    for i in range(n_qubits):
        qml.RX(features[i], wires=i)

    # Simple variational ansatz
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)

    # Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    return qml.expval(qml.PauliZ(0))


class QuantumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return quantum_circuit(self.params, x) + self.bias


def train_model(model, num_epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()  # For regression-style output
    best_val_acc = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = torch.stack([model(x.flatten()) for x in inputs])
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.sign(outputs)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = {"state_dict": model.state_dict(), "val_acc": val_acc}

        # Print progress
        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | "
            f"Train Loss: {train_loss/total_train:.4f} | "
            f"Train Acc: {correct_train/total_train:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # Load best model for testing
    model.load_state_dict(best_model["state_dict"])
    test_loss, test_acc = evaluate_model(model, test_loader)
    print(f"\nBest Validation Acc: {best_model['val_acc']:.4f}")
    print(f"Final Test Acc: {test_acc:.4f}")

    return model


def evaluate_model(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = torch.stack([model(x.flatten()) for x in inputs])
            loss = criterion(outputs, labels.float())

            total_loss += loss.item() * inputs.size(0)
            preds = torch.sign(outputs)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# Initialize and train
model = QuantumModel()
trained_model = train_model(model)
