import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from classical_nn import ClassicalNN
from quantum_layer import quantum_feature_extractor
import torch.optim as optim
import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("mps")
# Load FashionMNIST dataset
from torch.utils.data import Dataset


class BinaryFashionMNIST(Dataset):
    def __init__(self, dataset, class_1=4, class_2=6):
        # Filter indices for only the two classes we want
        self.indices = [
            i for i, (_, label) in enumerate(dataset) if label in [class_1, class_2]
        ]
        self.dataset = dataset
        self.class_1 = class_1
        self.class_2 = class_2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        # Convert label to binary (0 or 1)
        binary_label = 0 if label == self.class_1 else 1
        return image, binary_label


# Usage
def create_binary_datasets(
    full_dataset, class_1=4, class_2=6, train_size=200, test_size=50
):
    # Create binary dataset
    binary_dataset = BinaryFashionMNIST(full_dataset, class_1, class_2)

    # Split into train and test
    train_dataset = torch.utils.data.Subset(binary_dataset, range(train_size * 2))
    test_dataset = torch.utils.data.Subset(
        binary_dataset, range(train_size * 2, train_size * 2 + test_size * 2)
    )

    return train_dataset, test_dataset


# Setup code
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

full_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

# Create binary datasets
train_dataset, test_dataset = create_binary_datasets(full_dataset)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    """
    Train the model and return training history.
    """
    history = {"loss": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            # Process features
            features = images.view(images.shape[0], 28, 28)
            features = torch.nn.functional.interpolate(
                features.unsqueeze(1), size=(8, 8)
            ).squeeze(1)

            # Convert features using quantum extractor
            quantum_features = [
                torch.tensor(quantum_feature_extractor(img)) for img in features
            ]
            quantum_features = torch.stack(quantum_features).float().to(device)

            # Forward pass
            output = model(quantum_features)

            # Compute loss
            labels = labels.unsqueeze(1).float().to(device)
            loss = criterion(output, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        history["loss"].append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return history


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model and return accuracy and predictions.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # Process features
            features = images.view(images.shape[0], 28, 28)
            features = torch.nn.functional.interpolate(
                features.unsqueeze(1), size=(8, 8)
            ).squeeze(1)

            # Convert features using quantum extractor
            quantum_features = [
                torch.tensor(quantum_feature_extractor(img)) for img in features
            ]
            quantum_features = torch.stack(quantum_features).float().to(device)

            # Forward pass
            outputs = model(quantum_features)
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            # Calculate accuracy
            labels = labels.unsqueeze(1).float().to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Store predictions and labels
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_labels


# Example usage:
def train_and_evaluate(
    model, train_loader, test_loader, device, num_epochs=400, lr=0.01
):
    """
    Complete training and evaluation pipeline.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    print("Starting training...")
    history = train_model(model, train_loader, criterion, optimizer, device, num_epochs)

    # Evaluate the model
    print("\nEvaluating model...")
    accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    return history, accuracy, predictions, true_labels


# To use:
model = ClassicalNN().to(device)
history, accuracy, preds, labels = train_and_evaluate(
    model, train_loader, test_loader, device
)
