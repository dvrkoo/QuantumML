import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from classical_nn import ClassicalNN
from quantum_layer import quantum_feature_extractor_heuristic
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
from torch.utils.data import Dataset

selected_heuristic = "ansatz"
device = torch.device("cuda")
# Load FashionMNIST dataset


class BinaryFashionMNIST(Dataset):
    def __init__(self, dataset, class_1=4, class_2=6):
        # Filter indices for only the two classes we want (e.g. coat (4) and shirt (6))
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
        # Convert label to binary: 0 for class_1, 1 for class_2
        binary_label = 0 if label == self.class_1 else 1
        return image, binary_label


def create_binary_datasets(
    full_dataset, class_1=4, class_2=6, train_size=200, val_size=50, test_size=50
):
    """
    Splits the full dataset into training, validation, and test subsets.
    - train: first (train_size * 2) samples (from both classes)
    - val: next (val_size * 2) samples
    - test: next (test_size * 2) samples
    """
    binary_dataset = BinaryFashionMNIST(full_dataset, class_1, class_2)
    train_total = train_size * 2
    val_total = val_size * 2
    test_total = test_size * 2

    train_dataset = Subset(binary_dataset, range(0, train_total))
    val_dataset = Subset(binary_dataset, range(train_total, train_total + val_total))
    test_dataset = Subset(
        binary_dataset,
        range(train_total + val_total, train_total + val_total + test_total),
    )
    return train_dataset, val_dataset, test_dataset


# Setup transforms and load the full FashionMNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

full_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

# Create train, validation, and test datasets
train_dataset, val_dataset, test_dataset = create_binary_datasets(
    full_dataset, train_size=200, val_size=50, test_size=50
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def evaluate_model(model, loader, criterion, device):
    """
    Evaluate the model on a given dataset loader.
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            # Process features: from (batch, 1, 28, 28) to (batch, 8, 8)
            features = images.view(images.shape[0], 28, 28)
            features = torch.nn.functional.interpolate(
                features.unsqueeze(1), size=(16, 16)
            ).squeeze(1)

            # Convert features using quantum extractor for each image in the batch
            quantum_features = [
                torch.tensor(
                    quantum_feature_extractor_heuristic(
                        img,
                        heuristic=selected_heuristic,
                        locality=1,
                    )
                )
                for img in features
            ]
            quantum_features = torch.stack(quantum_features).float().to(device)

            outputs = model(quantum_features)
            # Compute loss
            labels = labels.unsqueeze(1).float().to(device)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            # For evaluation, threshold the logits at 0 (since BCEWithLogitsLoss is used)
            predictions = (outputs > 0).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scheduler,
    num_epochs=50,
):
    """
    Train the model and perform validation at every epoch.
    Returns a history dictionary containing train loss, validation loss, and validation accuracy.
    """
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)", leave=False
        ):
            # Process features: reshape to (batch, 28, 28) then downsample to (batch, 8, 8)
            features = images.view(images.shape[0], 28, 28)
            features = torch.nn.functional.interpolate(
                features.unsqueeze(1), size=(16, 16)
            ).squeeze(1)

            # Extract quantum features for each image in the batch
            quantum_features = torch.tensor(
                [
                    quantum_feature_extractor_heuristic(
                        img, heuristic=selected_heuristic, locality=1
                    )
                    for img in features
                ]
            )
            quantum_features = torch.stack(quantum_features).float().to(device)

            # Forward pass
            output = model(quantum_features)

            # Compute loss
            labels = labels.unsqueeze(1).float().to(device)
            loss = criterion(output, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        avg_train_loss = total_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # Validation at every epoch
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy*100:.2f}%"
        )

    return history


def train_and_evaluate(
    model, train_loader, val_loader, test_loader, device, num_epochs=200, lr=0.0001
):
    """
    Complete training and evaluation pipeline.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=5, verbose=True
    )

    print("Starting training...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler,
        num_epochs,
    )

    print("\nFinal evaluation on test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")

    return history, test_loss, test_accuracy


###############################
# Set device and run training
###############################

# For example, on macOS with MPS:
device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
model = ClassicalNN(input_size=16 if selected_heuristic == "ansatz" else 32).to(device)

history, test_loss, test_accuracy = train_and_evaluate(
    model, train_loader, val_loader, test_loader, device, num_epochs=100, lr=0.01
)
