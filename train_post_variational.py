import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from data_loader import create_binary_datasets
from quantum_layer import quantum_feature_extractor_heuristic
from classical_nn import ClassicalNN


selected_heuristic = "ansatz"
device = torch.device("cuda")
# Load FashionMNIST dataset


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
            quantum_features = []
            for img in features:
                # Extract features for each image
                qf = quantum_feature_extractor_heuristic(
                    img, heuristic=selected_heuristic, locality=1
                )
                # Convert to tensor if it's not already one
                if not isinstance(qf, torch.Tensor):
                    qf = torch.tensor(qf, dtype=torch.float32)
                quantum_features.append(qf)
            quantum_features = torch.stack(quantum_features).float().to(device)

            quantum_features = quantum_features.view(quantum_features.shape[0], -1)
            outputs = model(quantum_features)  # Compute loss
            labels = labels.view(-1, 1).float().to(device)
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
            quantum_features = []
            for img in features:
                # Extract features for each image
                qf = quantum_feature_extractor_heuristic(
                    img, heuristic=selected_heuristic, locality=1
                )
                # Convert to tensor if it's not already one
                if not isinstance(qf, torch.Tensor):
                    qf = torch.tensor(qf, dtype=torch.float32)
                quantum_features.append(qf)
            quantum_features = torch.stack(quantum_features).float().to(device)

            # Forward pass
            quantum_features = quantum_features.view(quantum_features.shape[0], -1)
            output = model(quantum_features)  # Compute loss
            labels = labels.view(-1, 1).float().to(device)
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
model = ClassicalNN(input_size=256).to(device)

history, test_loss, test_accuracy = train_and_evaluate(
    model, train_loader, val_loader, test_loader, device, num_epochs=100, lr=0.001
)
