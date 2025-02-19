import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from data_loader import create_binary_datasets
from tqdm import tqdm
from models.quantum_layer import create_post_variational_system

# set seed
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Downscale to 2x4 (8 features) using different pooling
        transforms.Lambda(
            lambda x: F.max_pool2d(
                x.unsqueeze(0),
                kernel_size=(14, 7),  # Vertical 14px, Horizontal 7px
                stride=(14, 7),
            ).squeeze(0)
        ),
        # Flatten to 8 features
        transforms.Lambda(lambda x: x.flatten()),
        # Rescale remains the same
        transforms.Lambda(lambda x: x * (2 * np.pi)),
    ]
)
# Assume create_binary_datasets is defined elsewhere
full_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

train_dataset, val_dataset, test_dataset = create_binary_datasets(
    full_dataset, class_1=4, class_2=6, train_size=200, val_size=50, test_size=50
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def hybrid_processor(circuit, mlp, params, features, shift_vectors, locality):
    quantum_features = []

    for shift in shift_vectors:
        shifted_params = params + shift
        # Convert circuit outputs to proper tensor
        obs_values = torch.stack(
            circuit(shifted_params, features)
        )  # Shape: [num_observables]
        quantum_features.append(obs_values)

    # Combine features across shifts
    features_tensor = torch.cat(
        quantum_features
    )  # Shape: [num_shifts * num_observables]
    features_tensor = features_tensor
    return mlp(features_tensor.unsqueeze(0))  # Add batch dimension


def validate(model, params, data_loader, shift_vectors, locality, criterion, circuit):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            batch_preds = []
            fixed_targets = ((targets + 1) / 2).float()  # now 0 or 1
            for x, y in zip(features, targets):
                pred = hybrid_processor(
                    circuit, model, params, x.flatten(), shift_vectors, locality
                )
                batch_preds.append(pred)

            batch_preds = torch.cat(batch_preds).squeeze(-1)
            loss = criterion(batch_preds, fixed_targets)

            total_loss += loss.item() * len(targets)
            pred_class = torch.sign(batch_preds)
            correct += (pred_class == targets).sum().item()
            total_samples += len(targets)

    return total_loss / total_samples, correct / total_samples


def test(model, params, data_loader, shift_vectors, locality, criterion, circuit):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            batch_preds = []
            fixed_targets = ((targets + 1) / 2).float()  # now 0 or 1
            for x, y in zip(features, targets):
                pred = hybrid_processor(
                    circuit, model, params, x.flatten(), shift_vectors, locality
                )
                batch_preds.append(pred)

            batch_preds = torch.cat(batch_preds).squeeze(-1)
            loss = criterion(batch_preds, fixed_targets)

            total_loss += loss.item() * len(targets)
            pred_class = torch.sign(batch_preds)
            correct += (pred_class == targets).sum().item()
            total_samples += len(targets)

    return total_loss / total_samples, correct / total_samples


def train_post_variational(locality=1, shift_order=1):
    circuit, mlp, shift_vectors = create_post_variational_system(locality, shift_order)
    params = torch.nn.Parameter(0.01 * torch.randn(16))
    optimizer = optim.Adam(
        [{"params": params, "lr": 0.01}, {"params": mlp.parameters(), "lr": 0.0001}]
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_params = None

    for epoch in range(35):
        # Training phase
        mlp.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch_preds = []

            fixed_targets = ((targets + 1) / 2).float()  # now 0 or 1
            for x, y in zip(features, targets):
                pred = hybrid_processor(
                    circuit, mlp, params, x.flatten(), shift_vectors, locality
                )
                batch_preds.append(pred)

            batch_preds = torch.cat(batch_preds).squeeze(-1)

            loss = criterion(batch_preds, fixed_targets)
            loss.backward()
            optimizer.step()

            # Track training metrics
            train_loss += loss.item() * len(targets)
            pred_class = torch.sign(batch_preds)
            train_correct += (pred_class == targets).sum().item()
            total_samples += len(targets)

        # Validation phase
        val_loss, val_acc = validate(
            mlp, params, val_loader, shift_vectors, locality, criterion, circuit
        )
        train_loss = train_loss / total_samples
        train_acc = train_correct / total_samples

        # Save best model
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_params = params.clone()
            torch.save(mlp.state_dict(), "best_mlp.pth")

        # Print metrics
        print(f"\nEpoch {epoch+1:03d}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Final testing with best parameters
    mlp.load_state_dict(torch.load("best_mlp.pth"))
    test_loss, test_acc = test(
        mlp, best_params, test_loader, shift_vectors, locality, criterion, circuit
    )
    print(f"Best val loss {best_val_loss:.4f} | Best val acc {best_val_acc:.4f}")
    print(f"\nFinal Test Metrics: Loss {test_loss:.4f} | Acc {test_acc:.4f}")

    return mlp, best_params


# Train the model
mlp, best_params = train_post_variational(locality=2, shift_order=1)
