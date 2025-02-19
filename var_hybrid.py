import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from data_loader import create_binary_datasets
import torch.nn.functional as F
from models.quantum_layer import create_hybrid_circuit, deriv_params, accuracy


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


def hybrid_classifier(circuit, params, bias, features, shift_vectors, locality):
    """
    Combines:
    1. Parameter shift expansion
    2. Pauli observable combination
    """
    all_outputs = []

    # Evaluate all parameter shifts
    for shift in shift_vectors:
        shifted_params = params + shift

        # Get all k-local observables
        observables = circuit(shifted_params, features)

        # Combine observables: sum or average
        combined = (
            torch.sum(torch.stack(observables))
            if locality > 1
            else torch.mean(torch.stack(observables))
        )

        all_outputs.append(combined)

    # Average across shifts and add bias
    return torch.mean(torch.stack(all_outputs)) + bias


def rmse_loss(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def train_hybrid_model(locality=1, shift_order=1, num_epochs=30):
    # Quantum circuit configuration
    num_params = 16

    # Generate parameter shift vectors
    shift_vectors = torch.tensor(
        deriv_params(thetas=num_params, order=shift_order), dtype=torch.float
    )

    # Initialize quantum components
    circuit = create_hybrid_circuit(locality)

    # Model parameters
    params = torch.nn.Parameter(0.01 * torch.randn(num_params))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = optim.Adam([params, bias], lr=0.01)

    # Metrics storage
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": None,
    }

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        # params.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch_preds = []

            # Process each sample in batch
            for x, y in zip(features, targets):
                pred = hybrid_classifier(
                    circuit, params, bias, x.flatten(), shift_vectors, locality
                )
                batch_preds.append(pred)

            batch_preds = torch.stack(batch_preds)
            loss = rmse_loss(batch_preds, targets.float())
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item() * len(targets)
            train_total += len(targets)
            train_correct += accuracy(batch_preds, targets) * len(targets)

        # Validation phase
        # params.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for features, targets in val_loader:
                batch_preds = []
                for x, y in zip(features, targets):
                    pred = hybrid_classifier(
                        circuit, params, bias, x.flatten(), shift_vectors, locality
                    )
                    batch_preds.append(pred)

                batch_preds = torch.stack(batch_preds)
                val_loss += rmse_loss(batch_preds, targets.float()).item() * len(
                    targets
                )
                val_total += len(targets)
                val_correct += accuracy(batch_preds, targets) * len(targets)

        # Store epoch metrics
        metrics["train_loss"].append(train_loss / train_total)
        metrics["val_loss"].append(val_loss / val_total)
        metrics["train_acc"].append(train_correct / train_total)
        metrics["val_acc"].append(val_correct / val_total)

        # Print progress
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {metrics['train_loss'][-1]:.4f} | "
            f"Val Loss: {metrics['val_loss'][-1]:.4f} | "
            f"Train Acc: {metrics['train_acc'][-1]:.4f} | "
            f"Val Acc: {metrics['val_acc'][-1]:.4f}"
        )

    # Final testing
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for features, targets in test_loader:
            batch_preds = []
            for x, y in zip(features, targets):
                pred = hybrid_classifier(
                    circuit, params, bias, x.flatten(), shift_vectors, locality
                )
                batch_preds.append(pred)

            batch_preds = torch.stack(batch_preds)
            test_total += len(targets)
            test_correct += accuracy(batch_preds, targets) * len(targets)

    metrics["test_acc"] = test_correct / test_total
    print(f"\nFinal Test Accuracy: {metrics['test_acc']:.4f}")

    return metrics, params, bias


# -----------------------------------------------------------------------------
# Hyperparameter Grid Search
# -----------------------------------------------------------------------------

# Test different combinations
results = []
for locality in [1, 2]:  # Measurement locality
    for shift_order in [1, 2]:  # Parameter shift order
        print(f"\nTraining Locality={locality}, Shift Order={shift_order}")
        # Get all return values but only store what we need
        metrics, _, _ = train_hybrid_model(locality, shift_order)
        test_acc = metrics["test_acc"]
        results.append((locality, shift_order, test_acc))
        print(f"Locality {locality}, Shift {shift_order}: Acc {test_acc:.2f}")

# Visualization
print("\nPerformance Matrix:")
print("Shift Order → Locality ↓")
print("      1       2")
for loc in [1, 2]:
    line = [f"{acc:.2f}" for _, order, acc in results if _ == loc]
    print(f" {loc}     {'   '.join(line)}")
