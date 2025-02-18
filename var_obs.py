import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from data_loader import create_binary_datasets
from tqdm import tqdm
from models.quantum_layer import create_obs_circuit, accuracy


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: F.max_pool2d(
                x.unsqueeze(0),
                kernel_size=(14, 7),  # Vertical 14px, Horizontal 7px
                stride=(14, 7),
            ).squeeze(0)
        ),
        transforms.Lambda(lambda x: x.flatten()),  #  Now 8 features
        transforms.Lambda(lambda x: x * (2 * np.pi)),
    ]
)

full_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

# Create binary datasets for two chosen classes (e.g. class 4 and class 6)
train_dataset, val_dataset, test_dataset = create_binary_datasets(
    full_dataset, class_1=4, class_2=6, train_size=200, val_size=50, test_size=50
)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def variational_classifier(circuit, params, bias, features):
    """
    Given a quantum circuit, parameters, and bias,
    returns the classification prediction.

    If the circuit returns a list (multiple expectation values),
    we combine them linearly using fixed weights (or you can also train these).
    Here, for simplicity, we average the outputs.
    """
    outputs = circuit(params, features)
    # If the output is a list (multiple observables), average them.
    if isinstance(outputs, (list, tuple)):
        output = torch.mean(torch.stack(outputs))
    else:
        output = outputs
    return output + bias


def rmse_loss(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


for locality in range(1, 4):
    print(f"\nTraining with {locality}-local observables:")

    # Create the quantum circuit for the given locality.
    circuit = create_obs_circuit(locality)

    # Initialize variational parameters:
    params = torch.nn.Parameter(0.01 * torch.randn(16))
    bias = torch.nn.Parameter(torch.tensor(0.0))

    optimizer = optim.Adam([params, bias], lr=0.01)
    num_epochs = 100

    for epoch in range(num_epochs):
        # Training phase.
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0
        # Process training data.
        for features_batch, targets_batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch_preds = []
            # Process each sample individually (avoid vmap issues).
            for features, target in zip(features_batch, targets_batch):
                # Ensure features is a float tensor.
                pred = variational_classifier(circuit, params, bias, features.float())
                batch_preds.append(pred)
            batch_preds = torch.stack(batch_preds)
            loss = rmse_loss(batch_preds, targets_batch.float())
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * features_batch.size(0)
            train_samples += features_batch.size(0)
            train_correct += accuracy(
                batch_preds, targets_batch.float()
            ) * features_batch.size(0)
        train_loss = train_loss_total / train_samples
        train_acc = train_correct / train_samples

        # Validation phase.
        val_loss_total = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for features_batch, targets_batch in val_loader:
                batch_preds = []
                for features, target in zip(features_batch, targets_batch):
                    pred = variational_classifier(
                        circuit, params, bias, features.float()
                    )
                    batch_preds.append(pred)
                batch_preds = torch.stack(batch_preds)
                loss = rmse_loss(batch_preds, targets_batch.float())
                val_loss_total += loss.item() * features_batch.size(0)
                val_samples += features_batch.size(0)
                val_correct += accuracy(
                    batch_preds, targets_batch.float()
                ) * features_batch.size(0)
        val_loss = val_loss_total / val_samples
        val_acc = val_correct / val_samples

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d} | Train RMSE: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val RMSE: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    # After training, evaluate on the test set.
    test_loss_total = 0.0
    test_correct = 0
    test_samples = 0
    with torch.no_grad():
        for features_batch, targets_batch in test_loader:
            batch_preds = []
            for features, target in zip(features_batch, targets_batch):
                pred = variational_classifier(circuit, params, bias, features.float())
                batch_preds.append(pred)
            batch_preds = torch.stack(batch_preds)
            loss = rmse_loss(batch_preds, targets_batch.float())
            test_loss_total += loss.item() * features_batch.size(0)
            test_samples += features_batch.size(0)
            test_correct += accuracy(
                batch_preds, targets_batch.float()
            ) * features_batch.size(0)
    test_loss = test_loss_total / test_samples
    test_acc = test_correct / test_samples
    print(f"Final Test RMSE: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n")
