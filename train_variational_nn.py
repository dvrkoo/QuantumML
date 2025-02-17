import pennylane as qml
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data_loader import create_binary_datasets
from torchvision import datasets, transforms
from quantum_layer import base_ansatz_circuit


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Apply max pooling over 7x7 patches to reduce from 28x28 to 4x4.
        transforms.Lambda(
            lambda x: F.max_pool2d(x.unsqueeze(0), kernel_size=7, stride=7).squeeze(0)
        ),
        # Remove the channel dimension if it still exists.
        transforms.Lambda(lambda x: x.squeeze(0) if x.shape[0] == 1 else x),
        # Rescale the pixel values to [0, 2π).
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

num_qubits = 8
dev = qml.device("lightning.gpu", wires=num_qubits)


def variational_classifier(weights, bias, x):
    # The circuit returns a scalar; add the bias term.
    return base_ansatz_circuit(weights, x, 8) + bias


def square_loss(labels, predictions):
    """
    Mean squared error loss.
    `predictions` is a list of scalars; we stack them into a tensor.
    """
    predictions = torch.stack(predictions)
    return torch.mean((labels - predictions) ** 2)


def accuracy(labels, predictions):
    """
    Compute classification accuracy given ground truth labels and predictions.
    Both labels and predictions are assumed to be torch tensors.
    """
    # Use sign thresholding for predictions.
    predicted_labels = torch.sign(predictions)
    correct = (predicted_labels == torch.sign(labels)).float().sum().item()
    return (correct / labels.numel()) * 100


def cost(params, X, Y):
    """
    Compute the cost on a batch.
    X: batch of feature vectors (each of size 8)
    Y: corresponding labels (expected to be -1 or +1)
    """
    # Evaluate the variational classifier on each sample.
    predictions = [
        variational_classifier(params["weights"], params["bias"], x) for x in X
    ]
    return square_loss(Y, predictions)


# For reproducibility.
np.random.seed(0)
torch.manual_seed(0)

# Initialize 16 ansatz parameters and a bias.
initial_weights = 0.01 * np.random.randn(16)
initial_bias = 0.0

params = {
    "weights": torch.tensor(initial_weights, dtype=torch.float, requires_grad=True),
    "bias": torch.tensor(0.0, dtype=torch.float, requires_grad=True),
}

optimizer = optim.Adam([params["weights"], params["bias"]], lr=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # because we want to reduce when the monitored quantity (loss) stops decreasing
    factor=0.5,  # factor by which the learning rate will be reduced (new_lr = lr * factor)
    patience=10,  # number of epochs with no improvement after which learning rate will be reduced
    verbose=True,  # print a message when the learning rate is reduced
    min_lr=1e-5,  # a lower bound on the learning rate
)


num_epochs = 80

for epoch in range(num_epochs):
    # Training phase
    train_loss = 0.0
    num_train_samples = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = cost(params, X_batch, y_batch)
        loss.backward()
        optimizer.step()
        # Accumulate loss weighted by the batch size
        train_loss += loss.item() * X_batch.size(0)
        num_train_samples += X_batch.size(0)
    train_loss /= num_train_samples

    # Validation phase (no gradients needed)
    val_loss = 0.0
    all_val_preds = []  # to store predictions
    all_val_labels = []  # to store corresponding labels
    num_val_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            # Compute predictions for each sample in the batch.
            predictions = [
                variational_classifier(params["weights"], params["bias"], x)
                for x in X_batch
            ]
            # Stack predictions into a tensor.
            preds_tensor = torch.stack(predictions)
            loss = torch.mean((y_batch - preds_tensor) ** 2)
            val_loss += loss.item() * X_batch.size(0)
            num_val_samples += X_batch.size(0)

            # Collect predictions and labels for accuracy computation.
            all_val_preds.append(preds_tensor)
            all_val_labels.append(y_batch)

    # Average loss over the validation set.
    val_loss /= num_val_samples
    scheduler.step(val_loss)

    # Concatenate all batches and compute accuracy.
    all_val_preds = torch.cat(all_val_preds)
    all_val_labels = torch.cat(all_val_labels)
    val_acc = accuracy(all_val_labels, all_val_preds)

    print(
        f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

test_loss = 0.0
num_test_samples = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        loss = cost(params, X_batch, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        num_test_samples += X_batch.size(0)
test_loss /= num_test_samples

print("Test Loss: ", test_loss)
