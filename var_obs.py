import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from data_loader import create_binary_datasets
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_sample(features, circuit, params, bias):
    # Evaluate the quantum circuit for one sample
    return variational_classifier(circuit, params, bias, features.float())


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


def feature_map(features):
    # Example: encode each feature into an RX rotation.
    for i, feature in enumerate(features):
        qml.RX(feature, wires=i)


def ansatz(weights):
    # Example ansatz: for a system with 4 qubits, use 2 rotations per qubit.
    num_qubits = 8
    for i in range(num_qubits):
        qml.RY(weights[i], wires=i)
        qml.RZ(weights[i + num_qubits], wires=i)


# =============================================================================
# Define a quantum circuit that returns an expectation value.
# We build a circuit with a given locality by (for example) choosing a different observable.
# Here we define a dummy "local_pauli_group" for demonstration.
# =============================================================================


def local_pauli_group(n_qubits, locality):
    """
    For demonstration, this dummy function returns a list of measurement observables.
    In your implementation, it should generate a list of Pauli strings based on the locality.
    """
    # For simplicity, we return one observable if locality==1,
    # two observables if locality==2, etc.
    # Replace this with your actual measurement generation.
    observables = []
    for i in range(locality):
        # For instance, measure PauliZ on qubit i (make sure i < n_qubits)
        observables.append(qml.PauliZ(i))
    return observables


def create_quantum_circuit(locality):
    """
    Returns a QNode that uses the given locality.
    The circuit encodes the input features via a feature map,
    applies an ansatz, and then returns expectation values for each observable.
    """
    # For this example, we use 4 qubits.
    dev = qml.device("default.qubit", wires=8)

    @qml.qnode(dev, interface="torch", batching="vector")
    def circuit(params, features):
        # Encode the classical features.
        feature_map(features)
        # Apply the variational ansatz.
        ansatz(params)
        # Generate observables based on the locality.
        observables = local_pauli_group(8, locality)
        # Measure expectation values for each observable.
        # If there is only one observable, return a scalar; otherwise return a list.
        return [qml.expval(obs) for obs in observables]

    return circuit


# =============================================================================
# Define the variational classifier and RMSE loss.
# =============================================================================


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
    # (Alternatively, you could learn a weight for each observable.)
    if isinstance(outputs, (list, tuple)):
        output = torch.mean(torch.stack(outputs))
    else:
        output = outputs
    return output + bias


def rmse_loss(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def accuracy(predictions, targets):
    # For binary classification, use the sign of the prediction.
    pred_class = torch.sign(predictions)
    # targets are assumed to be -1 or +1.
    return (pred_class == targets).float().mean().item()


for locality in range(1, 4):
    print(f"\nTraining with {locality}-local observables:")

    # Create the quantum circuit for the given locality.
    circuit = create_quantum_circuit(locality)

    # Initialize variational parameters:
    # For 4 qubits and 2 rotations per qubit → 8 parameters.
    params = torch.nn.Parameter(0.01 * torch.randn(16))
    bias = torch.nn.Parameter(torch.tensor(0.0))

    optimizer = optim.Adam([params, bias], lr=0.05)
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
