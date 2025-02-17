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
from itertools import combinations


def deriv_params(thetas: int, order: int):
    """
    This function generates parameter shift values for calculating derivatives
    of a quantum circuit.
    'thetas' is the number of parameters in the circuit.
    'order' determines the order of the derivative to calculate (1st order, 2nd order, etc.).
    """

    def generate_shifts(thetas: int, order: int):
        # Generate all possible combinations of parameters to shift for the given order.
        shift_pos = list(combinations(np.arange(thetas), order))

        # Initialize a 3D array to hold the shift values.
        # Shape: (number of combinations, 2^order, thetas)
        params = np.zeros((len(shift_pos), 2**order, thetas))

        # Iterate over each combination of parameter shifts.
        for i in range(len(shift_pos)):
            # Iterate over each possible binary shift pattern for the given order.
            for j in range(2**order):
                # Convert the index j to a binary string of length 'order'.
                for k, l in enumerate(f"{j:0{order}b}"):
                    # For each bit in the binary string:
                    if int(l) > 0:
                        # If the bit is 1, increment the corresponding parameter.
                        params[i][j][shift_pos[i][k]] += 1
                    else:
                        # If the bit is 0, decrement the corresponding parameter.
                        params[i][j][shift_pos[i][k]] -= 1

        # Reshape the parameters array to collapse the first two dimensions.
        params = np.reshape(params, (-1, thetas))
        return params

    # Start with a list containing a zero-shift array for all parameters.
    param_list = [np.zeros((1, thetas))]

    # Append the generated shift values for each order from 1 to the given order.
    for i in range(1, order + 1):
        param_list.append(generate_shifts(thetas, i))

    # Concatenate all the shift arrays along the first axis to create the final parameter array.
    params = np.concatenate(param_list, axis=0)

    # Scale the shift values by π/2.
    params *= np.pi / 2

    return params


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

    return qml.expval(qml.PauliZ(0))


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


epochs = 10

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    # Forward pass
    outputs = variational_classifier(circuit, model.parameters(), X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    train_preds = torch.sigmoid(outputs) > 0.5
    train_acc = accuracy_score(y_train_tensor.numpy(), train_preds.numpy())

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = variational_classifier(circuit, model.parameters(), X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_preds = torch.sigmoid(val_outputs) > 0.5
        val_acc = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())

    print(
        f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Train Acc: {train_acc}, Val Loss: {val_loss.item()}, Val Acc: {val_acc}"
    )

# Testing phase
model.eval()
with torch.no_grad():
    test_outputs = variational_classifier(circuit, model.parameters(), X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    test_preds = torch.sigmoid(test_outputs) > 0.5
    test_acc = accuracy_score(y_test_tensor.numpy(), test_preds.numpy())

print(f"Test Loss: {test_loss.item()}, Test Accuracy: {test_acc}")
