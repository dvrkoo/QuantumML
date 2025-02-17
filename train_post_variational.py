import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import combinations
import pennylane as qml
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from data_loader import create_binary_datasets
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def deriv_params(thetas: int, order: int):
    """
    Generate a set of parameter shift vectors for calculating derivatives of a quantum circuit.
    'thetas': number of parameters in the circuit.
    'order': the order of the derivative to calculate.
    """

    def generate_shifts(thetas: int, order: int):
        # Generate all possible combinations of parameters to shift.
        shift_pos = list(combinations(np.arange(thetas), order))
        # Create an array to hold shifts.
        # Shape: (number of combinations, 2^order, thetas)
        params_array = np.zeros((len(shift_pos), 2**order, thetas))
        # Iterate over each combination and each binary pattern.
        for i in range(len(shift_pos)):
            for j in range(2**order):
                # Convert integer j into a binary string of length 'order'.
                for k, l in enumerate(f"{j:0{order}b}"):
                    if int(l) > 0:
                        params_array[i][j][shift_pos[i][k]] += 1
                    else:
                        params_array[i][j][shift_pos[i][k]] -= 1
        # Collapse the first two dimensions.
        params_array = np.reshape(params_array, (-1, thetas))
        return params_array

    # Start with no shift.
    param_list = [np.zeros((1, thetas))]
    # Append shifts for orders 1 to `order`.
    for i in range(1, order + 1):
        param_list.append(generate_shifts(thetas, i))
    # Combine all shift arrays and scale by π/2.
    params_out = np.concatenate(param_list, axis=0)
    params_out *= np.pi / 2
    return params_out


# -----------------------------------------------------------------------------
# Enhanced Data Preprocessing (16 features → 16 qubits)
# -----------------------------------------------------------------------------
def accuracy(predictions, targets):
    # For binary classification, use the sign of the prediction.
    pred_class = torch.sign(predictions)
    # Assume targets are -1 or +1.
    return (pred_class == targets).float().mean().item()


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
# -----------------------------------------------------------------------------
# Quantum Components (16 qubits version)
# -----------------------------------------------------------------------------


def feature_map(features):
    for i in range(8):
        qml.RX(features[i], wires=i)


def ansatz(params):
    for i in range(8):
        qml.RY(params[i], wires=i)
        qml.RZ(params[i + 8], wires=i)


def local_pauli_group(n_qubits, locality):
    """Generate all k-local Pauli-Z observables"""
    obs = []
    for qubits in combinations(range(n_qubits), locality):
        if len(qubits) == 1:
            obs.append(qml.PauliZ(qubits[0]))
        else:
            obs.append(qml.prod(*[qml.PauliZ(q) for q in qubits]))
    return obs


def create_quantum_circuit(locality):
    """Circuit with variable locality observables"""
    dev = qml.device("lightning.qubit", wires=8)

    @qml.qnode(dev, interface="torch", batching="vector")
    def circuit(params, features):
        feature_map(features)
        ansatz(params)
        return [qml.expval(o) for o in local_pauli_group(8, locality)]

    return circuit


class QuantumMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.quantum_features = None  # Store quantum features for inspection
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, quantum_features):
        self.quantum_features = quantum_features.detach()
        self.quantum_features = self.quantum_features
        return self.mlp(quantum_features)


def create_post_variational_system(locality=1, shift_order=1):
    # Quantum components
    n_qubits = 8
    circuit = create_quantum_circuit(locality)
    shift_vectors = torch.tensor(deriv_params(16, shift_order), dtype=torch.float)

    # Classical components
    num_observables = len(local_pauli_group(n_qubits, locality))
    input_dim = num_observables * len(shift_vectors)
    mlp = QuantumMLP(input_dim)
    mlp = mlp

    return circuit, mlp, shift_vectors


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


# Modified training function with integrated validation
def train_post_variational(locality=1, shift_order=1):
    circuit, mlp, shift_vectors = create_post_variational_system(locality, shift_order)
    params = torch.nn.Parameter(0.01 * torch.randn(16))
    optimizer = optim.Adam(
        [{"params": params, "lr": 0.01}, {"params": mlp.parameters(), "lr": 0.001}]
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_params = None

    for epoch in range(100):
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
    print(f"\nFinal Test Metrics: Loss {test_loss:.4f} | Acc {test_acc:.4f}")

    return mlp, best_params


# Train the model
mlp, best_params = train_post_variational(locality=1, shift_order=1)
