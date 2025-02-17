import numpy as np
from itertools import combinations
import pennylane as qml
import torch
import torch.nn as nn


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


def create_obs_circuit(locality):
    """
    Returns a QNode that uses the given locality.
    The circuit encodes the input features via a feature map,
    applies an ansatz, and then returns expectation values for each observable.
    """
    dev = qml.device("lightning.qubit", wires=8)

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


def create_ansatz_circuit():
    dev = qml.device("lightning.qubit", wires=8)

    @qml.qnode(dev, interface="torch", batching="vector")
    def circuit(params, features):
        feature_map(features)
        ansatz(params)
        # For simplicity, we measure PauliZ on the first qubit.
        return qml.expval(qml.PauliZ(0))

    return circuit


def create_hybrid_circuit(locality):
    """Circuit with variable locality observables"""
    dev = qml.device("lightning.qubit", wires=8)

    @qml.qnode(dev, interface="torch", batching="vector")
    def circuit(params, features):
        feature_map(features)
        ansatz(params)
        return [qml.expval(o) for o in local_pauli_group(8, locality)]

    return circuit


def create_baseline_circuit():
    dev = qml.device("lightning.qubit", wires=8)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(params, features):
        feature_map(features)  # Keep original encoding
        simple_ansatz(params)
        return qml.expval(qml.PauliZ(0))

    return circuit


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


def feature_map(features):
    for i, feature in enumerate(features):
        qml.RX(feature, wires=i)


def ansatz(params):
    for i in range(8):
        qml.RY(params[i], wires=i)
        qml.RZ(params[i + 8], wires=i)


def simple_ansatz(params):
    """Shallow variational circuit with minimal parameters"""
    num_qubits = 8
    # First rotation layer (RY only)
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)

    # Basic entangling layer
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[num_qubits - 1, 0])  # Close the chain

    # Final rotation layer (single RZ)
    for i in range(num_qubits):
        qml.RZ(params[i + num_qubits], wires=i)


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


def accuracy(predictions, targets):
    # For binary classification, use the sign of the prediction.
    pred_class = torch.sign(predictions)
    # Assume targets are -1 or +1.
    return (pred_class == targets).float().mean().item()
