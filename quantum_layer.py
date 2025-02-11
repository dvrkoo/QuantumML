import pennylane as qml
import torch


def feature_map(features):
    # features is assumed to be a 2D array of shape (num_rows, num_qubits)
    num_rows, num_qubits = features.shape

    # Apply Hadamard to all qubits (to generate superposition)
    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    # For each row, embed the angles using alternating rotation axes
    for i in range(num_rows):
        # For even-indexed rows, embed using X-rotation; for odd rows, use Z-rotation
        rotation = "X" if i % 2 == 0 else "Z"
        # AngleEmbedding can be applied on all qubits at once
        qml.AngleEmbedding(
            features=features[i], wires=range(num_qubits), rotation=rotation
        )


def ansatz(params):
    num_qubits = 2  # Based on our image columns
    # First layer of RY rotations
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)

    # First entangling layer: cyclic CNOTs
    for i in range(num_qubits):
        qml.CNOT(wires=[(i - 1) % num_qubits, i])

    # Second layer of RY rotations
    for i in range(num_qubits):
        qml.RY(params[i + num_qubits], wires=i)

    # Second entangling layer: cyclic CNOTs in reverse order
    for i in range(num_qubits):
        qml.CNOT(
            wires=[(num_qubits - 2 - i) % num_qubits, (num_qubits - i - 1) % num_qubits]
        )


num_qubits = 8
fixed_params = torch.zeros(16)  # Fixed parameters for the ansatz

dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev, interface="torch")
def quantum_feature_extractor(features):
    # features: a tensor of shape (8, 8)
    feature_map(features)
    ansatz(fixed_params)
    # Measure expectation values of PauliZ on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
