import pennylane as qml
import torch
import numpy as np
from itertools import combinations


def feature_map(features):
    # features is assumed to be a 2D array of shape (num_rows, num_qubits)
    print(features.shape)
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


num_qubits = 16
fixed_params = torch.zeros(16)  # Fixed parameters for the ansatz

dev = qml.device("lightning.gpu", wires=num_qubits)


@qml.qnode(dev, interface="torch")
def quantum_feature_extractor_ansatz(features, params):
    # features: a tensor of shape (8, 8)
    feature_map(features)
    ansatz(params.T)
    # Measure expectation values of PauliZ on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]


@qml.qnode(dev, interface="torch")
def quantum_feature_extractor_observable(features, locality=1):
    """
    Uses the feature map and then directly measures multiple observables.
    Here, for demonstration, we measure PauliZ and PauliX on each qubit.
    """
    feature_map(features)
    measurements = []
    for i in range(num_qubits):
        # For each qubit, measure two observables:
        measurements.append(qml.expval(qml.PauliZ(i)))
        measurements.append(qml.expval(qml.PauliX(i)))
    return measurements


def preprocess_image(image, locality=1):
    """
    Prepares an image for quantum feature encoding by flattening and normalizing it.

    Args:
        image (torch.Tensor): Input image tensor (shape: HxW).
        locality (int): Defines the level of locality (currently unused but reserved for future use).

    Returns:
        np.array: Flattened and normalized pixel values.
    """
    # Flatten the image (convert from 2D to 1D)
    flattened = image.flatten()

    # Normalize pixel values between [0, π] (for angle encoding)
    normalized = (
        (flattened - flattened.min()) / (flattened.max() - flattened.min()) * np.pi
    )

    return normalized.numpy()


def quantum_feature_extractor_heuristic(image, heuristic="ansatz", locality=1):
    """
    Extracts quantum features from an image using a heuristic method.

    Args:
        image (torch.Tensor): Input image.
        heuristic (str): The heuristic method to use ("observable" or "ansatz").
        locality (int): Level of locality for feature encoding.

    Returns:
        np.array: Extracted quantum features.
    """
    features = image

    if heuristic == "observable":
        # Compute features using the observable-based heuristic
        return quantum_feature_extractor_observable(features)

    elif heuristic == "ansatz":
        # Compute the number of parameters for the ansatz
        theta_count = features.shape[0]

        # Generate shift values for parameter-shift differentiation
        shifted_params = deriv_params(theta_count, order=1)

        # Apply quantum circuit with shifted parameters
        return quantum_feature_extractor_ansatz(features, shifted_params.T)

    else:
        raise ValueError("Invalid heuristic choice. Choose 'observable' or 'ansatz'.")


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
