from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from data_loader import create_binary_datasets
import pennylane as qml
from observable import local_pauli_group
from ansatz import feature_map
import torch
from classical_nn import ClassicalNN
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda")

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
# Get the full training and testing data as tensors.
X_train_tensor, y_train_tensor = next(iter(train_loader))
X_val_tensor, y_val_tensor = next(iter(val_loader))
X_test_tensor, y_test_tensor = next(iter(test_loader))


def process_batch(data_batch, locality):
    processed_batch = []
    for features in data_batch:
        # Ensure features are in the correct format
        features = np.array(features)
        # Execute the quantum circuit
        result = circuit(features, locality)
        processed_batch.append(result)
    return torch.tensor(processed_batch, dtype=torch.float32)


def vcircuit_loader(dataloader):
    new_train_features_list = []
    new_train_labels_list = []
    for images, labels in train_loader:
        # Assume each 'images' batch has shape (batch_size, H, W) where H x W is 4x4 (or as needed).
        # vcircuit processes a batch of images and returns a tensor of shape (batch_size, n_measurements).
        new_features = vcircuit(images)
        new_train_features_list.append(new_features)
        new_train_labels_list.append(labels)
    new_train_features = torch.cat(new_train_features_list, dim=0)
    new_train_labels = torch.cat(new_train_labels_list, dim=0)
    return new_train_features, new_train_labels


train_accuracies_O = []
test_accuracies_O = []

for locality in range(1, 4):
    print(str(locality) + "-local: ")

    # Define a quantum device with 8 qubits using the default simulator.
    dev = qml.device("lightning.gpu", wires=16)

    # Define a quantum node (qnode) with the quantum circuit that will be executed on the device.
    @qml.qnode(dev)
    def circuit(features, locality):
        # Generate all possible Pauli strings for the given locality.
        measurements = local_pauli_group(16, locality)

        # Apply the feature map to encode classical data into quantum states.
        feature_map(features)

        # Measure the expectation values of the generated Pauli operators.
        return [
            qml.expval(qml.pauli.string_to_pauli_word(measurement))
            for measurement in measurements
        ]

    # Vectorize the quantum circuit function to apply it to multiple data points in parallel.
    vcircuit = torch.vmap(circuit)
    if locality == 1:
        input_dim = 49
    elif locality == 2:
        input_dim = 1129
    elif locality == 3:
        input_dim = 16249
    model = ClassicalNN(input_dim, 64, 1)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    print("Training the classifier...")

    num_epochs = 100  # adjust as needed
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for features, labels in tqdm(train_loader):
            quantum_features = process_batch(features, locality)
            labels = ((labels + 1) / 2).float()
            labels = labels.float().to(device)
            quantum_features = quantum_features.to(device)
            optimizer.zero_grad()
            outputs = model(quantum_features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            # Compute accuracy on training batch.
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        train_acc = correct / total

        # Evaluate on test data.
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                quantum_features = process_batch(features, locality)
                quantum_features = quantum_features.to(device)
                labels = ((labels + 1) / 2).float()
                labels = labels.float().to(device)
                outputs = model(quantum_features).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= total
        test_acc = correct / total

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {val_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

    # Store final accuracies.
    train_accuracies_O.append(train_acc)
    test_accuracies_O.append(test_acc)
    print(f"Final Training accuracy: {train_acc:.4f}")
    print(f"Final Testing accuracy: {test_acc:.4f}\n")
