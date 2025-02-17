import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from data_loader import create_binary_datasets
import torch.nn.functional as F
from models.quantum_layer import create_ansatz_circuit, deriv_params, accuracy


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

train_dataset, val_dataset, test_dataset = create_binary_datasets(
    full_dataset, class_1=4, class_2=6, train_size=200, val_size=50, test_size=50
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def expanded_variational_classifier(circuit, params, bias, features, shift_vectors):
    """
    Evaluate the circuit for a series of shifted parameter settings (given by shift_vectors)
    and combine the outputs (here, by averaging) to produce a prediction.
    """
    outputs = []
    for shift in shift_vectors:
        shifted_params = params + shift
        outputs.append(circuit(shifted_params, features))
    outputs = torch.stack(outputs)
    output = torch.mean(outputs)
    return output + bias


def rmse_loss(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


# We will run training for different shift orders.
for shift_order in [1, 2]:
    print(f"\nTraining with shift order {shift_order}")

    # Generate the shift vectors for the current order.
    shift_vectors = torch.tensor(
        deriv_params(thetas=16, order=shift_order), dtype=torch.float
    )

    # Create the quantum circuit.
    circuit = create_ansatz_circuit()

    # Initialize the variational parameters and bias.
    params = torch.nn.Parameter(0.01 * torch.randn(16))
    bias = torch.nn.Parameter(torch.tensor(0.0))

    optimizer = optim.Adam([params, bias], lr=0.01)
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0

        for features_batch, targets_batch in tqdm(
            train_loader, desc=f"Epochs for shift order {shift_order}"
        ):
            optimizer.zero_grad()
            batch_preds = []
            for features, target in zip(features_batch, targets_batch):
                # Use the expanded variational classifier with the current shift vectors.
                pred = expanded_variational_classifier(
                    circuit, params, bias, features.float(), shift_vectors
                )
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
                    pred = expanded_variational_classifier(
                        circuit, params, bias, features.float(), shift_vectors
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

        print(
            f"Shift Order {shift_order} | Epoch {epoch+1:3d} | Train RMSE: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
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
                pred = expanded_variational_classifier(
                    circuit, params, bias, features.float(), shift_vectors
                )
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
    print(
        f"Final Test RMSE for shift order {shift_order}: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n"
    )
