from torch.utils.data import Dataset, Subset


class BinaryFashionMNIST(Dataset):
    def __init__(self, dataset, class_1=4, class_2=6):
        """
        Keep only the two classes we want.
        Here we convert the labels to binary:
            - Use -1 for class_1 (e.g. 'coat' if class_1==4)
            - Use +1 for class_2 (e.g. 'shirt' if class_2==6)
        """
        self.indices = [
            i for i, (_, label) in enumerate(dataset) if label in [class_1, class_2]
        ]
        self.dataset = dataset
        self.class_1 = class_1
        self.class_2 = class_2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        # Convert label to binary: -1 for class_1, +1 for class_2.
        binary_label = (
            -1 if label == self.class_1 else 1
        )  # TODO: fix labels for post_variational_nn
        return image, binary_label


def create_binary_datasets(
    full_dataset, class_1=4, class_2=6, train_size=200, val_size=50, test_size=50
):
    """
    Splits the full dataset (which should already be binary-filtered) into
    training, validation, and test subsets.
    - train: first (train_size * 2) samples (from both classes)
    - val: next (val_size * 2) samples
    - test: next (test_size * 2) samples
    """
    binary_dataset = BinaryFashionMNIST(full_dataset, class_1, class_2)
    train_total = train_size * 2
    val_total = val_size * 2
    test_total = test_size * 2

    train_dataset = Subset(binary_dataset, range(0, train_total))
    val_dataset = Subset(binary_dataset, range(train_total, train_total + val_total))
    test_dataset = Subset(
        binary_dataset,
        range(train_total + val_total, train_total + val_total + test_total),
    )
    return train_dataset, val_dataset, test_dataset
