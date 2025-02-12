import torch.nn as nn
import torch.nn.functional as F
import torch


class ClassicalNN(nn.Module):
    def __init__(self, input_size=32, hidden_size=128, output_size=1):
        super(ClassicalNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(x)  # Add sigmoid activation
        return x
