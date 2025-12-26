import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.action_size = action_size

        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x
