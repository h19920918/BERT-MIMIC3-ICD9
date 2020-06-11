import torch
from torch import nn

class ReplayPolicyNet(nn.Module):
    def __init__(self, args):
        super(ReplayPolicyNet, self).__init__()
        self.linear1 = nn.Linear(args.hidden_size, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
