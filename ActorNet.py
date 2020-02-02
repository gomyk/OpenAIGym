import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

HIDDEN_512 = 600
HIDDEN_256 = 300
HIDDEN_128 = 150


class Actor(nn.Module):

    def __init__(self, obs_size, outputs, init_w: float = 3e-3, ):
        super(Actor, self).__init__()

        self.linear = nn.Linear(obs_size, HIDDEN_512)
        self.linear2 = nn.Linear(HIDDEN_512, HIDDEN_256)
        self.head = nn.Linear(HIDDEN_256, outputs)

        self.bn512 = nn.BatchNorm1d(HIDDEN_512)
        self.bn256 = nn.BatchNorm1d(HIDDEN_256)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear2.weight.data.uniform_(-1.5e-3, 1.5e-3)
        self.head.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn512(self.linear(state)))
        x = F.relu(self.bn256(self.linear2(x)))
        return self.head(x).tanh()
