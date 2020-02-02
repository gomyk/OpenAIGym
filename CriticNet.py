import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

HIDDEN_512 = 600
HIDDEN_256 = 300
HIDDEN_128 = 150


class Critic(nn.Module):

    def __init__(self, obs_size, action_size, init_w: float = 3e-3, ):
        super(Critic, self).__init__()

        # self.seed = torch.manual_seed(0)

        self.linear = nn.Linear(obs_size, HIDDEN_512)
        self.linear2 = nn.Linear(HIDDEN_512, HIDDEN_256)
        self.linear3 = nn.Linear(action_size, HIDDEN_256)
        self.head = nn.Linear(HIDDEN_256, 1)

        self.bn512 = nn.BatchNorm1d(HIDDEN_512)
        self.bn256 = nn.BatchNorm1d(HIDDEN_256)
        self.bn128 = nn.BatchNorm1d(HIDDEN_128)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear3.weight.data.uniform_(-1.5e-3, 1.5e-3)
        self.head.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # x = torch.cat((state, action), dim=-1)
        x = F.relu(self.bn512(self.linear(state)))
        x = self.linear2(x)

        y = self.linear3(action)

        x = F.relu(torch.add(x, y))
        return self.head(x)