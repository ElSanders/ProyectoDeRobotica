import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch.optim as optim

class DQN(nn.Module):

    def __init__(self,input_dims):
        super(DQN, self).__init__()
        self.network = torch.nn.Sequential(
        torch.nn.Linear(*input_dims,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,3)
        )
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.optimizer = optim.Adam(self.parameters(),lr=1e-2)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        action = self.network(state)
        return action
        
    def compute_loss(self, loss):
        loss.to(self.device)
        loss.backward()
        self.optimizer.step()