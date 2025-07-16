import torch
import torch.nn as nn

class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(TransitionModel, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        
        # Build network with configurable hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, state_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)