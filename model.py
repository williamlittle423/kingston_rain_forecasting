# model.py

import torch
import torch.nn as nn

class PrecipitationClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=512, num_layers=3, dropout_rate=0.5):
        """
        Parameters:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in each hidden layer
            num_layers (int): Number of hidden layers
            dropout_rate (float): Dropout probability
        """
        super(PrecipitationClassifier, self).__init__()
        layers = []
        current_input = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(current_input, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_input = hidden_size
        layers.append(nn.Linear(hidden_size, 1))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the network
        """
        return self.network(x)
