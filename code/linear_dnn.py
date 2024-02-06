import random
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm 

class LinearDNN(nn.Module):
    def __init__(self, num_input, num_output, hidden_arch, lr):
        self.lr = lr
        super().__init__()   
        layers = []
        node_seq = [num_input] + hidden_arch + [num_output] 
        layers.append(nn.Linear(node_seq[0], node_seq[1]))
        for i in range(1, len(node_seq)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(node_seq[i], node_seq[i+1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def train(self, X, y, epochs, tol=1e-3):
        optimizer = self.configure_optimizers()
        loss_values = []
        for epoch in tqdm(range(epochs)):
            # Forward pass
            y_hat = self.forward(X)
            # Compute loss
            loss = self.loss(y_hat, y)
            # Zero gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            loss_values.append(loss.item())
            # Stop training if loss is close to zero
            if loss.item() < tol:  
                print(f'Training stopped at epoch {epoch+1}, Loss: {loss.item()}')
                break
        return loss_values