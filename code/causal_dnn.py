import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 

class CausalDNN(nn.Module):
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
    
    def train(self, X, y, z, epochs, tol=1e-3):
        n = y.shape[0]  
        optimizer = self.configure_optimizers()
        loss_values = []
        for epoch in tqdm(range(epochs)):
            # Forward pass
            ab = self.forward(X)
            alpha = ab[:,0]
            beta = ab[:,1]
            # Reshape variables
            alpha = torch.reshape(alpha,(n,1))
            beta = torch.reshape(beta,(n,1))
            y_hat = alpha + torch.mul(beta,z)
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
        # Store the estimated coefficients
        self.alpha_vec = alpha
        self.beta_vec = beta
        # Reset computational graph
        ab = ab.detach().requires_grad_(True)
        alpha = ab[:,0].reshape(n,1)
        beta = ab[:,1].reshape(n,1)
        y_hat = alpha + torch.mul(beta,z)
        loss_sum_obs = .5*F.mse_loss(y_hat, y, reduction='sum')        
        # Compute and store score
        score = torch.autograd.grad(loss_sum_obs, ab, create_graph=True, retain_graph=True)[0]
        self.score_vec = score
        # Compute and store Hessian
        score_sum_obs = score.sum(dim=0)
        hess_alpha = torch.autograd.grad(score_sum_obs[0], ab, create_graph=True, retain_graph=True)[0]
        hess_beta = torch.autograd.grad(score_sum_obs[1], ab, create_graph=True, retain_graph=True)[0]        
        self.hess_vec = torch.cat((hess_alpha, hess_beta),dim=1)
        return loss_values