import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 
from causal_dnn import CausalDNN
from linear_dnn import LinearDNN

def make_lam(dat, dnn, arch=[5,5], epochs=200):
    # Data elements
    X = dat["X"]
    y = dat["y"]
    z = dat["z"]
    n = y.shape[0]
    # Constants
    d_in = X.shape[1]

    # Extract Parameters
    ab = dnn.forward(X)
    ab = ab.detach().requires_grad_(True)
    alpha = ab[:,0].reshape(n,1)
    beta = ab[:,1].reshape(n,1)

    # Prediction and Loss
    y_hat = alpha + torch.mul(beta,z)
    loss_sum_obs = .5*F.mse_loss(y_hat, y, reduction='sum')   

    # Compute gradient
    score = torch.autograd.grad(loss_sum_obs, ab, create_graph=True, retain_graph=True)[0]

    # Compute and store Hessian
    score_sum_obs = score.sum(dim=0)
    hess_alpha = torch.autograd.grad(score_sum_obs[0], ab, create_graph=True, retain_graph=True)[0]
    hess_beta = torch.autograd.grad(score_sum_obs[1], ab, create_graph=True, retain_graph=True)[0]        
    hess_vec = torch.cat((hess_alpha, hess_beta),dim=1)

    # Compute projection of Hessian
    proj_H = []
    for i in range(hess_vec.shape[1]):
        Hy = hess_vec[:,i].clone().detach().reshape(n,1)
        tempDNN = LinearDNN(num_input = d_in, num_output = 1, hidden_arch = arch, lr = 0.01)
        tempDNN.train(X, Hy, epochs=epochs)
        proj_H.append(tempDNN)
    
    return proj_H

def proc_res(dat, dnn, proj_H, H_func):
    # Data elements
    X = dat["X"]
    y = dat["y"]
    z = dat["z"]
    n = y.shape[0]
    # Constants
    d_in = X.shape[1]

    # Extract Parameters
    ab = dnn.forward(X)
    ab = ab.detach().requires_grad_(True)
    alpha = ab[:,0].reshape(n,1)
    beta = ab[:,1].reshape(n,1)

    # Prediction and Loss
    y_hat = alpha + torch.mul(beta,z)
    loss_sum_obs = .5*F.mse_loss(y_hat, y, reduction='sum')   

    # Compute gradient
    score = torch.autograd.grad(loss_sum_obs, ab, create_graph=True, retain_graph=True)[0]
    score = score.detach().reshape(2,n)

    # Compute lambda projects
    hess_vec = []
    for i in range(len(proj_H)):
        hess_vec.append(proj_H[i].forward(X))
    hess_vec = torch.cat(hess_vec,dim=1)

    # Auto-differentiation of H
    H_val = H_func(dat, ab)
    Hab = torch.autograd.grad(H_val.sum(), ab, create_graph=True, retain_graph=True)[0]   
    Hab = Hab.detach().reshape(n,2)
    inv_V = []
    for i in range(n):
        temp = hess_vec[i,:].clone().detach().reshape(2,2)
        inv_V.append(torch.linalg.pinv(temp).detach())
    
    H_val = H_val.detach()
    auto_if = torch.zeros(n,1)
    for i in range(n):
        auto_if[i] = H_val[i] + torch.matmul(torch.matmul(Hab[i,:],inv_V[i]), score[:,i])
    
    return auto_if
