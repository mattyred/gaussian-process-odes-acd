import torch
import torch.nn.functional as F
import numpy as np


def softplus(x):
    lower = 1e-12
    return F.softplus(x) + lower


def invsoftplus(x):
    lower = 1e-12
    xs = torch.max(x - lower, torch.tensor(torch.finfo(x.dtype).eps).to(x))
    return xs + torch.log(-torch.expm1(-xs))

def init_lower_triangular_uniform(D_in, eps=1e-7):
    full_L = np.random.uniform(-1,1,(D_in,D_in))
    P = full_L @ np.transpose(full_L)
    lower_L = np.linalg.cholesky(P + eps)  
    return torch.tensor(lower_L[np.tril_indices(D_in)],  dtype=torch.float32)