from src.misc.constraint_utils import softplus, invsoftplus

import numpy as np
import torch
from torch import nn
from torch.nn import init

from torch.distributions import Normal

prior_weights = Normal(0.0, 1.0)


def sample_normal(shape, seed=None):
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    return torch.tensor(rng.normal(size=shape).astype(np.float32))


class RBF(torch.nn.Module):
    """
    Implements squared exponential kernel with kernel computation and weights and frequency sampling for Fourier features
    """

    def __init__(self, D_in, D_out=None, dimwise=False):
        """
        @param D_in: Number of input dimensions
        @param D_out: Number of output dimensions
        @param dimwise: If True, different kernel parameters are given to output dimensions
        """
        super(RBF, self).__init__()
        self.D_in = D_in
        self.D_out = D_in if D_out is None else D_out
        self.dimwise = dimwise
        lengthscales_shape = (self.D_out, self.D_in) if dimwise else (self.D_in,)
        variance_shape = (self.D_out,) if dimwise else (1,)
        self.unconstrained_lengthscales = nn.Parameter(torch.ones(size=lengthscales_shape),
                                                       requires_grad=True)
        self.unconstrained_variance = nn.Parameter(torch.ones(size=variance_shape),
                                                   requires_grad=True)
        self._initialize()

    def _initialize(self):
        init.constant_(self.unconstrained_lengthscales, invsoftplus(torch.tensor(1.3)).item())
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(0.5)).item())

    @property
    def lengthscales(self):
        return softplus(self.unconstrained_lengthscales)

    @property
    def variance(self):
        return softplus(self.unconstrained_variance)

    def square_dist_dimwise(self, X, X2=None):
        """
        Computes squared euclidean distance (scaled) for dimwise kernel setting
        @param X: Input 1 (N,D_in)
        @param X2: Input 2 (M,D_in)
        @return: Tensor (D_out, N,M)
        """
        X = X.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,N,D_in)
        Xs = torch.sum(torch.pow(X, 2), dim=2)  # (D_out,N)
        if X2 is None:
            return -2 * torch.einsum('dnk, dmk -> dnm', X, X) + \
                   Xs.unsqueeze(-1) + Xs.unsqueeze(1)  # (D_out,N,N)
        else:
            X2 = X2.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,M,D_in)
            X2s = torch.sum(torch.pow(X2, 2), dim=2)  # (D_out,N)
            return -2 * torch.einsum('dnk, dmk -> dnm', X, X2) + Xs.unsqueeze(-1) + X2s.unsqueeze(1)  # (D_out,N,M)

    def square_dist(self, X, X2=None):
        """
        Computes squared euclidean distance (scaled) for non dimwise kernel setting
        @param X: Input 1 (N,D_in)
        @param X2: Input 2 (M,D_in)
        @return: Tensor (N,M)
        """
        X = X / self.lengthscales  # (N,D_in)
        Xs = torch.sum(torch.pow(X, 2), dim=1)  # (N,)
        if X2 is None:
            return -2 * torch.matmul(X, X.t()) + \
                   torch.reshape(Xs, (-1, 1)) + torch.reshape(Xs, (1, -1))  # (N,1)
        else:
            X2 = X2 / self.lengthscales  # (M,D_in)
            X2s = torch.sum(torch.pow(X2, 2), dim=1)  # (M,)
            return -2 * torch.matmul(X, X2.t()) + torch.reshape(Xs, (-1, 1)) + torch.reshape(X2s, (1, -1))  # (N,M)

    def K(self, X, X2=None):
        """
        Computes K(X, X_2)
        @param X: Input 1 (N,D_in)
        @param X2:  Input 2 (M,D_in)
        @return: Tensor (D,N,M) if dimwise else (N,M)
        """
        if self.dimwise:
            sq_dist = torch.exp(- 0.5 * self.square_dist_dimwise(X, X2))  # (D_out,N,M)
            return self.variance[:, None, None] * sq_dist  # (D_out,N,M)
        else:
            sq_dist = torch.exp(-0.5 * self.square_dist(X, X2))  # (N,M)
            return self.variance * sq_dist  # (N,M)

    def sample_freq(self, S, seed=None):
        """
        Computes random samples from the spectral density for Squared exponential kernel
        @param S: Number of features
        @param seed: random seed
        @return: Tensor a random sample from standard Normal (D_in, S, D_out) if dimwise else (D_in, S)
        """
        omega_shape = (self.D_in, S, self.D_out) if self.dimwise else (self.D_in, S)
        omega = sample_normal(omega_shape, seed)  # (D_in, S, D_out) or (D_in, S)
        lengthscales = self.lengthscales.T.unsqueeze(1) if self.dimwise else self.lengthscales.unsqueeze(
            1)  # (D_in,1,D_out) or (D_in,1)
        return omega / lengthscales  # (D_in, S, D_out) or (D_in, S)


class RBFACD(torch.nn.Module):

    def __init__(self, D_in, D_out=None, dimwise=False):
        """
        @param D_in: Number of input dimensions
        @param D_out: Number of output dimensions
        @param dimwise: If True, different kernel parameters are given to output dimensions
        """
        super(RBFACD, self).__init__()
        self.D_in = D_in
        self.D_out = D_in if D_out is None else D_out
        self.dimwise = dimwise
        L_shape  = (D_out, D_in*(D_in+1)//2) if dimwise else (D_in*(D_in+1)//2,) 
        variance_shape = (self.D_out,) if dimwise else (1,)
        self.unconstrained_L = nn.Parameter(torch.ones(size=L_shape), requires_grad=True)
        self.unconstrained_variance = nn.Parameter(torch.ones(size=variance_shape), requires_grad=True)
        self._initialize()
    
    def _initialize(self):
        init.constant_(self.unconstrained_L, invsoftplus(torch.tensor(1.3)).item())
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(0.5)).item())

    @property
    def variance(self):
        return softplus(self.unconstrained_variance)
    
    @property
    def L(self):
        return self.unconstrained_L

    @property
    def precision(self):
        Lm = self._fill_triangular()
        Lambda = torch.matmul(Lm, Lm.transpose(1,2)) if self.dimwise else torch.matmul(Lm, Lm.t())
        return Lambda
    
    def _fill_triangular(self):
        lower_indices = torch.tril_indices(self.D_in, self.D_in) # (2, lsize)
        l_matrix = torch.zeros(self.D_in, self.D_in, device=self.L.device, dtype=self.L.dtype) # (input_dim, input_dim)
        l_matrix[lower_indices.tolist()] = self.L
        return l_matrix
    
    def malhanobis_dist(self, X, X2):
        if X2 is None:
            X2 = X
        precision = self.precision
        XP = torch.matmul(torch.unsqueeze(X, 1), precision)
        X2P = torch.matmul(torch.unsqueeze(X2, 1), precision)
        X11 = torch.squeeze(torch.matmul(XP, torch.unsqueeze(X, -1)), -1)
        X22 = torch.squeeze(torch.matmul(X2P, torch.unsqueeze(X2, -1)), -1).t()
        X12 = torch.matmul(torch.matmul(X, precision), X2.t())

        dist = X11 - 2*X12 + X22

        return dist
    
    def K(self, X, X2=None, presliced=False):
        res = self.variance * torch.exp(-0.5 * self.malhanobis_dist(X, X2))
        return res

    def sample_freq(self, S, seed=None):
        """
        Computes random samples from the spectral density for Squared exponential kernel
        @param S: Number of features
        @param seed: random seed
        @return: Tensor a random sample from standard Normal (D_in, S, D_out) if dimwise else (D_in, S)
        """
        omega_shape = (self.D_in, S, self.D_out) if self.dimwise else (self.D_in, S)
        omega = sample_normal(omega_shape, seed)  # (D_in, S, D_out) or (D_in, S)
        Lm = self._fill_triangular() # (D_out, D_in, D_in) or (D_in, D_in)
        omega_scaled = torch.matmul(Lm, omega.transpose(1,2)).view(self.D_in, S, self.D_out) if self.dimwise else torch.matmul(Lm, omega) # (D_in, S, D_out) or (D_in, S)
        return omega_scaled