"""STGCN (Yu, Yin & Zhu, IJCAI 2018) in plain PyTorch: ST-Conv blocks = gated temporal conv -> Chebyshev graph conv
-> gated temporal conv, followed by an output layer. Input [B, C, T=12, N=307] -> output [B, N] (flow at t+3)."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def distance_graph(csv_path, n_nodes):
    """Weighted adjacency from road distances: w_ij = exp(-(d_ij / sigma)^2), sigma = std of edge distances
    (Gaussian kernel as in the STGCN paper), symmetric, no self loops. Only the listed road edges get a weight."""
    d = pd.read_csv(csv_path)
    sigma = d["cost"].std()
    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    w = np.exp(-(d["cost"].to_numpy() / sigma) ** 2)
    W[d["from"], d["to"]] = w
    W[d["to"], d["from"]] = w
    return W, float(sigma)


def cheb_polynomials(W, K):
    """T_0..T_{K-1} of the scaled normalised Laplacian L~ = 2L/lambda_max - I, L = I - D^-1/2 W D^-1/2."""
    deg = W.sum(1)
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    L = np.eye(len(W)) - d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    lam = np.linalg.eigvalsh(L).max()
    Lt = 2 * L / lam - np.eye(len(W))
    polys = [np.eye(len(W)), Lt]
    for _ in range(2, K):
        polys.append(2 * Lt @ polys[-1] - polys[-2])
    return torch.tensor(np.stack(polys[:K]), dtype=torch.float32), float(lam)


class TemporalGLU(nn.Module):
    """Gated 1-D conv along time with a residual: (P + x) * sigmoid(Q). Shortens T by kt-1."""
    def __init__(self, c_in, c_out, kt):
        super().__init__()
        self.kt, self.c_out = kt, c_out
        self.conv = nn.Conv2d(c_in, 2 * c_out, (kt, 1))
        self.align = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                                   # [B, C, T, N]
        res = self.align(x)[:, :, self.kt - 1:, :]
        p, q = self.conv(x).chunk(2, dim=1)
        return (p + res) * torch.sigmoid(q)


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, cheb):
        super().__init__()
        self.register_buffer("cheb", cheb)                  # [K, N, N]
        self.weight = nn.Parameter(torch.empty(cheb.shape[0], c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.align = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):                                   # [B, C, T, N]
        h = x.permute(0, 2, 3, 1)                           # [B, T, N, C]
        out = sum(torch.einsum("nm,btmc,cd->btnd", self.cheb[k], h, self.weight[k]) for k in range(len(self.cheb)))
        out = (out + self.bias).permute(0, 3, 1, 2)
        return torch.relu(out + self.align(x))


class STBlock(nn.Module):
    def __init__(self, c_in, channels, cheb, n_nodes, kt, dropout):
        super().__init__()
        c1, c2, c3 = channels
        self.t1 = TemporalGLU(c_in, c1, kt)
        self.g = ChebGraphConv(c1, c2, cheb)
        self.t2 = TemporalGLU(c2, c3, kt)
        self.norm = nn.LayerNorm([n_nodes, c3])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.t2(self.g(self.t1(x)))
        return self.drop(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class STGCN(nn.Module):
    def __init__(self, cheb, n_nodes, c_in=2, channels=(64, 16, 64), kt=3, seq_len=12, dropout=0.0):
        super().__init__()
        self.b1 = STBlock(c_in, channels, cheb, n_nodes, kt, dropout)
        self.b2 = STBlock(channels[2], channels, cheb, n_nodes, kt, dropout)
        t_left = seq_len - 4 * (kt - 1)
        c = channels[2]
        self.out = nn.Sequential(nn.Conv2d(c, c, (t_left, 1)), nn.ReLU(), nn.Conv2d(c, 1, 1))

    def forward(self, x):                                   # [B, C, T, N] -> [B, N]
        return self.out(self.b2(self.b1(x)))[:, 0, 0, :]
