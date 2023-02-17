from utils import *

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import functional as F
matplotlib.use("TkAgg")
# import seaborn as sns
from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn import datasets

sns.set_theme()


class ActNorm(FlowModule):
    def __init__(self,input_size):
        super().__init__()
        self.s = nn.Parameter(torch.randn(input_size))
        self.t = nn.Parameter(torch.randn(input_size))
        self.init = False

    def initialize_parameters(self, x):
        with torch.no_grad():
            #self.s.data = torch.std(x, dim=0, keepdim=True)
            self.s.data = torch.log(torch.std(x, dim=0, keepdim=True))/torch.sqrt(torch.tensor([2]))
            self.t.data = torch.mean(x, dim=0, keepdim=True)
            self.init = True

    def decoder(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.init:
            self.initialize_parameters(y)
        return y*torch.exp(self.s) + self.t, torch.sum(self.s)

    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.init:
            self.initialize_parameters(x)
        return (x-self.t)*torch.exp(-self.s), -torch.sum(self.s)




class AffineCoupling(FlowModule):
    def __init__(self,input_size,hidden_size,i):
        super().__init__()
        self.scale_net = MLP(input_size//2, input_size - input_size//2, hidden_size)
        self.shift_net = MLP(input_size//2, input_size - input_size//2, hidden_size)
        self.l = input_size//2
        self.i = i

    def decoder(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.i % 2 == 0:
            y1, y2 = y.split(self.l, dim=-1)
        else:
            y2, y1 = y.split(self.l, dim=-1)
        s = self.scale_net(y1)
        t = self.shift_net(y1)
        out1 = y1
        out2 = y2*torch.exp(s) + t
        if self.i % 2 == 0:
            x = torch.cat([out1, out2], dim=1)
        else:
            x = torch.cat([out2, out1], dim=1)
        log_det = torch.sum(s, dim=1)
        return x, log_det

    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.i % 2 == 0:
            x1, x2 = x.split(self.l, dim=-1)
        else:
            x2, x1 = x.split(self.l, dim=-1)
        s = self.scale_net(x1)
        t = self.shift_net(x1)
        out1 = x1
        out2 = (x2-t)*torch.exp(-s)
        if self.i % 2 == 0:
            y = torch.cat([out1, out2], dim=1)
        else:
            y = torch.cat([out2, out1], dim=1)
        log_det = -torch.sum(s, dim=1)
        return y, log_det


class Invertible1x1Conv(FlowModule):
    """ 
    As introduced in Glow paper.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        # Decompose Q in P (L + Id) (S + U)
        # https://pytorch.org/docs/stable/generated/torch.lu_unpack.html
        P, L, U = torch.lu_unpack(*Q.lu())
        # Not optimizated
        self.P = nn.Parameter(P, requires_grad=False)
        # Lower triangular
        self.L = nn.Parameter(L)
        # Diagonal
        self.S = nn.Parameter(U.diag())
        self.U = nn.Parameter(torch.triu(U, diagonal=1))

    def _assemble_W(self):
        """Computes W from P, L, S and U"""
        # https://pytorch.org/docs/stable/generated/torch.tril.html
        # Excludes the diagonal
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device))
        # https://pytorch.org/docs/stable/generated/torch.triu.html
        # Excludes the diagonal
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def decoder(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        return y @ self._assemble_W(), torch.sum(torch.log(torch.abs(self.S)))

    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return x @ torch.inverse(self._assemble_W()), -torch.sum(torch.log(torch.abs(self.S)))



""" class Flow(FlowModule):
    def __init__(self, input_size, hidden_size, i):
        super().__init__()
        self.actnorm = ActNorm(input_size)
        self.invconv = Invertible1x1Conv(input_size)
        self.coupling = AffineCoupling(input_size, hidden_size, i)

    def decoder(self, y):
        x, logdet = self.coupling.decoder(y)
        x, det1 = self.invconv.decoder(x)
        x, det2 = self.actnorm.decoder(x)
        logdet = logdet + det1 + det2
        return x, logdet

    def encoder(self, x):
        y, logdet = self.actnorm.encoder(x)
        y, det1 = self.invconv.encoder(y)
        y, det2 = self.coupling.encoder(y)
        logdet = logdet + det1 + det2
        return y, logdet

 """



n_samples = 1500
input_size = 2
hidden_size = 100
nflow = 10
mean_x = torch.zeros(input_size)
var_x = 0.2*torch.ones(input_size)
prior = MultivariateNormal(mean_x, torch.diag(var_x))

mix = torch.distributions.Categorical(torch.ones(2,))
comp = torch.distributions.Independent(torch.distributions.Normal(torch.tensor([[-0.5,-0.5],[0.5,0.5]]), torch.ones(2,2)*0.2), 1)
prior = torch.distributions.MixtureSameFamily(mix, comp)
#model = FlowModel(prior, *[Flow(input_size, hidden_size, i) for i in range(nflow)])

convs = [Invertible1x1Conv(input_size) for i in range(nflow)]
norms = [ActNorm(input_size) for _ in range(nflow)]
couplings = [AffineCoupling(input_size, hidden_size, i) for i in range(nflow)]
flows = []
for cv, nm, cp in zip(convs, norms, couplings):
    flows += [nm, cv, cp]
model = FlowModel(prior, *flows) 
optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)
epochs = 5000

for iter in range(epochs):

        data, _ = datasets.make_moons(n_samples=n_samples,  shuffle=True, noise=0.05, random_state=0)
        sample = torch.from_numpy(data).float()

        logprob, _, logdet = model.encoder(sample)
        optim.zero_grad()
        loss = - (logprob + logdet).mean()
        loss.backward()
        optim.step()

        if iter % 100 == 0:
            print(f'Epoch {iter} : loss : {loss}')


model.plot(torch.from_numpy(data).float(),200)
