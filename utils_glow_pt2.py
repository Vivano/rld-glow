import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')



def squeeze(x):
    """
    x : b x c x h x w -> b x 4c x h/2 x w/2
    """
    b, c, h, w = x.shape
    if h % 2 != 0:
        x = F.pad(x, (0, 0, 1, 0))
        h += 1
    if w % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1))
        w += 1
    # print(h)
    h_med = h // 2
    w_med = w // 2
    # print(h_med, w_med)
    x = x.reshape(b, c, h_med, 2, w_med, 2)
    output = x.permute(0, 1, 3, 5, 2, 4).reshape(b, 4 * c, h_med, w_med)
    return output



def unsqueeze(z):
    b, c, h, w = z.shape
    h *= 2
    w *= 2
    x = z.reshape(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h, w)
    return x




def split(y):
    """
    y (b x c x h x w) -> z (b x c/2 x h x w) , h (b x c/2 x h x w)
    """
    _, c, _, _ = y.shape
    med = c // 2
    z, h = y[:, :med, :, :], y[:, med:, :, :]
    return z, h




class ActNorm(nn.Module):
    def __init__(self, n_channels) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, n_channels, 1, 1))
        self.t = nn.Parameter(torch.randn(1, n_channels, 1, 1))
        self.initialized = False

    def initialize_parameters(self, x):
        with torch.no_grad():
            self.s.data = torch.log(torch.std(x, dim=(0, 2, 3), keepdim=True)) / torch.sqrt(torch.tensor([2]))
            self.t.data = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            self.initialized = True

    def encoder(self, x):
        if not self.initialized:
            self.initialize_parameters(x)
        return (x - self.t) * torch.exp(-self.s), -torch.sum(torch.exp(self.s))

    def decoder(self, z):
        if not self.initialized:
            self.initialize_parameters(z)
        return self.t + z * torch.exp(self.s), torch.sum(torch.exp(self.s))



class InvConv2d(nn.Module):

    def __init__(self, n_channels) -> None:
        super().__init__()
        # print("n channels inv conv 2d : ", n_channels)
        self.n_channels = n_channels
        Q = torch.nn.init.orthogonal_(torch.randn(n_channels, n_channels))
        # Decompose Q in P (L + Id) (S + U)
        P, L, U = torch.lu_unpack(*Q.lu())
        # Not optimized
        self.P = nn.Parameter(P, requires_grad=False)
        # Lower triangular
        self.L = nn.Parameter(L)
        # Diagonal
        self.S = nn.Parameter(U.diag())
        self.U = nn.Parameter(torch.triu(U, diagonal=1))

    def weight(self):
        """Computes W from P, L, S and U"""
        # Excludes the diagonal
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.n_channels, device=self.L.device))
        # Excludes the diagonal
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def decoder(self, z):
        _, _, h, w = z.shape
        weight = self.weight()
        x = F.conv2d(z, weight.unsqueeze(2).unsqueeze(3))
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return x, h * w * log_det

    def encoder(self, x):
        _, _, h, w = x.shape
        weight = self.weight()
        # print(weight.shape)
        z = F.conv2d(x, torch.inverse(weight).unsqueeze(2).unsqueeze(3))
        log_det_ = -torch.sum(torch.log(torch.abs(self.S)))
        return z, h * w * log_det_





class AffineCoupling(nn.Module):

    def __init__(self, n_channels, hidden_channels, i):
        super().__init__()
        self.pad = False
        if n_channels % 2 != 0:
            n_channels += 1
            self.pad = True
        self.scale_net = nn.Sequential(
            nn.Conv2d(n_channels // 2, hidden_channels, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_channels // 2, 3, padding="same"))
        self.shift_net = nn.Sequential(
            nn.Conv2d(n_channels // 2, hidden_channels, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_channels // 2, 3, padding="same"))
        self.ite = i

    def decoder(self, z):
        b, c, h, w = z.shape
        if c % 2 != 0:
            z = F.pad(input=z, pad=(0, 0, 0, 0, 1, 0), mode='constant', value=0.)
            c += 1
        # print(c)
        med_c = c // 2
        z1, z2 = z[:, :med_c, :, :], z[:, med_c:, :, :]
        y = torch.zeros(b, c, h, w)
        if self.ite % 2 == 0:
            s = self.scale_net(z1)
            t = self.shift_net(z1)
            y[:, :med_c, :, :] = z1
            y[:, med_c:, :, :] = z2 * torch.exp(s) + t
        else:
            s = self.scale_net(z2)
            t = self.shift_net(z2)
            y[:, :med_c, :, :] = z2
            y[:, med_c:, :, :] = z1 * torch.exp(s) + t
        log_det = torch.sum(torch.exp(s))
        if self.pad:
            return y[:, :-1, :, :], log_det
        else:
            return y, log_det

    def encoder(self, x):
        b, c, h, w = x.shape
        # print(x.shape)
        if c % 2 != 0:
            x = F.pad(input=x, pad=(0, 0, 0, 0, 1, 0), mode='constant', value=0.)
            c += 1
        # (x.shape)
        med_c = c // 2
        x1, x2 = x[:, :med_c, :, :], x[:, med_c:, :, :]
        # print(x1.shape, x2.shape)
        y = torch.zeros(b, c, h, w)
        # print(y.shape)
        if self.ite % 2 == 0:
            s = self.scale_net(x1)
            t = self.shift_net(x1)
            # print(x1.shape, y.shape)
            y[:, :med_c, :, :] = x1
            y[:, med_c:, :, :] = (x2 - t) / torch.exp(s)
        else:
            s = self.scale_net(x2)
            t = self.shift_net(x2)
            y[:, :med_c, :, :] = x2
            y[:, med_c:, :, :] = (x1 - t) / torch.exp(s)
        log_det_ = -torch.sum(torch.exp(s))
        if self.pad:
            return y[:, :-1, :, :], log_det_
        else:
            return y, log_det_
