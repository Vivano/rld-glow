from utils_glow_pt2 import *



class Flow(nn.Module):

    def __init__(self, n_channels, hidden_channels, i):
        super().__init__()
        print(n_channels)
        self.act_norm = ActNorm(n_channels)
        self.inv_conv = InvConv2d(n_channels)
        self.affine = AffineCoupling(n_channels, hidden_channels, i)

    def decoder(self, z):
        z, tmp1 = self.act_norm.decoder(z)
        z, tmp2 = self.inv_conv.decoder(z)
        z, tmp3 = self.affine.decoder(z)
        log_det = tmp1 + tmp2 + tmp3
        return z, log_det

    def encoder(self, x):
        x, tmp1 = self.act_norm.encoder(x)
        x, tmp2 = self.inv_conv.encoder(x)
        x, tmp3 = self.affine.encoder(x)
        log_det_ = tmp1 + tmp2 + tmp3
        return x, log_det_




class FlowBlock(nn.Module):

    def __init__(self, n_flow, n_channels, hidden_channels):
        super().__init__()
        self.n_flow = n_flow
        self.flows = nn.ModuleList()
        for i in range(self.n_flow):
            self.flows.append(Flow(n_channels, hidden_channels, i))

    def decoder(self, z):
        log_det = torch.tensor(0.)
        for flow in self.flows:
            z, tmp = flow.decoder(z)
            log_det += tmp
        return z, log_det

    def encoder(self, x):
        log_det_ = torch.tensor(0.)
        for flow in self.flows:
            x, tmp = flow.encoder(x)
            log_det_ += tmp
        return x, log_det_




class GLOW(nn.Module):

    def __init__(self, prior, n_block, n_flow, n_channels, hidden_channels):
        super().__init__()
        self.prior = prior
        self.n_blocks = n_block
        self.glows = nn.ModuleList()
        for i in range(n_block):
            if i == 0:
                self.glows.append(FlowBlock(n_flow, n_channels, hidden_channels))
            else:
                n_channels *= 2
                self.glows.append(FlowBlock(n_flow, n_channels, hidden_channels))
            # print(n_channels)

    def decoder(self, z):
        pass

    def encoder(self, x):
        z = x
        z_seq = []
        log_det_ = torch.tensor(0.)
        counter = 1
        for glow in self.glows:
            if counter < self.n_blocks:
                z, tmp = glow.encoder(squeeze(z))
                z_new, z = split(z)
                z_seq.append(z_new)
                log_det_ += tmp
            else:
                z_new, tmp = glow.encoder(squeeze(z))
                z_seq.append(z_new)
                log_det_ += tmp
            counter += 1
        log_prob = self.prior.log_prob(z_seq[-1])
        return log_prob, z, log_det_
