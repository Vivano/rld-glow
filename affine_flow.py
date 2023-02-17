from utils import *
from torch.distributions.multivariate_normal import MultivariateNormal

class AffineFlow(FlowModule):

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.randn(2))
        self.t = nn.Parameter(torch.randn(2))
    
    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        z = (x - self.t) * torch.exp(-self.s)
        return [z], -torch.sum(self.s)
    
    def decoder(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z * torch.exp(self.s) + self.t
        return [x], torch.sum(self.s)


epochs = 10000
n_samples = 1000

mean_x = torch.tensor([3., 3.])
var_x = 3 * torch.ones(2)
normal_x = MultivariateNormal(mean_x, torch.diag(var_x))

mean_z = torch.zeros(2)
var_z = torch.ones(2)
normal_z = MultivariateNormal(mean_z, torch.diag(var_z))


model = AffineFlow()
# print(model.parameters())
criterion = nn.MSELoss()
optim = torch.optim.SGD(params=model.parameters(), lr=1e-3)

for t in range(epochs):
    
    optim.zero_grad()
    Z = normal_z.sample((n_samples,))
    xhat, logdet = model.decoder(Z)
    xhat = xhat.pop()
    loss = - (normal_x.log_prob(xhat) + logdet).mean()
    # loss = criterion(p_approx, normal_x.log_prob(X))
    loss.backward()
    optim.step()

    if t==0 or (t+1)%1000==0:
        print(f"Itération {t+1} : loss={loss}")
        print(model.s)
        print(model.t)
        print("\n")


X = normal_x.sample((n_samples,))
fig_raw, ax_raw = plt.subplots()
ax_raw.scatter(Z[:,0], Z[:,1], label='Z')
ax_raw.scatter(X[:,0], X[:,1], label='X')
ax_raw.legend(loc='best')
ax_raw.set_title('Distributions before decoding')
plt.savefig('raw_distrib')

x_decoded = xhat.detach()
fig_decod, ax_decod = plt.subplots()
ax_decod.scatter(x_decoded[:,0], x_decoded[:,1], label='decoded')
ax_decod.scatter(X[:,0], X[:,1], label='X')
ax_decod.legend(loc='best')
ax_decod.set_title('Distributions after decoding')
plt.savefig('decoded_distrib')