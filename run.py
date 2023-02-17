from affine_flow import *
from glow_pt1 import *



########################### AFFINE FLOW ######################################

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

    if t == 0 or (t + 1) % 1000 == 0:
        print(f"Itération {t + 1} : loss={loss}")
        print(model.s)
        print(model.t)
        print("\n")

X = normal_x.sample((n_samples,))
fig_raw, ax_raw = plt.subplots()
ax_raw.scatter(Z[:, 0], Z[:, 1], label='Z')
ax_raw.scatter(X[:, 0], X[:, 1], label='X')
ax_raw.legend(loc='best')
ax_raw.set_title('Distributions before decoding')
plt.savefig('raw_distrib')

x_decoded = xhat.detach()
fig_decod, ax_decod = plt.subplots()
ax_decod.scatter(x_decoded[:, 0], x_decoded[:, 1], label='decoded')
ax_decod.scatter(X[:, 0], X[:, 1], label='X')
ax_decod.legend(loc='best')
ax_decod.set_title('Distributions after decoding')
plt.savefig('decoded_distrib')



################################## GLOW PART I ########################################


n_samples = 1500
input_size = 2
hidden_size = 100
nflow = 10
mean_x = torch.zeros(input_size)
var_x = 0.2*torch.ones(input_size)
prior = MultivariateNormal(mean_x, torch.diag(var_x))

convs = [Invertible1x1Conv(input_size) for i in range(nflow)]
norms = [ActNorm(input_size) for _ in range(nflow)]
couplings = [AffineCoupling(input_size, hidden_size, i) for i in range(nflow)]
flows = []
for cv, nm, cp in zip(convs, norms, couplings):
    flows += [nm, cv, cp]


### Choix du prior

prior_type = "mixture"   # ou "simple" pour un prior gaussien simple

if prior_type == "mixture":
    mix = torch.distributions.Categorical(torch.ones(2, ))
    comp = torch.distributions.Independent(torch.distributions.Normal(torch.tensor([[-0.5, -0.5], [0.5, 0.5]]), torch.ones(2, 2) * 0.2), 1)
    prior = torch.distributions.MixtureSameFamily(mix, comp)
elif prior_type == "simple":
    mean_x = torch.zeros(input_size)
    var_x = 0.2 * torch.ones(input_size)
    prior = MultivariateNormal(mean_x, torch.diag(var_x))

# Entraînement

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