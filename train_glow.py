from glow import *
from torchvision import datasets, transforms
from torch.distributions import Independent, Normal

train_dataset = datasets.CIFAR10(
    root='./cifar_data/', train=True,
    transform=transforms.ToTensor(), download=True
)

# toy dataset
C, H, W = 3, 32, 32
x = torch.randn(10, C, H, W)
n_epochs = 10

gaussian_prior = Independent(Normal(torch.zeros((int(C), H, W)).flatten(), torch.ones((int(C), H, W)).flatten()), 1)

model = GLOW(prior=gaussian_prior, n_block=3, n_flow=5, n_channels=3, hidden_channels=512)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

for t in range(n_epochs):
    log_prob, _, log_det_ = model.encoder(x)
    loss = -(log_prob + log_det_).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Iteration {t+1} : loss = {loss.item()}")
