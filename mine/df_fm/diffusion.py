import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import FakeActionDataset
from model import TrajModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = FakeActionDataset(
    chunk_size=50,
    size=1000,
    type=3
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = TrajModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_steps = 1000
beta_start = 1e-4
beta_end = 2e-2

betas = torch.linspace(
    beta_start,
    beta_end,
    num_steps,
    device=device
)

alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def q_sample(x_0, k, eps):
    alpha_bar_k = alpha_bars[k][:, None, None]
    x_k = alpha_bar_k.sqrt() * x_0
    x_k = x_k + (1 - alpha_bar_k).sqrt() * eps
    return x_k

def diffusion_loss(model, x_0):
    batch_size = x_0.shape[0]

    k = torch.randint(0, num_steps, (batch_size,), device=x_0.device)

    eps = torch.randn_like(x_0)

    x_k = q_sample(x_0, k, eps)

    t = k.float() / (num_steps - 1)

    eps_pred = model(x_k, t)

    return F.mse_loss(eps, eps_pred)

for epoch in range(1000):
    model.train()
    total_loss = 0.0

    for x_0 in dataloader:
        x_0 = x_0.to(device)

        optimizer.zero_grad()

        loss = diffusion_loss(model, x_0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}, Loss: {average_loss:.6f}")

torch.save(model.state_dict(), "diffusion.pth")
