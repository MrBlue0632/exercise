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

def get_x_t(x_1, x_0, t):
    t = t[:, None, None]
    x_t = x_0 + t * (x_1 - x_0)
    return x_t

def flow_matching_loss(model, x_1):
    x_0 = torch.randn_like(x_1)  # 噪声 [B, 50, 2]  

    batch_size = x_1.shape[0]
    t = torch.rand(batch_size).to(device)

    x_t = get_x_t(x_1, x_0, t)
    target_velocity = x_1 - x_0
    pred_velocity = model(x_t, t)
    loss = F.mse_loss(pred_velocity, target_velocity)
    return loss

for epoch in range(1000):
    model.train()
    total_loss = 0
    for x_1 in dataloader:
        x_1 = x_1.to(device)
        loss = flow_matching_loss(model, x_1)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")
torch.save(model.state_dict(), "flowmatching.pth")

    # model.eval()
    # with torch.no_grad():
    #     x_1 = torch.randn(4, 50, 2).to(device)
    #     x_0 = torch.randn_like(x_1)
    #     t = torch.rand(4).to(device)
    #     x_t = get_x_t(x_1, x_0, t)