import math
import data
import torch
import torch.nn as nn

def timestep_embedding(t, dim = 64):
    '''
    
    '''
    half = dim // 2

    freq = torch.exp(
        torch.linspace(
            0, math.log(1000),half,device=t.device
        )
    )

    x = 2 * math.pi * t[:, None] * freq[None, :]

    emb = torch.cat(
        [
            torch.sin(x),
            torch.cos(x)
        ],
        dim=-1
    )

    return emb

# t = torch.tensor([1.0])
# print(t)
# emb = timestep_embedding(t)
# print(emb)

class TrajModel(nn.Module):
    def __init__(
        self,
        chunk_size = 50,
        action_dim = 2,
        time_dim = 64,
        hidden_dim = 512
        ):
        super().__init__()

        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.time_dim = time_dim
        self.input_dim = chunk_size * action_dim + time_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.input_dim - self.time_dim),
        )

    def forward(self, x_t, t):
        batch_size = x_t.shape[0]

        x_t = x_t.reshape(batch_size, -1)

        t = timestep_embedding(t, self.time_dim)

        x = torch.cat([x_t, t], dim=-1)

        return self.net(x).reshape(batch_size, self.chunk_size, self.action_dim)

# model = TrajModel()

# x_t = torch.randn(1, 50, 2)
# t = torch.tensor([1.0])

# print(model(x_t, t))