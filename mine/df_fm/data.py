import torch
from torch.utils.data import Dataset, DataLoader

class FakeActionDataset(Dataset):
    def __init__(self, chunk_size = 50, size = 1000, type = 3):
        self.chunk_size = chunk_size
        self.size = size
        self.type = type
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        traj_type = torch.randint(0, self.type, (1,)).item()

        t = torch.linspace(0, 1, self.chunk_size)

        if traj_type == 0:
            y = t
            x = t

        if traj_type == 1:
            y = torch.zeros_like(t)
            x = t

        if traj_type == 2:
            x = torch.zeros_like(t)
            y = t
        
        traj = torch.stack([x,y],dim=1)
        return traj

if __name__ == "__main__":
    dataset = FakeActionDataset(
        chunk_size=50,
        size=1000,
        type=3
    )

    traj = dataset[0]
    x = traj[:, 0]
    y = traj[:, 1]  
    print("traj.shape", traj.shape, "\ntraj",traj)

