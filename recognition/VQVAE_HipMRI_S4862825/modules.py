import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)


    def forward(self, x):
        return x

def C(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, s, p), nn.ReLU(inplace=True))

def D(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, k, s, p), nn.ReLU(inplace=True))

class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            C(in_ch, 64),      # 128 to 64
            C(64, 128),        # 64 to 32
            Residual(128),
            C(128, hidden),    # 32 to 16
            Residual(hidden),
            nn.Conv2d(hidden, z_dim, 1),
        )
    def forward(self, x): 
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z_dim=64, hidden=128, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_dim, hidden, 1),
            Residual(hidden),
            D(hidden, 128),    # 16 to 32
            Residual(128),
            D(128, 64),        # 32 to 64
            D(64, 32),         # 64 to 128
            nn.Conv2d(32, out_ch, 3, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, z): 
        return self.net(z)

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        #Wire encoder/decoder and VQ block

    def forward(self, x):
        return x, torch.tensor(0.0), {}