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

class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128, z_dim=64):
        super().__init__()
        #conv downsamples

    def forward(self, x):
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim=64, hidden=128, out_ch=1):
        super().__init__()
        #conv transpose upsamples

    def forward(self, z):
        return z

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        #Wire encoder/decoder and VQ block

    def forward(self, x):
        return x, torch.tensor(0.0), {}