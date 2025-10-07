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
class Codebook(nn.Module):
    def __init__(self, n_codes: int = 512, dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.beta = beta
        self.embed = nn.Embedding(n_codes, dim)
        nn.init.uniform_(self.embed.weight, -1.0/n_codes, 1.0/n_codes)

    def forward(self, z_e):   # (B, D, H, W)
        B, D, H, W = z_e.shape
        z = z_e.permute(0,2,3,1).contiguous().view(-1, D)     # (BHW, D)
        e = self.embed.weight                                  # (K, D)

        # squared Euclidean distance
        d = (z.pow(2).sum(1, keepdim=True)
             - 2*z @ e.t()
             + e.pow(2).sum(1, keepdim=True).t())             # (BHW, K)

        idx = torch.argmin(d, dim=1)
        z_q = e[idx].view(B, H, W, D).permute(0,3,1,2).contiguous()

        # losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commit_loss   = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # perplexity (monitoring)
        with torch.no_grad():
            enc = F.one_hot(idx, e.shape[0]).float()
            avg = enc.mean(0)
            perp = torch.exp(-(avg * torch.log(avg + 1e-10)).sum())

        return z_q, vq_loss, perp

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
    def __init__(self, img_ch=1, hidden=128, z_dim=64, n_codes=512, beta=0.25):
        super().__init__()
        self.enc = Encoder(img_ch, hidden, z_dim)
        self.vq  = Codebook(n_codes, z_dim, beta)
        self.dec = Decoder(z_dim, hidden, img_ch)

    def forward(self, x):
        z_e = self.enc(x)
        z_q, vq_loss, perp = self.vq(z_e)
        x_hat = self.dec(z_q)
        recon = F.mse_loss(x_hat, x)
        total = recon + vq_loss
        return x_hat, total, {"recon": recon.detach(), "vq": vq_loss.detach(), "perp": perp.detach()}