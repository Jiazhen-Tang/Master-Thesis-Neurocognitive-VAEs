# ══════════════════════════════════════════════════════════════════════════════
# model.py — NCVA model architecture
#
# All classes accept eeg_chan, time_steps, c_dim as constructor arguments
# so the same code works for:
#   - Stimulus-locked ERP : (N, C, ~602)
#   - Response-locked ERP : (N, C, ~602)
#   - FRP                 : (N, 17, ~90)
#
# CHANGE (replicates original):
#   * DecoderX now consumes the FULL 3D DDM latent [v, a, ndt] (was 1D before).
#   * EEGVAE.forward takes the sampled DDM latent z_ddm and passes it to the
#     decoder — matching the original where the decoder is conditioned on
#     [v, a, ndt] from the PosteriorDDM, not the raw RT.
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Z_DIM, D_FEAT


def latent_sample(mu, logvar):
    """Reparameterization trick: z = mu + eps * sigma."""
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


class NetDown(nn.Module):
    def __init__(self, d, dropout=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d*4, d*4, 6, 1, 3, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(d*4), nn.LeakyReLU(0.1),
            nn.Conv1d(d*4, d*4, 6, 2, 2, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(d*4), nn.LeakyReLU(0.1),
        )
        self.do = nn.Dropout(0.7) if dropout else nn.Identity()

    def forward(self, x):
        return self.do(self.net(x))


class NetUp(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=6, stride=3, output_padding=0):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride,
                               output_padding=output_padding),
            nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.1),
        )
        self.fin = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride,
                                      output_padding=output_padding)

    def forward(self, x, final_layer=False):
        return self.fin(x) if final_layer else self.up(x)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.q  = nn.Conv2d(in_dim, in_dim//2, 1)
        self.k  = nn.Conv2d(in_dim, in_dim//2, 1)
        self.v  = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        xi = x.unsqueeze(-1)
        B, C, W, H = xi.shape
        q   = self.q(xi).view(B, -1, W*H).permute(0, 2, 1)
        k   = self.k(xi).view(B, -1, W*H)
        v   = self.v(xi).view(B, -1, W*H)
        att = self.sm(torch.bmm(q, k))
        out = torch.bmm(v, att.permute(0, 2, 1)).view(B, C, W, H)
        return (self.gamma * out + xi).squeeze(-1), att


class FeatureExtractor(nn.Module):
    def __init__(self, eeg_chan, d=D_FEAT, dropout=False):
        super().__init__()
        self.down0 = nn.Sequential(
            nn.Conv1d(eeg_chan, d*4, 1), nn.LeakyReLU(0.1))
        self.down1 = NetDown(d, dropout)
        self.down2 = NetDown(d, dropout)
        self.att1  = SelfAttention(d*4)
        self.down3 = NetDown(d, dropout)
        self.down4 = NetDown(d, dropout)
        self.pool  = nn.AdaptiveAvgPool1d(16)

    def forward(self, x):
        x = self.down0(x)
        x = self.down1(x); x = self.down2(x)
        x, _ = self.att1(x)
        x = self.down3(x); x = self.down4(x)
        x = self.pool(x)
        return torch.flatten(x, 1), None   # (B, 128*16)


class EncoderX(nn.Module):
    def __init__(self, eeg_chan, c_dim, z_dim=Z_DIM, d=D_FEAT):
        super().__init__()
        self.dropout = nn.Dropout(0.3)          # input dropout (matches original)
        self.fe = FeatureExtractor(eeg_chan, d)
        self.fc = nn.Linear(128*16 + c_dim, z_dim*2)

    def forward(self, x, c):
        x       = self.dropout(x)               # regularize the EEG input
        feat, _ = self.fe(x)
        c_flat  = c.squeeze(1) if c.dim()==3 else c
        return self.fc(torch.cat([feat, c_flat], 1))


class BridgerDDM(nn.Module):
    def __init__(self, c_dim, z_dim=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 128), nn.LeakyReLU(0.1))
        self.fc = nn.Linear(128 + c_dim, 6)   # 3 means + 3 logvars

    def forward(self, y, c):
        h      = self.net(y)
        c_flat = c.squeeze(1) if c.dim()==3 else c
        return self.fc(torch.cat([h, c_flat], 1))


class BridgerEEG(nn.Module):
    def __init__(self, c_dim, z_dim=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 128), nn.LeakyReLU(0.1))
        self.fc = nn.Linear(128 + c_dim, z_dim*2)

    def forward(self, ddm, c):
        h      = self.net(ddm)
        c_flat = c.squeeze(1) if c.dim()==3 else c
        return self.fc(torch.cat([h, c_flat], 1))


class DecoderX(nn.Module):
    def __init__(self, eeg_chan, time_steps, c_dim, z_dim=Z_DIM, hid_dim=128,
                 cond_addition=False):
        super().__init__()
        self.cond_addition = cond_addition

        # ── Project the FULL 3D DDM latent [v, a, ndt] ──────────────────────
        # (was nn.Linear(1, ...) which only received the drift rate / raw RT).
        # This matches the original decoder, which is conditioned on all three
        # decision parameters.
        self.fc = nn.Sequential(
            nn.Linear(3, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim // 4),
            nn.LeakyReLU(0.1),
        )

        # Up-sampling layers
        self.netD_up1 = NetUp(z_dim + hid_dim // 4, hid_dim * 4, kernel_size=8, stride=4)
        self.netD_up2 = NetUp(hid_dim * 4, hid_dim * 2, output_padding=1, kernel_size=8, stride=4)
        self.netA_1   = SelfAttention(hid_dim * 2)
        self.netD_up3 = NetUp(hid_dim * 2, hid_dim, output_padding=2, kernel_size=6, stride=3)
        self.netD_up4 = NetUp(hid_dim, hid_dim, stride=1, output_padding=0, kernel_size=6)
        self.netA_2   = SelfAttention(hid_dim)
        self.netD_up5 = NetUp(hid_dim, eeg_chan, kernel_size=10, stride=2, output_padding=0)

        # Force the output to the configured number of timepoints
        self.final_pool = nn.AdaptiveAvgPool1d(time_steps)

    def forward(self, zX, z_ddm, c):
        """
        zX    : (B, z_dim)   EEG latent
        z_ddm : (B, 3)       sampled DDM latent [v, a, ndt]
        c     : (B, 1, c_dim) or (B, c_dim)  one-hot condition
        """
        B = zX.size(0)

        z_ddm_feat = self.fc(z_ddm)                       # (B, hid//4)

        # ── Condition gating (original method) ──────────────────────────────
        # argmax(c) + 1  →  c_idx ∈ {1, 2}, NEVER 0, so multiplying does not
        # zero the DDM features on "easy" trials (that was the earlier bug,
        # caused by using argmax WITHOUT the +1).
        c_flat = c.squeeze(1) if c.dim() == 3 else c
        c_idx  = (torch.argmax(c_flat, dim=-1) + 1).float().unsqueeze(1)   # (B, 1)
        if self.cond_addition:
            z_ddm_feat = z_ddm_feat + c_idx
        else:
            z_ddm_feat = z_ddm_feat * c_idx

        zX_flat    = zX.view(B, -1)
        z_combined = torch.cat([zX_flat, z_ddm_feat.view(B, -1)], dim=1)

        # Inflate to 3D for Conv layers: (B, Features, Time=1)
        z = z_combined.unsqueeze(2)

        # Up-sampling chain
        z = self.netD_up1(z); z = self.netD_up2(z)
        z, _ = self.netA_1(z)
        z = self.netD_up3(z); z = self.netD_up4(z)
        z, _ = self.netA_2(z)

        # Final reconstruction & pooling
        eeg = self.netD_up5(z, final_layer=True)
        eeg = self.final_pool(eeg)

        return eeg.unsqueeze(1)   # (B, 1, eeg_chan, time_steps)


class EEGVAE(nn.Module):
    def __init__(self, eeg_chan, time_steps, c_dim, z_dim=Z_DIM):
        super().__init__()
        self.encoderX   = EncoderX(eeg_chan, c_dim, z_dim)
        self.decoderX   = DecoderX(eeg_chan, time_steps, c_dim, z_dim)
        self.bridgerDDM = BridgerDDM(c_dim, z_dim)
        self.bridgerEEG = BridgerEEG(c_dim, z_dim)
        self.z_dim      = z_dim

    def forward(self, x, c, z_ddm):
        """
        x     : (B, eeg_chan, T)  EEG input
        c     : (B, 1, c_dim)     condition one-hot
        z_ddm : (B, 3)            sampled DDM latent [v, a, ndt]

        Returns
        -------
        recon, mu_x, lv_x
        """
        enc         = self.encoderX(x, c)
        mu_x, lv_x  = enc[:, :self.z_dim], enc[:, self.z_dim:]
        zX          = latent_sample(mu_x, lv_x)
        recon       = self.decoderX(zX, z_ddm, c)
        return recon, mu_x, lv_x


class PosteriorDDM(nn.Module):
    def __init__(self, eeg_chan, c_dim, z_dim=Z_DIM, max_ndt=1.0):
        super().__init__()
        self.fe      = FeatureExtractor(eeg_chan, dropout=True)
        in_sz        = 128*16 + c_dim
        self.fv_mu   = nn.Linear(in_sz, 1)
        self.fv_lv   = nn.Linear(in_sz, 1)
        self.fa_mu   = nn.Linear(in_sz, 1)
        self.fa_lv   = nn.Linear(in_sz, 1)
        self.fn_mu   = nn.Linear(in_sz, 1)
        self.fn_lv   = nn.Linear(in_sz, 1)
        self.fc_     = nn.Sequential(nn.Linear(in_sz, 1), nn.Tanh())
        self.max_ndt = max_ndt

    def forward(self, x, c):
        feat, _  = self.fe(x)
        c_flat   = c.squeeze(1) if c.dim()==3 else c
        f        = torch.cat([feat, c_flat], 1)
        v_stat   = torch.cat([self.fv_mu(f), self.fv_lv(f)], 1)
        a_mu     = F.softplus(self.fa_mu(f)) + 0.1
        a_stat   = torch.cat([a_mu, self.fa_lv(f)], 1)
        n_mu     = torch.sigmoid(self.fn_mu(f)) * self.max_ndt
        n_stat   = torch.cat([n_mu, self.fn_lv(f)], 1)
        choice   = self.fc_(f)
        return v_stat, a_stat, n_stat, choice


def build_models(eeg_chan, time_steps, c_dim, z_dim=Z_DIM, device=None):
    """
    Convenience function: build all models with auto-detected dimensions.

    Returns
    -------
    vae, p_ddm, device
    """
    import torch
    if device is None:
        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    vae   = EEGVAE(eeg_chan, time_steps, c_dim, z_dim).to(device)
    p_ddm = PosteriorDDM(eeg_chan, c_dim, z_dim).to(device)

    n_vae  = sum(p.numel() for p in vae.parameters())
    n_pddm = sum(p.numel() for p in p_ddm.parameters())

    print(f"✅  Models built on {device}")
    print(f"    eeg_chan={eeg_chan}  time_steps={time_steps}  c_dim={c_dim}  z_dim={z_dim}")
    print(f"    VAE params      : {n_vae:,}")
    print(f"    PosteriorDDM    : {n_pddm:,}")

    return vae, p_ddm, device
