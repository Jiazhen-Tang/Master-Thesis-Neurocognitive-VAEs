# ══════════════════════════════════════════════════════════════════════════════
# losses.py — All loss functions for NCVA training
# ══════════════════════════════════════════════════════════════════════════════

import math
import torch
import torch.nn.functional as F


def kl_standard(mu, logvar):
    """KL[ q(z|x) || N(0,I) ] averaged over batch."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def kl_between(mu1, lv1, mu2, lv2):
    """KL[ N(mu1,s1) || N(mu2,s2) ] averaged over batch."""
    return 0.5 * torch.mean(
        lv2 - lv1 - 1 + (lv1.exp() + (mu1 - mu2).pow(2)) / lv2.exp().clamp(min=1e-8)
    )


def wiener_loss(t, v, t0, a, backprop_ndt=False):
    """
    Wiener First Passage Time negative log-likelihood.
    Navarro & Fuss (2009) small-time series approximation.

    Parameters
    ----------
    t  : (B,1) signed RT — negative = correct (lower boundary)
    v  : (B,1) drift rate
    t0 : (B,1) non-decision time (positive, < |t|)
    a  : (B,1) boundary separation (positive)
    """
    dev = t.device
    w   = torch.tensor(0.5, device=dev)
    err = torch.tensor(0.02, device=dev)
    kk  = torch.arange(-4, 6, device=dev, dtype=t.dtype
                        ).unsqueeze(0).expand(t.shape[0], -1)

    if backprop_ndt:
        tt = torch.max(t.abs() - t0, err) / torch.max(err, a)**2
    else:
        tt = torch.max(t.abs() - t0.detach(), err) / torch.max(err, a)**2

    tt_vec = tt.expand(-1, 10)
    pp = torch.cumsum(
        (w + 2*kk) * torch.exp(-((w + 2*kk)**2 / 2) / tt_vec), dim=1)
    pp = pp[:, -1:] / torch.sqrt(
        2 * torch.tensor(math.pi, device=dev) * tt**3)

    v = torch.where(t > 0, -v.abs(), v).clamp(-6, 6)
    p = pp * torch.exp(
        -v * a.clamp(min=err) * w - v**2 * t.abs() / 2
    ) / a.clamp(min=err)**2

    return -torch.log(p.clamp(min=1e-30)).mean()


def condition_wise_average(eeg, cond):
    """
    Replace each trial's EEG with the mean over all trials of the same condition.
    Used for the trial-averaging reconstruction loss.

    eeg  : (B, C, T)
    cond : (B, 1, c_dim) or (B, c_dim)
    """
    c_flat = cond.squeeze(1) if cond.dim() == 3 else cond
    idx    = torch.argmax(c_flat, -1)
    out    = torch.zeros_like(eeg)
    for ci in torch.unique(idx):
        m = idx == ci
        if m.sum() > 0:
            out[m] = eeg[m].mean(0, keepdim=True).expand_as(eeg[m])
    return out


def pearson_r(a, b):
    """Pearson correlation between two tensors (detached)."""
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() < 2:
        return torch.tensor(0.0)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b) / (a.norm() * b.norm()).clamp(min=1e-8)
