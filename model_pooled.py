# ══════════════════════════════════════════════════════════════════════════════
# model_pooled.py — NCVA with a SUBJECT EMBEDDING (pooled multi-subject training)
#
# Option A placement: the subject embedding enters ONLY the encoding half
#   - EncoderX      (EEG → z_X)
#   - PosteriorDDM  (EEG → v, a, ndt)
# The decoder and bridgers stay subject-agnostic (single-subject model.py logic).
#
# Mechanism: a shared "context" vector  ctx = [condition_onehot ‖ subj_emb(id)]
# replaces the bare condition wherever the encoder/posteriorDDM used it.
# ctx_dim = c_dim + emb_dim.  The decoder still uses the plain condition.
#
# Single-subject model.py is left intact — this is a separate file.
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Z_DIM, D_FEAT

# reuse the unchanged building blocks from the single-subject model
from model import (latent_sample, NetDown, NetUp, SelfAttention,
                   FeatureExtractor, DecoderX, BridgerDDM, BridgerEEG)

EMB_DIM = 8          # subject embedding size


class EncoderXSubj(nn.Module):
    """EncoderX, but its condition input is [cond ‖ subject_embedding]."""
    def __init__(self, eeg_chan, ctx_dim, z_dim=Z_DIM, d=D_FEAT):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fe = FeatureExtractor(eeg_chan, d)
        self.fc = nn.Linear(128*16 + ctx_dim, z_dim*2)

    def forward(self, x, ctx):
        x       = self.dropout(x)
        feat, _ = self.fe(x)
        return self.fc(torch.cat([feat, ctx], 1))


class PosteriorDDMSubj(nn.Module):
    """PosteriorDDM, but its condition input is [cond ‖ subject_embedding]."""
    def __init__(self, eeg_chan, ctx_dim, z_dim=Z_DIM, max_ndt=1.0):
        super().__init__()
        self.fe      = FeatureExtractor(eeg_chan, dropout=True)
        in_sz        = 128*16 + ctx_dim
        self.fv_mu   = nn.Linear(in_sz, 1)
        self.fv_lv   = nn.Linear(in_sz, 1)
        self.fa_mu   = nn.Linear(in_sz, 1)
        self.fa_lv   = nn.Linear(in_sz, 1)
        self.fn_mu   = nn.Linear(in_sz, 1)
        self.fn_lv   = nn.Linear(in_sz, 1)
        self.fc_     = nn.Sequential(nn.Linear(in_sz, 1), nn.Tanh())
        self.max_ndt = max_ndt

    def forward(self, x, ctx):
        feat, _ = self.fe(x)
        f       = torch.cat([feat, ctx], 1)
        v_stat  = torch.cat([self.fv_mu(f), self.fv_lv(f)], 1)
        a_mu    = F.softplus(self.fa_mu(f)) + 0.1
        a_stat  = torch.cat([a_mu, self.fa_lv(f)], 1)
        n_mu    = torch.sigmoid(self.fn_mu(f)) * self.max_ndt
        n_stat  = torch.cat([n_mu, self.fn_lv(f)], 1)
        choice  = self.fc_(f)
        return v_stat, a_stat, n_stat, choice


class EEGVAEPooled(nn.Module):
    """
    Pooled NCVA. A single nn.Embedding(n_subjects, EMB_DIM) is SHARED between
    the encoder and the posteriorDDM (passed in from the training loop).

    forward(x, c, z_ddm, subj_idx) — same signature as the single-subject VAE
    plus subj_idx.  Decoder is subject-agnostic (uses plain c).
    """
    def __init__(self, eeg_chan, time_steps, c_dim, n_subjects,
                 z_dim=Z_DIM, emb_dim=EMB_DIM):
        super().__init__()
        self.z_dim   = z_dim
        self.emb_dim = emb_dim
        self.subj_emb = nn.Embedding(n_subjects, emb_dim)
        ctx_dim = c_dim + emb_dim

        self.encoderX   = EncoderXSubj(eeg_chan, ctx_dim, z_dim)
        self.decoderX   = DecoderX(eeg_chan, time_steps, c_dim, z_dim)   # plain c
        self.bridgerDDM = BridgerDDM(c_dim, z_dim)                       # plain c
        self.bridgerEEG = BridgerEEG(c_dim, z_dim)                       # plain c

    def _ctx(self, c, subj_idx):
        """Build [condition ‖ subject_embedding]."""
        c_flat = c.squeeze(1) if c.dim() == 3 else c          # (B, c_dim)
        e      = self.subj_emb(subj_idx)                      # (B, emb_dim)
        return torch.cat([c_flat, e], dim=1)                  # (B, ctx_dim)

    def forward(self, x, c, z_ddm, subj_idx):
        ctx        = self._ctx(c, subj_idx)
        enc        = self.encoderX(x, ctx)
        mu_x, lv_x = enc[:, :self.z_dim], enc[:, self.z_dim:]
        zX         = latent_sample(mu_x, lv_x)
        recon      = self.decoderX(zX, z_ddm, c)              # decoder uses plain c
        return recon, mu_x, lv_x


def build_pooled_models(eeg_chan, time_steps, c_dim, n_subjects,
                        z_dim=Z_DIM, emb_dim=EMB_DIM, device=None):
    if device is None:
        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu")

    vae   = EEGVAEPooled(eeg_chan, time_steps, c_dim, n_subjects, z_dim, emb_dim).to(device)
    # PosteriorDDM shares the SAME embedding as the VAE encoder
    p_ddm = PosteriorDDMSubj(eeg_chan, c_dim + emb_dim, z_dim).to(device)

    n_vae  = sum(p.numel() for p in vae.parameters())
    n_pddm = sum(p.numel() for p in p_ddm.parameters())
    print(f"✅  Pooled models on {device}")
    print(f"    eeg_chan={eeg_chan} time_steps={time_steps} c_dim={c_dim} "
          f"n_subjects={n_subjects} emb_dim={emb_dim}")
    print(f"    VAE params   : {n_vae:,}   PosteriorDDM : {n_pddm:,}")
    return vae, p_ddm, device


# helper so the training loop can build ctx for the (separate) posteriorDDM
def make_ctx(vae, c, subj_idx):
    """ctx = [cond ‖ shared subject embedding], using the VAE's embedding table."""
    c_flat = c.squeeze(1) if c.dim() == 3 else c
    e      = vae.subj_emb(subj_idx)
    return torch.cat([c_flat, e], dim=1)