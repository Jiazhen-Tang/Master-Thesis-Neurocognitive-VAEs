# ══════════════════════════════════════════════════════════════════════════════
# svd_filter.py — SVD spatial filter (collapse multichannel EEG → 1 ERP channel)
#
# Replicates the original Nunez/Vo "v0" projection:
#   - Compute the first right singular vector of the condition-averaged ERP
#     (optionally within an analysis window).
#   - Project every trial through it: erp = Σ_channels  v0 · eeg  →  (N, T)
#
# If the input already has a single channel (C == 1) the filter is trivially
# [±1] and the projection is the channel itself, so this degrades gracefully.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch


def compute_svd_filter(erp, window=None, sign_ref_idx=None):
    """
    Parameters
    ----------
    erp          : (C, T) numpy  condition-averaged ERP
    window       : (start, end) sample indices to run SVD over; None = full epoch
    sign_ref_idx : list[int] channel indices that should be POSITIVE
                   (e.g. parietal for CPP/P300). Flips the filter if their
                   summed weight is negative, for cross-subject sign consistency.

    Returns
    -------
    v0 : (C,) float32  spatial filter (first right singular vector)
    """
    seg = erp[:, window[0]:window[1]] if window is not None else erp
    # SVD on (T, C): right singular vectors span the channel space
    u, s, vt = np.linalg.svd(seg.T, full_matrices=False)
    v0 = vt[0].copy()

    if sign_ref_idx is not None and len(sign_ref_idx) > 0:
        if v0[sign_ref_idx].sum() < 0:
            v0 = -v0

    return v0.astype("float32")


def svd_project_np(eeg, v0):
    """eeg (N, C, T) numpy, v0 (C,) → (N, T) numpy."""
    return np.tensordot(eeg, v0, axes=([1], [0]))


def svd_project_torch(eeg, v0):
    """eeg (B, C, T) tensor, v0 (C,) tensor → (B, T) tensor."""
    return torch.einsum('bct,c->bt', eeg, v0)