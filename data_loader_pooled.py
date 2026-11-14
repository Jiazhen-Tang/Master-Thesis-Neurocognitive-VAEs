# ══════════════════════════════════════════════════════════════════════════════
# data_loader_pooled.py — pooled multi-subject loading (Level 1 + 2 prep)
#
# load_pooled_data(subject_ids, signal_type):
#   - loads each subject's 64-channel _eeg.npy
#   - per-subject GLOBAL scaling (single mean/std) — matches original study,
#     preserves between-channel relative amplitude for the SVD collapse
#   - within-subject RT z-score         (separates between/within-subject RT)
#   - builds subject_idx (0..S-1) aligned to every trial  (for the embedding)
#   - concatenates all subjects into single tensors
#
# Returns everything load_signal_data returns, PLUS subject_idx and n_subjects.
# Keeps raw (un-normalised) RT too, since evaluation/plots need real seconds.
#
# This is ADDITIVE — your single-subject load_signal_data is untouched.
# ══════════════════════════════════════════════════════════════════════════════

import os, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import TENSOR_DIR

CHANNEL_MODE = "eeg"                      # pooled run uses the clean 64ch tensors
_SUFFIX = {"eeg": "_eeg", "pick": "_pick", "cpp": "_cpp", "full": ""}


def _load_one(subj_id, signal_type, tensor_dir):
    """Load one subject's X, rt, cond, choice from the chosen channel variant."""
    d = os.path.join(tensor_dir, f"subj_{subj_id}")
    suffix = _SUFFIX[CHANNEL_MODE]
    xpath  = os.path.join(d, f"subj_{subj_id}_X_{signal_type}{suffix}.npy")
    if not os.path.exists(xpath):
        raise FileNotFoundError(
            f"{xpath} missing. Build the {CHANNEL_MODE} tensor for subj {subj_id} first.")
    X = np.load(xpath)                                   # (N, C, T)
    with open(os.path.join(d, f"subj_{subj_id}_beh.pkl"), "rb") as f:
        beh = pickle.load(f)
    return X, beh["rt"], beh["condition"], beh["choice"]


def load_pooled_data(subject_ids, signal_type, tensor_dir=None, crop_t=400):
    """
    Parameters
    ----------
    subject_ids : list[str]   e.g. ["01","02","03","04"]
    signal_type : "stim" | "resp"
    crop_t      : int | None  keep only the first crop_t time points (speed).
                  400 ≈ -0.2s..2.9s at 128Hz, covers the P300 window.
                  Set None to keep the full 602.

    Returns
    -------
    X_raw        : (N_total, C, T)  per-subject-per-channel z-scored
    rt_norm      : (N_total,)       within-subject z-scored RT (model input)
    rt_seconds   : (N_total,)       raw RT in seconds (for eval/plots)
    cond         : (N_total, 2)     condition one-hot
    choice       : (N_total,)
    subject_idx  : (N_total,)       int 0..S-1, aligned to every trial
    eeg_chan, time_steps, c_dim, n_subjects : ints
    """
    tensor_dir = tensor_dir or TENSOR_DIR

    Xs, RTn, RTs, Cs, Ys, SI = [], [], [], [], [], []
    print(f"\n{'='*60}\n  Pooling {len(subject_ids)} subjects | {signal_type}\n{'='*60}")

    for si, subj_id in enumerate(subject_ids):
        X, rt, cond, choice = _load_one(subj_id, signal_type, tensor_dir)

        # crop time window (keep first crop_t points) — big speedup, P300 kept
        if crop_t is not None and X.shape[2] > crop_t:
            X = X[:, :, :crop_t]

        # drop NaN-RT trials
        rt = np.asarray(rt, dtype="float32")
        valid  = ~np.isnan(rt)
        X, rt   = X[valid], rt[valid]
        cond    = np.asarray(cond, dtype="float32")[valid]
        choice  = np.asarray(choice, dtype="float32")[valid]

        # ── Global scaling (matches the original study) ───────────────────────
        # Original NCVA uses a SINGLE global mean/std over the whole EEG array,
        # so between-channel relative amplitude is preserved (e.g. a large P300
        # at Pz stays large). This differs from per-channel z-scoring, which
        # would flatten channel-to-channel amplitude differences and can distort
        # the subsequent SVD collapse.
        mu  = X.mean()                                    # single scalar
        sd  = X.std() + 1e-8                              # single scalar
        Xz  = ((X - mu) / sd).astype("float32")

        # ── within-subject RT z-score (model input) ───────────────────────────
        rt_mu, rt_sd = rt.mean(), rt.std() + 1e-8
        rt_z = ((rt - rt_mu) / rt_sd).astype("float32")

        Xs.append(Xz)
        RTn.append(rt_z)
        RTs.append(rt)                                   # keep raw seconds
        Cs.append(cond)
        Ys.append(choice)
        SI.append(np.full(len(Xz), si, dtype="int64"))

        n_hard = int((cond[:, 1] == 1).sum())
        print(f"  [{si}] subj {subj_id}: n={len(Xz):4d}  "
              f"RT μ={rt_mu:.2f}s σ={rt_sd:.2f}  hard={n_hard}")

    X_raw      = np.concatenate(Xs, 0)
    rt_norm    = np.concatenate(RTn, 0)
    rt_seconds = np.concatenate(RTs, 0)
    cond       = np.concatenate(Cs, 0)
    choice     = np.concatenate(Ys, 0)
    subject_idx = np.concatenate(SI, 0)

    _, eeg_chan, time_steps = X_raw.shape
    c_dim       = cond.shape[1]
    n_subjects  = len(subject_ids)

    print(f"{'-'*60}")
    print(f"  Pooled total: {len(X_raw)} trials | {eeg_chan}ch × {time_steps} | "
          f"{n_subjects} subjects")
    print(f"  X shape {X_raw.shape}  subject_idx range 0..{subject_idx.max()}")
    print(f"{'='*60}")

    return (X_raw, rt_norm, rt_seconds, cond, choice, subject_idx,
            eeg_chan, time_steps, c_dim, n_subjects)


def make_pooled_dataloaders(X_raw, rt_norm, cond, choice, subject_idx,
                            batch_size=16, val_frac=0.2, seed=42):
    """
    Train/val split STRATIFIED BY SUBJECT (each subject appears in both splits),
    so val isn't dominated by a few people.

    Returns train_loader, val_loader, and the full tensors (incl. subj_t).
    """
    rng = np.random.default_rng(seed)
    tr_mask = np.zeros(len(X_raw), dtype=bool)
    for s in np.unique(subject_idx):
        idx = np.where(subject_idx == s)[0]
        rng.shuffle(idx)
        n_val = max(1, int(val_frac * len(idx)))
        tr_mask[idx[n_val:]] = True                      # rest → train

    X_t    = torch.tensor(X_raw,      dtype=torch.float32)
    rt_t   = torch.tensor(rt_norm,    dtype=torch.float32).unsqueeze(1)
    cond_t = torch.tensor(cond,       dtype=torch.float32).unsqueeze(1)
    y_t    = torch.tensor(choice,     dtype=torch.float32)
    subj_t = torch.tensor(subject_idx, dtype=torch.long)

    tr = torch.tensor(tr_mask, dtype=torch.bool)
    va = ~tr

    def _ds(m):
        return TensorDataset(X_t[m], rt_t[m], cond_t[m], y_t[m], subj_t[m])

    train_loader = DataLoader(_ds(tr), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(_ds(va), batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"  Train: {int(tr.sum())} trials ({len(train_loader)} batches)  "
          f"Val: {int(va.sum())} trials ({len(val_loader)} batches)  (stratified by subject)")

    return train_loader, val_loader, X_t, rt_t, cond_t, y_t, subj_t


if __name__ == "__main__":
    # quick smoke test on 4 subjects
    subs = ["01", "02", "03", "04"]
    out = load_pooled_data(subs, "stim")
    X_raw, rt_norm, rt_sec, cond, choice, subj_idx, ec, ts, cd, ns = out
    make_pooled_dataloaders(X_raw, rt_norm, cond, choice, subj_idx, batch_size=16)
    print("\n✅ smoke test passed — pooled loader works.")