# ══════════════════════════════════════════════════════════════════════════════
# train_pooled.py — POOLED multi-subject NCVA training
#
# Wires together the pooled pieces:
#   • data_loader_pooled.load_pooled_data      (per-subject norm, subject_idx)
#   • model_pooled.build_pooled_models          (subject embedding, Option A)
#   • PER-SUBJECT priors gathered by subject_idx (not one global prior)
#   • RAW RT in seconds → wiener loss + priors    (DDM lives in seconds)
#   • train/val split STRATIFIED by subject
#
# Success signal = the train/val gap SHRINKS vs single-subject (less overfit),
# not the absolute v↔RT number.
#
# Run:  python train_pooled.py
# ══════════════════════════════════════════════════════════════════════════════

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pickle
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (MODEL_DIR, Z_DIM, P1_END, P2_END, P3_END, N_EPOCHS,
                    NDT_EPOCH, BATCH_SIZE, LR, EVAL_EVERY, SAVE_EVERY,
                    REC_AVG_W, REC_X_W, CHOICE_W, KL_DDM_W, KL_Y_WEIGHT)

SAVE_EVERY = 20        # checkpoint every 20 epochs (overrides config for pooled runs)
BATCH_SIZE = 64        # larger batch for speed on the pooled dataset
from losses              import (kl_standard, kl_between, wiener_loss,
                                 condition_wise_average, pearson_r)
from losses_with_priors  import kl_informed
from data_loader_pooled  import load_pooled_data, make_pooled_dataloaders
from model_pooled        import build_pooled_models, make_ctx

PRIOR_PATH = Path(os.path.expanduser("~/Desktop/NCVA_results/ddm_priors_bayes.pkl"))


def _gather_priors(subject_ids, device):
    """
    Build per-subject prior tensors indexed 0..S-1 (aligned to subject_idx).
    Returns dict of (S,1) tensors for v/a/ndt mu and logvar.
    """
    with open(PRIOR_PATH, "rb") as f:
        P = pickle.load(f)
    def col(param, key):
        return torch.tensor(
            [[P[s]["overall"][param][key]] for s in subject_ids],
            dtype=torch.float32, device=device)                 # (S,1)
    return {
        "v_mu": col("v","mu"),   "v_lv": col("v","logvar"),
        "a_mu": col("a","mu"),   "a_lv": col("a","logvar"),
        "n_mu": col("ndt","mu"), "n_lv": col("ndt","logvar"),
    }


def run_pooled(subject_ids, signal_type):
    print(f"\n{'═'*60}\n  POOLED NCVA — {len(subject_ids)} subjects | {signal_type}\n{'═'*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    (X_raw, rt_norm, rt_seconds, cond, choice, subject_idx,
     eeg_chan, time_steps, c_dim, n_subjects) = load_pooled_data(subject_ids, signal_type, crop_t=None)

    # models (subject embedding lives in the VAE; shared with p_ddm via make_ctx)
    vae, p_ddm, device = build_pooled_models(eeg_chan, time_steps, c_dim, n_subjects, Z_DIM)

    # per-subject priors, gathered by index
    PR = _gather_priors(subject_ids, device)

    # ── DataLoaders (stratified by subject); we feed RAW seconds separately ───
    train_loader, val_loader, X_t, rtN_t, cond_t, y_t, subj_t = \
        make_pooled_dataloaders(X_raw, rt_norm, cond, choice, subject_idx,
                                batch_size=BATCH_SIZE)
    # raw-RT tensor aligned to X_t order (dataloaders carry normalized RT;
    # we need seconds for wiener — build a lookup by re-attaching to the split)
    rtS_t = torch.tensor(rt_seconds, dtype=torch.float32).unsqueeze(1)

    # Because make_pooled_dataloaders shuffles internally, we pass raw RT INTO
    # the loaders instead. Rebuild loaders carrying BOTH rts. (simplest: below)
    from torch.utils.data import DataLoader, TensorDataset
    rng = np.random.default_rng(42)
    tr_mask = np.zeros(len(X_raw), dtype=bool)
    for s in np.unique(subject_idx):
        idx = np.where(subject_idx == s)[0]; rng.shuffle(idx)
        n_val = max(1, int(0.2*len(idx))); tr_mask[idx[n_val:]] = True
    tr = torch.tensor(tr_mask); va = ~tr

    Xt   = torch.tensor(X_raw, dtype=torch.float32)
    ct   = torch.tensor(cond, dtype=torch.float32).unsqueeze(1)
    yt   = torch.tensor(choice, dtype=torch.float32)
    st   = torch.tensor(subject_idx, dtype=torch.long)
    rtSt = torch.tensor(rt_seconds, dtype=torch.float32).unsqueeze(1)

    def _ds(m): return TensorDataset(Xt[m], rtSt[m], ct[m], yt[m], st[m])
    train_loader = DataLoader(_ds(tr), batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(_ds(va), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"  Train {int(tr.sum())} | Val {int(va.sum())}  (stratified, RAW RT to wiener)")

    kl_weight = 20.0 * (Z_DIM + 3) / (eeg_chan * time_steps)
    print(f"  kl_weight = {kl_weight:.6f}\n")

    # ── Optimizers (embedding trains with encoder/DDM groups) ─────────────────
    opt_ddm = optim.Adam(list(p_ddm.parameters()) + list(vae.subj_emb.parameters()), lr=LR)
    opt_vae = optim.Adam(list(vae.encoderX.parameters()) + list(vae.decoderX.parameters())
                         + list(vae.subj_emb.parameters()), lr=LR)
    opt_all = optim.Adam(list(vae.parameters()) + list(p_ddm.parameters()), lr=LR)
    opt_br  = optim.Adam(list(vae.bridgerDDM.parameters()) + list(vae.bridgerEEG.parameters()), lr=LR)
    def _opt(e):
        if e <= P1_END: return opt_ddm
        if e <= P2_END: return opt_vae
        if e <= P3_END: return opt_all
        return opt_br

    save_dir = MODEL_DIR / f"pooled_{len(subject_ids)}subj_{signal_type}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from newest checkpoint, if any ─────────────────────────────────
    import glob as _glob
    start_epoch = 0
    _resumed_hist = []
    _ckpts = _glob.glob(str(save_dir / "ckpt_ep*.pth"))
    if _ckpts:
        _latest = max(_ckpts, key=os.path.getmtime)
        _ck = torch.load(_latest, map_location=device)
        vae.load_state_dict(_ck["vae"])
        p_ddm.load_state_dict(_ck["p_ddm"])
        # optimizer states (only present in checkpoints saved by this new version)
        for _name, _optzr in [("opt_ddm", opt_ddm), ("opt_vae", opt_vae),
                            ("opt_all", opt_all), ("opt_br", opt_br)]:
            if _ck.get(_name) is not None:
                try:    _optzr.load_state_dict(_ck[_name])
                except Exception: pass
        start_epoch = int(_ck.get("epoch", 0))
        _resumed_hist = _ck.get("hist", [])
        print(f"  \u27f3  Resumed from {os.path.basename(_latest)} \u2014 continuing at epoch {start_epoch+1}")
        if _ck.get("opt_ddm") is None:
            print(f"     (note: old checkpoint had no optimizer state \u2014 Adam momentum reset, minor)")
    else:
        print("  (no checkpoint found \u2014 training from scratch)")

    KEYS = ['total','rec','wfpt','kl_eeg','kl_ddm','corr_v']

    def run_batch(x, rtS, c, y, subj, epoch):
        ctx = make_ctx(vae, c, subj)                       # [cond ‖ subj_emb]
        v_stat, a_stat, n_stat, ch_pred = p_ddm(x, ctx)
        v_mu, v_lv = v_stat[:, :1], v_stat[:, 1:]
        a_mu, a_lv = a_stat[:, :1], a_stat[:, 1:]
        n_mu, n_lv = n_stat[:, :1], n_stat[:, 1:]

        v   = v_mu + torch.randn_like(v_mu) * (0.5*v_lv).exp()
        a   = (a_mu + torch.randn_like(a_mu) * (0.5*a_lv).exp()).clamp(min=0.1)
        ndt = (n_mu + torch.randn_like(n_mu) * (0.5*n_lv).exp()).clamp(
               min=0.05, max=rtS.min().item()*0.95)

        z_ddm = torch.cat([v, a, ndt], dim=1)
        recon, mu_x, lv_x = vae(x, c, z_ddm, subj)
        recon_sq = recon.squeeze(1)

        rec_l = F.mse_loss(recon_sq, x) * REC_X_W
        ch_l  = F.binary_cross_entropy_with_logits(ch_pred.squeeze(), y) * CHOICE_W

        # wiener uses RAW seconds
        rt_signed = rtS * (1 - 2*y.unsqueeze(1))
        try:
            wfpt_l = wiener_loss(rt_signed, v, ndt, a, backprop_ndt=(epoch > NDT_EPOCH))
            if torch.isnan(wfpt_l) or torch.isinf(wfpt_l):
                wfpt_l = torch.zeros(1, requires_grad=True).squeeze()
        except Exception:
            wfpt_l = torch.zeros(1, requires_grad=True).squeeze()

        kl_eeg = kl_standard(mu_x, lv_x) * kl_weight

        # PER-SUBJECT informed prior: gather each trial's own subject prior
        vmu = PR["v_mu"][subj]; vlv = PR["v_lv"][subj]
        amu = PR["a_mu"][subj]; alv = PR["a_lv"][subj]
        nmu = PR["n_mu"][subj]; nlv = PR["n_lv"][subj]
        kl_ddm = (kl_informed(v_mu, v_lv, vmu, vlv) * KL_DDM_W[0] +
                  kl_informed(a_mu, a_lv, amu, alv) * KL_DDM_W[1] +
                  kl_informed(n_mu, n_lv, nmu, nlv) * KL_DDM_W[2])

        if   epoch <= P1_END: total = kl_ddm + ch_l + wfpt_l
        elif epoch <= P2_END: total = rec_l + kl_eeg
        elif epoch <= P3_END: total = rec_l + ch_l + kl_eeg + kl_ddm + wfpt_l
        else:
            b = vae.bridgerDDM(rtS, c); mu_br, lv_br = b[:, :3], b[:, 3:]
            mu_e = torch.cat([v_mu,a_mu,n_mu],1); lv_e = torch.cat([v_lv,a_lv,n_lv],1)
            total = kl_between(mu_e, lv_e, mu_br, lv_br) * KL_Y_WEIGHT

        losses = dict(total=total, rec=rec_l, wfpt=wfpt_l, kl_eeg=kl_eeg, kl_ddm=kl_ddm)
        diag   = dict(corr_v=pearson_r(v_mu, rtS).item())
        return losses, diag

    def epoch_pass(loader, epoch, train):
        (vae.train() if train else vae.eval())
        (p_ddm.train() if train else p_ddm.eval())
        opt = _opt(epoch)
        tot = {k: 0.0 for k in KEYS}
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, rtS, c, y, subj in loader:
                x,rtS,c,y,subj = [t.to(device) for t in (x,rtS,c,y,subj)]
                if train: opt.zero_grad()
                losses, diag = run_batch(x, rtS, c, y, subj, epoch)
                if train:
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(vae.parameters())+list(p_ddm.parameters()), 1.0)
                    opt.step()
                for k,v in losses.items(): tot[k]+=v.item()
                for k,v in diag.items():   tot[k]+=v
        n = max(len(loader),1)
        return {k:v/n for k,v in tot.items()}

    # ── Training loop ─────────────────────────────────────────────────────────
    hist = list(_resumed_hist)
    print(f"{'Ep':>5} {'Ph':>3} | {'Total':>8} {'Rec(tr)':>8} {'Rec(val)':>8} {'corr_v':>8}")
    print("─"*52)
    for epoch in range(start_epoch + 1, N_EPOCHS+1):
        tr = epoch_pass(train_loader, epoch, True)
        hist.append({'epoch':epoch,'split':'train',**tr})
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            vl = epoch_pass(val_loader, epoch, False)
            hist.append({'epoch':epoch,'split':'val',**vl})
            ph = (1 if epoch<=P1_END else 2 if epoch<=P2_END else 3 if epoch<=P3_END else 4)
            print(f"{epoch:5d} P{ph}  | {tr['total']:8.4f} {tr['rec']:8.4f} "
                  f"{vl['rec']:8.4f} {tr['corr_v']:8.4f}")
        if epoch % SAVE_EVERY == 0:
            torch.save({'epoch':epoch,'eeg_chan':eeg_chan,'time_steps':time_steps,
                        'c_dim':c_dim,'n_subjects':n_subjects,
                        'vae':vae.state_dict(),'p_ddm':p_ddm.state_dict(),
                        'subject_ids':subject_ids, 'hist':hist,
                        'opt_ddm':opt_ddm.state_dict(), 'opt_vae':opt_vae.state_dict(),
                        'opt_all':opt_all.state_dict(), 'opt_br':opt_br.state_dict()},
                       save_dir/f"ckpt_ep{epoch:04d}.pth")
            print(f"  \U0001f4be  saved epoch {epoch}")

    # ── loss curves: train vs val reconstruction (the overfitting check) ──────
    hd = pd.DataFrame(hist); trd = hd[hd.split=='train']; vld = hd[hd.split=='val']
    fig, ax = plt.subplots(1, 3, figsize=(14,4))
    for a_,k,t in zip(ax, ['total','rec','corr_v'],
                      ['Total','Reconstruction (train vs val)','v↔RT corr']):
        a_.plot(trd.epoch, trd[k], label='train')
        a_.plot(vld.epoch, vld[k], '--', label='val')
        for xe in (P1_END,P2_END,P3_END): a_.axvline(xe, color='gray', ls=':', alpha=.5)
        a_.set_title(t); a_.set_xlabel('epoch'); a_.legend(fontsize=8)
    fig.suptitle(f"Pooled {len(subject_ids)} subjects | {signal_type}")
    fig.tight_layout(); fig.savefig(save_dir/"loss_curves_pooled.png", dpi=150)
    print(f"\n✅  done → {save_dir}")
    return vae, p_ddm, hd


if __name__ == "__main__":
    # first pooled test: the 3 clean subjects whose _eeg.npy + priors exist
    # all 19 clean subjects (22 total minus bad 03/23/24)
    run_pooled(["01", "02", "04", "05", "06", "07", "10", "11", "12", "13",
                "14", "15", "16", "17", "18", "19", "20", "21", "22"], "stim")