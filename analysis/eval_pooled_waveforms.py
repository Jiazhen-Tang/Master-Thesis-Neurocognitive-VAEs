# ══════════════════════════════════════════════════════════════════════════════
# eval_pooled_waveforms.py — paper-style waveform figures for the POOLED model
#
# For EACH subject, and for BOTH generation pathways:
#   • ERP butterfly   (all channels overlaid, original vs generated)
#   • per-electrode   (F7,F3,Fz,...,O2 panels, original blue vs generated orange)
#
# Pathways:
#   from-EEG : encode real EEG → decode           (reconstruction, like Fig.2)
#   from-RT  : bridgerDDM(RT) → bridgerEEG → decode (generation, like Fig.3)
#
# Files per subject:  wf_butterfly_{eeg,rt}_subjXX.png , wf_electrodes_{eeg,rt}_subjXX.png
#
# Run:  python analysis/eval_pooled_waveforms.py
# ══════════════════════════════════════════════════════════════════════════════

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import Z_DIM, TMIN_STIM, TENSOR_DIR
from data_loader_pooled import load_pooled_data
from model_pooled       import build_pooled_models, make_ctx

SUBJECTS = ["01", "02", "04", "05", "06", "07", "10", "11", "12", "13",
            "14", "15", "16", "17", "18", "19", "20", "21", "22"]
SIGNAL   = "stim"
CROP_T   = None          # must match the value used in training
FS       = 128
TMIN     = TMIN_STIM
XLIM     = (-200, 1000)
OUT      = os.path.expanduser("~/Desktop/NCVA_results")
CK_DIR   = os.path.join(OUT, "models", f"pooled_{len(SUBJECTS)}subj_{SIGNAL}")

# Centro-parietal montage: the sites over which the P300 is expressed
# (Section 2.3). Occipital sites are excluded; Cz is included as the midline
# central reference point of the Cz–CPz–Pz chain.
PANEL_LAYOUT = {
              "C3": (0,0), "Cz": (0,1), "C4": (0,2),
                           "CPz": (1,1),
              "P3": (2,0), "Pz": (2,1), "P4": (2,2),
              "P7": (3,0),              "P8": (3,2),
}
CP_PICKS = list(PANEL_LAYOUT.keys())


def load_ch_names(subj_id, n):
    d = os.path.join(TENSOR_DIR, f"subj_{subj_id}")
    for f in (f"subj_{subj_id}_eeg_channels.txt", f"subj_{subj_id}_ch_names.txt"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            names = [l.strip() for l in open(p) if l.strip()]
            if len(names) >= n: return names[:n]
    return [f"ch{i}" for i in range(n)]


# ── load data + model ─────────────────────────────────────────────────────────
(X, rt_norm, rt_sec, cond, choice, subj_idx,
 eeg_chan, time_steps, c_dim, n_subjects) = load_pooled_data(SUBJECTS, SIGNAL, crop_t=CROP_T)
vae, p_ddm, device = build_pooled_models(eeg_chan, time_steps, c_dim, n_subjects, Z_DIM)
ckpt = max(glob.glob(os.path.join(CK_DIR, "ckpt_ep*.pth")), key=os.path.getmtime)
ck = torch.load(ckpt, map_location=device)
vae.load_state_dict(ck["vae"]); p_ddm.load_state_dict(ck["p_ddm"])
vae.eval(); p_ddm.eval()
print(f"Loaded {os.path.basename(ckpt)}")

Xt   = torch.tensor(X, dtype=torch.float32)
ct   = torch.tensor(cond, dtype=torch.float32).unsqueeze(1)
st   = torch.tensor(subj_idx, dtype=torch.long)
rtSt = torch.tensor(rt_sec, dtype=torch.float32)
times_ms = (TMIN + np.arange(time_steps)/FS) * 1000.0


def recon_from_eeg(mask):
    idx = torch.as_tensor(np.where(mask)[0], dtype=torch.long); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            j = idx[i:i+16]
            x = Xt[j].to(device); c = ct[j].to(device); sj = st[j].to(device)
            ctx = make_ctx(vae, c, sj)
            v_s, a_s, n_s, _ = p_ddm(x, ctx)
            z = torch.cat([v_s[:, :1], a_s[:, :1], n_s[:, :1]], 1)
            r, *_ = vae(x, c, z, sj)
            out.append(r.squeeze(1).cpu().numpy())
    return np.concatenate(out)


def recon_from_rt(mask):
    idx = torch.as_tensor(np.where(mask)[0], dtype=torch.long); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            j = idx[i:i+16]
            c = ct[j].to(device); rtb = rtSt[j].unsqueeze(1).to(device)
            b = vae.bridgerDDM(rtb, c); mu = b[:, :3]
            zN = vae.bridgerEEG(mu, c)[:, :Z_DIM]
            r = vae.decoderX(zN, mu, c).squeeze(1).cpu().numpy()
            out.append(r)
    return np.concatenate(out)


def butterfly(orig, gen, subj, tag, cp_idx):
    """Overlay the centro-parietal channels, original beside generated."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for a_, d, t in zip(ax, [orig, gen], ["Original", "Generated"]):
        for ci in cp_idx:
            a_.plot(times_ms, d[ci], lw=.9, alpha=.85)
        a_.axvline(0, color="k", lw=.5); a_.axhline(0, color="k", lw=.4)
        a_.set_xlim(*XLIM); a_.set_title(t); a_.set_xlabel("Time (ms)")
    ax[0].set_ylabel("Normalised amplitude")
    fig.suptitle(f"subj {subj} — centro-parietal channels ({tag})")
    fig.tight_layout()
    fig.savefig(os.path.join(CK_DIR, f"wf_butterfly_{tag}_subj{subj}.png"), dpi=140)
    plt.close(fig)


def electrodes(orig, gen, subj, tag, ch_names):
    n2i = {n: i for i, n in enumerate(ch_names)}
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(4, 3, figure=fig, hspace=.45, wspace=.25)
    ax = None
    for elec, (r, c) in PANEL_LAYOUT.items():
        if elec not in n2i:
            continue
        i = n2i[elec]
        ax = fig.add_subplot(gs[r, c])
        rr = np.corrcoef(orig[i], gen[i])[0, 1]
        ax.plot(times_ms, orig[i], color="tab:blue", lw=1.3, label="Original")
        ax.plot(times_ms, gen[i],  color="tab:orange", lw=1.3, label="Generated")
        ax.axvline(0, color="k", lw=.5); ax.axhline(0, color="k", lw=.4)
        ax.set_xlim(*XLIM); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{elec}   r = {rr:+.2f}", fontsize=10)
    if ax is not None:
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(f"subj {subj} — centro-parietal reconstruction ({tag})", fontsize=13)
    fig.savefig(os.path.join(CK_DIR, f"wf_electrodes_{tag}_subj{subj}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


RESULTS = {"eeg": [], "rt": []}

for si, s_id in enumerate(SUBJECTS):
    m = subj_idx == si
    ch_names = load_ch_names(s_id, eeg_chan)
    n2i = {n: i for i, n in enumerate(ch_names)}
    cp_idx = [n2i[e] for e in CP_PICKS if e in n2i]
    missing = [e for e in CP_PICKS if e not in n2i]

    orig = X[m].mean(0)                                   # (C,T) trial-avg original
    print(f"\nsubj {s_id}: {int(m.sum())} trials"
          + (f"   [missing: {missing}]" if missing else ""))

    for tag, fn in [("eeg", recon_from_eeg), ("rt", recon_from_rt)]:
        gen = fn(m).mean(0)                               # (C,T) trial-avg generated
        butterfly(orig, gen, s_id, tag, cp_idx)
        electrodes(orig, gen, s_id, tag, ch_names)

        rs_all = np.array([np.corrcoef(orig[i], gen[i])[0, 1]
                           for i in range(orig.shape[0])])
        rs_cp  = rs_all[cp_idx]
        RESULTS[tag].append(dict(subj=s_id,
                                 cp_mean=float(np.nanmean(rs_cp)),
                                 cp_min=float(np.nanmin(rs_cp)),
                                 cp_max=float(np.nanmax(rs_cp)),
                                 all_mean=float(np.nanmean(rs_all)),
                                 rs_all=rs_all, rs_cp=rs_cp))
        print(f"  {tag}:  centro-parietal r = {np.nanmean(rs_cp):+.3f} "
              f"({np.nanmin(rs_cp):+.2f} to {np.nanmax(rs_cp):+.2f})"
              f"    all 64 channels r = {np.nanmean(rs_all):+.3f}")

# ── summary across subjects ───────────────────────────────────────────────────
print(f"\n{'='*76}")
print("  Reconstruction fidelity — Pearson r between original and generated ERP")
print(f"{'='*76}")
print(f"  {'':6}{'from-EEG':>22}      {'from-RT':>22}")
print(f"  {'subj':<6}{'centro-par.':>12}{'all ch.':>10}      "
      f"{'centro-par.':>12}{'all ch.':>10}")
print("  " + "-" * 62)
for a, b in zip(RESULTS["eeg"], RESULTS["rt"]):
    print(f"  {a['subj']:<6}{a['cp_mean']:>+12.3f}{a['all_mean']:>+10.3f}      "
          f"{b['cp_mean']:>+12.3f}{b['all_mean']:>+10.3f}")

def summarise(key, field):
    v = np.array([d[field] for d in RESULTS[key]])
    return np.median(v), v.min(), v.max()

print("  " + "-" * 62)
for key in ("eeg", "rt"):
    med_cp, lo_cp, hi_cp = summarise(key, "cp_mean")
    med_all, lo_all, hi_all = summarise(key, "all_mean")
    label = "from-EEG" if key == "eeg" else "from-RT"
    print(f"  {label:<10} centro-parietal: median {med_cp:+.3f} "
          f"(range {lo_cp:+.2f} to {hi_cp:+.2f})   |   "
          f"all channels: median {med_all:+.3f} "
          f"(range {lo_all:+.2f} to {hi_all:+.2f})")

# subject at the median of the from-EEG centro-parietal fidelity → main-text figure
cp_means = np.array([d["cp_mean"] for d in RESULTS["eeg"]])
med_subj = RESULTS["eeg"][int(np.argsort(cp_means)[len(cp_means)//2])]["subj"]
print(f"\n  Median subject (from-EEG, centro-parietal): subj {med_subj}")
print(f"  → main text: wf_electrodes_eeg_subj{med_subj}.png "
      f"and wf_electrodes_rt_subj{med_subj}.png")

np.savez(os.path.join(CK_DIR, "waveform_fidelity.npz"),
         subjects=np.array(SUBJECTS), picks=np.array(CP_PICKS),
         r_eeg_all=np.array([d["rs_all"] for d in RESULTS["eeg"]]),
         r_rt_all=np.array([d["rs_all"] for d in RESULTS["rt"]]),
         r_eeg_cp=np.array([d["rs_cp"] for d in RESULTS["eeg"]]),
         r_rt_cp=np.array([d["rs_cp"] for d in RESULTS["rt"]]))
print(f"  per-channel values saved to waveform_fidelity.npz")

print(f"\n✅  all waveform figures saved to {CK_DIR}")