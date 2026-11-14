# ══════════════════════════════════════════════════════════════════════════════
# H1.py — DDM parameters against reaction time, in the format of Vo et al. (2024)
#
#   rows:  (a) fitted from EEG (training)      (b) predicted from EEG (test)
#          (c) fitted from choice-RT (training)(d) predicted from choice-RT (test)
#   cols:  drift (δ)  ·  boundary (α)  ·  non-decision time (ndt)
#
# Spearman rho above each panel; dashed horizontal lines mark group means.
#
# COLOUR_BY controls the grouping:
#   "participant" — one colour per participant. Shows the participant-specific
#                   bands that make the pooled correlation an artefact of
#                   between-participant offsets. This is the finding.
#   "condition"   — easy vs hard, matching the original figure's use of noise
#                   level. Closer to Vo et al., but hides the band structure.
#
# Run:  python analysis/H1.py
# ══════════════════════════════════════════════════════════════════════════════

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob
import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import Z_DIM
from data_loader_pooled import load_pooled_data
from model_pooled       import build_pooled_models, make_ctx

SUBJECTS   = ["01", "02", "04", "05", "06", "07", "10", "11", "12", "13",
              "14", "15", "16", "17", "18", "19", "20", "21", "22"]
SIGNAL     = "stim"
CROP_T     = None              # must match the value used in training
COLOUR_BY  = "participant"     # "participant" or "condition"
OUT        = os.path.expanduser("~/Desktop/NCVA_results")
CK_DIR     = os.path.join(OUT, "models", f"pooled_{len(SUBJECTS)}subj_{SIGNAL}")


def split_indices(n, subject_idx, val_frac=0.2, seed=42):
    """Reproduce the stratified split used in training."""
    rng = np.random.default_rng(seed)
    tr = np.zeros(n, dtype=bool)
    for s in np.unique(subject_idx):
        idx = np.where(subject_idx == s)[0]
        rng.shuffle(idx)
        n_val = max(1, int(val_frac * len(idx)))
        tr[idx[n_val:]] = True
    return tr, ~tr


# ── data and model ────────────────────────────────────────────────────────────
(X, rt_norm, rt_sec, cond, choice, subj_idx,
 eeg_chan, time_steps, c_dim, n_subjects) = load_pooled_data(
        SUBJECTS, SIGNAL, crop_t=CROP_T)

vae, p_ddm, device = build_pooled_models(eeg_chan, time_steps, c_dim,
                                         n_subjects, Z_DIM)
ckpt = max(glob.glob(os.path.join(CK_DIR, "ckpt_ep*.pth")), key=os.path.getmtime)
ck = torch.load(ckpt, map_location=device)
vae.load_state_dict(ck["vae"]); p_ddm.load_state_dict(ck["p_ddm"])
vae.eval(); p_ddm.eval()
print(f"Loaded {os.path.basename(ckpt)} (epoch {ck.get('epoch','?')}, "
      f"checkpoint time_steps {ck.get('time_steps','?')}, data {time_steps})")

Xt   = torch.tensor(X, dtype=torch.float32)
ct   = torch.tensor(cond, dtype=torch.float32).unsqueeze(1)
st   = torch.tensor(subj_idx, dtype=torch.long)
rtSt = torch.tensor(rt_sec, dtype=torch.float32)
tr_mask, va_mask = split_indices(len(X), subj_idx)

# ── all three parameters, both pathways ───────────────────────────────────────
Ve, Ae, Ne, Vr, Ar, Nr = [], [], [], [], [], []
with torch.no_grad():
    for i in range(0, len(Xt), 32):
        x   = Xt[i:i+32].to(device)
        c   = ct[i:i+32].to(device)
        sj  = st[i:i+32].to(device)
        rtb = rtSt[i:i+32].unsqueeze(1).to(device)

        ctx = make_ctx(vae, c, sj)
        v_s, a_s, n_s, _ = p_ddm(x, ctx)
        Ve.append(v_s[:, 0].cpu().numpy())
        Ae.append(a_s[:, 0].cpu().numpy())
        Ne.append(n_s[:, 0].cpu().numpy())

        b = vae.bridgerDDM(rtb, c)            # (B, 6): three means, three logvars
        Vr.append(b[:, 0].cpu().numpy())
        Ar.append(b[:, 1].cpu().numpy())
        Nr.append(b[:, 2].cpu().numpy())

P = {
    "eeg": {"v": np.concatenate(Ve), "a": np.concatenate(Ae), "ndt": np.concatenate(Ne)},
    "rt":  {"v": np.concatenate(Vr), "a": np.concatenate(Ar), "ndt": np.concatenate(Nr)},
}
rt = rt_sec

rows = [
    ("(a) Fitted from EEG (training data)",        "eeg", tr_mask),
    ("(b) Predicted from EEG (test data)",         "eeg", va_mask),
    ("(c) Fitted from choice-RTs (training data)", "rt",  tr_mask),
    ("(d) Predicted from choice-RTs (test data)",  "rt",  va_mask),
]
cols = [("v", "δ"), ("a", "α"), ("ndt", "ndt (s)")]


def rho(vals, mask):
    if mask.sum() < 5:
        return np.nan
    return stats.spearmanr(np.abs(rt[mask]), vals[mask])[0]


# ── grouping for colour ───────────────────────────────────────────────────────
if COLOUR_BY == "condition":
    grp_idx     = np.argmax(cond, axis=1)                # 0 = easy, 1 = hard
    grp_names   = ["Easy", "Hard"]
    grp_colours = ["tab:green", "tab:red"]
else:
    grp_idx     = subj_idx
    grp_names   = [f"subj {s}" for s in SUBJECTS]
    grp_colours = plt.cm.tab20(np.linspace(0, 1, len(SUBJECTS)))

n_grp = len(grp_names)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(13.5, 17.5))

# Layout is fixed up front so that everything placed relative to the axes
# afterwards (row captions, legend) lands in the right place.
fig.subplots_adjust(top=0.965, bottom=0.09, left=0.07, right=0.98,
                    hspace=0.52, wspace=0.30)

for r, (row_label, pathway, mask) in enumerate(rows):
    for c_, (pkey, plabel) in enumerate(cols):
        ax   = axes[r, c_]
        vals = P[pathway][pkey]

        for g in range(n_grp):
            m = (grp_idx == g) & mask
            if not m.any():
                continue
            colour = grp_colours[g]
            ax.scatter(np.abs(rt[m]), vals[m], s=7, alpha=.45, linewidths=0,
                       color=colour,
                       label=grp_names[g] if (r == 0 and c_ == 1) else None)
            if COLOUR_BY == "condition":
                # group means, as in the original figure — legible only with
                # a handful of groups, so omitted when colouring by participant
                ax.axhline(vals[m].mean(), color=colour, ls="--", lw=1.0, alpha=.8)

        ax.set_title(f"ρ: {rho(vals, mask):.2f}", fontsize=12)
        ax.set_xlabel("RT (s)", fontsize=10)
        ax.set_ylabel(plabel, fontsize=13, style="italic")
        ax.tick_params(labelsize=9)

    # row caption: anchored to the bottom of the middle panel and offset in
    # points, so it sits just below the x-axis label regardless of layout.
    # With three equal-width columns the middle panel's centre is the row's.
    axes[r, 1].annotate(row_label, xy=(0.5, 0), xycoords="axes fraction",
                        xytext=(0, -38), textcoords="offset points",
                        ha="center", va="top", fontsize=11,
                        annotation_clip=False)

# legend from the top-middle panel
handles, labels = axes[0, 1].get_legend_handles_labels()
if COLOUR_BY == "condition":
    axes[0, 1].legend(handles, labels, fontsize=9, frameon=True, markerscale=2.2)
else:
    fig.legend(handles, labels, loc="lower center", ncol=10, fontsize=8.5,
               frameon=False, markerscale=2.2, columnspacing=0.9,
               handletextpad=0.3, bbox_to_anchor=(0.5, 0.004))

out = os.path.join(CK_DIR, "h1_grid_pooled.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

# ── per-participant table, all three parameters ───────────────────────────────
print(f"\n{'='*100}")
print("  Within-participant Spearman correlations with reaction time")
print(f"{'='*100}")
hdr = f"  {'subj':<6}"
for tag in ["(a)", "(b)", "(c)", "(d)"]:
    for pkey, _ in cols:
        hdr += f"{tag + '-' + pkey:>12}"
print(hdr)
print("  " + "-" * 96)
for si, s in enumerate(SUBJECTS):
    line = f"  {s:<6}"
    for _, pathway, mask in rows:
        for pkey, _ in cols:
            line += f"{rho(P[pathway][pkey], (subj_idx == si) & mask):>+12.3f}"
    print(line)

print("  " + "-" * 96)
for label, fn in [("median", np.nanmedian), ("pooled", None)]:
    line = f"  {label:<6}"
    for _, pathway, mask in rows:
        for pkey, _ in cols:
            if fn is None:
                line += f"{rho(P[pathway][pkey], mask):>+12.3f}"
            else:
                vals = [rho(P[pathway][pkey], (subj_idx == si) & mask)
                        for si in range(len(SUBJECTS))]
                line += f"{fn(vals):>+12.3f}"
    print(line)

# ── direction consistency on held-out trials, against H1's predictions ────────
print(f"\n  H1 predictions, held-out trials (from EEG):")
for pkey, plabel in cols:
    vals = np.array([rho(P["eeg"][pkey], (subj_idx == si) & va_mask)
                     for si in range(len(SUBJECTS))])
    if pkey == "v":
        expected, n_ok = "negative", int((vals < 0).sum())
    else:
        expected, n_ok = "positive", int((vals > 0).sum())
    print(f"    {plabel:<8} predicted {expected:>8}:  {n_ok:>2} / {len(vals)} participants"
          f"   (median {np.nanmedian(vals):+.3f})")

# ── participant level: mean EEG-inferred drift against mean reaction time ─────
md = [P["eeg"]["v"][subj_idx == si].mean() for si in range(len(SUBJECTS))]
mr = [rt_sec[subj_idx == si].mean() for si in range(len(SUBJECTS))]
rho_p, p_p = stats.spearmanr(md, mr)
print(f"\n  Participant level (n = {len(SUBJECTS)}, all trials): "
      f"Spearman rho(mean drift, mean RT) = {rho_p:.2f}, p = {p_p:.3f}")
