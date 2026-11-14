# ══════════════════════════════════════════════════════════════════════════════
# H3.py — drift perturbation: does perturbing the drift rate change the
# generated waveform?
#
# The drift supplied to the decoder is shifted by ±0.5 (the increment used by
# Vo et al., 2024) and by ±1.0, to test whether the pathway is insensitive to
# the parameter or merely insensitive to small changes in it.
#
# Produces:
#   eval_sensitivity.png            SVD-collapsed waveform, all five drift levels
#   eval_sensitivity_amplitude.png  P300 peak amplitude and latency vs drift shift
#
# Run:  python analysis/H3.py
# ══════════════════════════════════════════════════════════════════════════════

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob
import numpy as np
import torch
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import Z_DIM, TMIN_STIM
from data_loader_pooled import load_pooled_data
from model_pooled       import build_pooled_models
from svd_filter         import compute_svd_filter, svd_project_np

SUBJECTS   = ["01", "02", "04", "05", "06", "07", "10", "11", "12", "13",
              "14", "15", "16", "17", "18", "19", "20", "21", "22"]
SIGNAL     = "stim"
CROP_T     = None                       # must match the value used in training
FS         = 128
TMIN       = TMIN_STIM
P300_WIN   = (200, 600)
LOWPASS_HZ = 10.0
V_DELTAS   = [-1.0, -0.5, 0.0, +0.5, +1.0]
OUT        = os.path.expanduser("~/Desktop/NCVA_results")
CK_DIR     = os.path.join(OUT, "models", f"pooled_{len(SUBJECTS)}subj_{SIGNAL}")


def lowpass(x, fs, hz):
    b, a = butter(4, hz/(fs/2), btype="low")
    return filtfilt(b, a, x, axis=-1)


# ── data + model ──────────────────────────────────────────────────────────────
print(f"\n{'='*62}\n  H3 — sensitivity of the generated waveform to drift\n{'='*62}")

(X, rt_norm, rt_sec, cond, choice, subj_idx,
 eeg_chan, time_steps, c_dim, n_subjects) = load_pooled_data(
        SUBJECTS, SIGNAL, crop_t=CROP_T)

vae, p_ddm, device = build_pooled_models(eeg_chan, time_steps, c_dim, n_subjects, Z_DIM)
ckpt = max(glob.glob(os.path.join(CK_DIR, "ckpt_ep*.pth")), key=os.path.getmtime)
ck = torch.load(ckpt, map_location=device)
vae.load_state_dict(ck["vae"]); p_ddm.load_state_dict(ck["p_ddm"])
vae.eval(); p_ddm.eval()
print(f"  Loaded {os.path.basename(ckpt)} (epoch {ck.get('epoch','?')}, "
      f"time_steps {ck.get('time_steps','?')})")

ct   = torch.tensor(cond, dtype=torch.float32).unsqueeze(1)
rtSt = torch.tensor(rt_sec, dtype=torch.float32)

times_ms = (TMIN + np.arange(time_steps)/FS) * 1000.0
i0 = int(np.argmin(np.abs(times_ms - P300_WIN[0])))
i1 = int(np.argmin(np.abs(times_ms - P300_WIN[1])))

v0_all = compute_svd_filter(X.mean(0))              # one filter for all conditions


def generate(v_delta):
    """Generate EEG from choice-RT with the drift shifted by v_delta."""
    out = []
    with torch.no_grad():
        for i in range(0, len(ct), 32):
            c   = ct[i:i+32].to(device)
            rtb = rtSt[i:i+32].unsqueeze(1).to(device)
            b   = vae.bridgerDDM(rtb, c); mu = b[:, :3]
            z3  = torch.cat([mu[:, :1] + v_delta, mu[:, 1:2], mu[:, 2:3]], 1)
            zN  = vae.bridgerEEG(z3, c)[:, :Z_DIM]
            r   = vae.decoderX(zN, z3, c).squeeze(1).cpu().numpy()
            out.append(svd_project_np(r, v0_all))
    return lowpass(np.concatenate(out, 0), FS, LOWPASS_HZ)


# ── generate at each drift level ──────────────────────────────────────────────
print(f"\n  {'drift shift':>12}{'P300 amp':>11}{'latency (ms)':>15}"
      f"{'Δamp vs 0':>12}")
print("  " + "-"*50)

waves, amps, lats = {}, {}, {}
for d in V_DELTAS:
    w = generate(d)
    mean_w = w.mean(0)
    waves[d] = mean_w
    seg = mean_w[i0:i1]
    k = int(np.argmax(seg))
    amps[d] = float(seg[k])
    lats[d] = float(times_ms[i0 + k])

for d in V_DELTAS:
    print(f"  {d:>+12.1f}{amps[d]:>11.4f}{lats[d]:>15.1f}"
          f"{amps[d]-amps[0.0]:>+12.4f}")

rng_amp = max(amps.values()) - min(amps.values())
rng_lat = max(lats.values()) - min(lats.values())
print("  " + "-"*50)
print(f"  amplitude varies by {rng_amp:.4f} across the full ±1.0 range")
print(f"  latency varies by   {rng_lat:.1f} ms across the full ±1.0 range")

# empirical grand average for reference
data_w = lowpass(svd_project_np(X, v0_all), FS, LOWPASS_HZ).mean(0)

# ── Figure: waveforms at each drift level ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(times_ms, data_w, color="black", lw=2.5, label="empirical", zorder=5)
colours = {-1.0: "#8B1A1A", -0.5: "#D6604D", 0.0: "#404040",
           +0.5: "#4393C3", +1.0: "#1A4A8B"}
for d in V_DELTAS:
    ax.plot(times_ms, waves[d], color=colours[d],
            ls="--" if d == 0.0 else "-", lw=2,
            label="inferred drift" if d == 0.0 else f"drift {d:+.1f}")
ax.axvspan(P300_WIN[0], P300_WIN[1], color="orange", alpha=.08)
ax.axvline(0, color="k", lw=.5)
ax.set_xlim(-200, 1000)
ax.set_xlabel("time (ms)"); ax.set_ylabel("amplitude (SVD-projected)")
ax.set_title("Generated waveform under drift-rate perturbation")
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(CK_DIR, "eval_sensitivity.png"), dpi=150); plt.close(fig)
print(f"\n  saved → eval_sensitivity.png")

# ── Figure: amplitude and latency as a function of the shift ──────────────────
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(V_DELTAS, [amps[d] for d in V_DELTAS], "o-", color="tab:blue")
ax[0].set_xlabel("drift shift"); ax[0].set_ylabel("P300 peak amplitude")
ax[0].set_title("Amplitude")
ax[1].plot(V_DELTAS, [lats[d] for d in V_DELTAS], "o-", color="tab:orange")
ax[1].set_xlabel("drift shift"); ax[1].set_ylabel("P300 peak latency (ms)")
ax[1].set_title("Latency")
for a_ in ax:
    a_.axvline(0, color="grey", ls=":", lw=1)
fig.suptitle("P300 response to drift-rate perturbation", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(CK_DIR, "eval_sensitivity_amplitude.png"), dpi=150)
plt.close(fig)
print(f"  saved → eval_sensitivity_amplitude.png")

print(f"\n✅  H3 complete → {CK_DIR}")