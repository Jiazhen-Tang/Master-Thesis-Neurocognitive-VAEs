# ══════════════════════════════════════════════════════════════════════════════
# check_alignment.py — is the RT stored in beh.pkl the RT of the same trial as
# the EEG epoch it is paired with?
#
# Uses only the local tensors (numpy only; no network drive, no torch).
#
# Idea: the stimulus-locked epoch (-0.2 .. 4.5 s around stimulus onset) and the
# response-locked epoch (-4.5 .. 0.2 s around trial end) of one trial are cut
# from the same continuous recording, so they contain an identical stretch of
# EEG, displaced by the duration of that trial.
#
#   step 1  For every stimulus-locked epoch, find the displacement at which it
#           coincides with a response-locked epoch. This gives the duration of
#           the trial from the EEG alone, without using beh.pkl.
#   step 2  Compare that duration with the RT stored in the same row of
#           beh.pkl and in the neighbouring rows. (The trial_end marker follows
#           the response by a constant delay, estimated per participant.)
#
#   row offset  0 -> the EEG epoch carries its own RT            (aligned)
#   row offset +1 -> its RT is stored one row further down       (shifted)
#
# Run:  python check_alignment.py            (all 19 participants, ~5 minutes)
#       python check_alignment.py 01 07      (selected participants)
# ══════════════════════════════════════════════════════════════════════════════
import os, sys, pickle
import numpy as np

TENSOR_DIR = os.path.expanduser(os.environ.get("NCVA_TENSORS", "~/Desktop/NCVA_results/tensors"))
SUBJECTS = ["01","02","04","05","06","07","10","11","12","13",
            "14","15","16","17","18","19","20","21","22"]
FS      = 128.0
EPOCH   = 4.3            # resp epoch starts 4.5 s before trial end, stim epoch 0.2 s before onset
CH      = [10, 30, 47]   # three channels are enough
STEP    = 4              # coarse search on every 4th sample, refined at full rate
MIN_OV  = 0.4            # s of overlap required (trials longer than ~8.4 s cannot be tested)
R_MIN   = 0.9            # the two epochs must correlate above this to count as found
TOL     = 0.03           # s, tolerance when comparing with the stored RT
OFFSETS = range(-3, 4)


def corr(a, b, k):
    """correlation of a[:, i] with b[:, i + k] over the overlapping samples"""
    L = a.shape[1]
    x, y = (a[:, :L-k], b[:, k:]) if k >= 0 else (a[:, -k:], b[:, :L+k])
    x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
    return (x*y).sum() / (np.sqrt((x*x).sum() * (y*y).sum()) + 1e-20)


def duration(a, b):
    """(r, trial duration in s) at the displacement where a and b coincide best"""
    ac, bc = a[:, ::STEP], b[:, ::STEP]
    Lc = ac.shape[1]; mo = int(MIN_OV * FS / STEP)
    best = (-2.0, 0)
    for k in range(-(Lc - mo), Lc - mo + 1):
        r = corr(ac, bc, k)
        if r > best[0]: best = (r, k)
    best_f = (-2.0, 0)
    for k in range(best[1]*STEP - STEP, best[1]*STEP + STEP + 1):   # refine at full rate
        if abs(k) > a.shape[1] - int(MIN_OV * FS): continue
        r = corr(a, b, k)
        if r > best_f[0]: best_f = (r, k)
    return best_f[0], EPOCH - best_f[1] / FS


def check(subj):
    d  = os.path.join(TENSOR_DIR, f"subj_{subj}")
    Xs = np.load(os.path.join(d, f"subj_{subj}_X_stim_eeg.npy"), mmap_mode="r")
    Xr = np.asarray(np.load(os.path.join(d, f"subj_{subj}_X_resp_eeg.npy"), mmap_mode="r")[:, CH, :], float)
    with open(os.path.join(d, f"subj_{subj}_beh.pkl"), "rb") as f:
        rt = np.asarray(pickle.load(f)["rt"], float)
    n = len(Xs); dur = np.full(n, np.nan); shift = 0
    for i in range(n):
        a = np.asarray(Xs[i, CH, :], float); best = (-2.0, np.nan, 0)
        # stim and resp epochs were rejected separately, so the matching resp
        # epoch may sit a few rows away; follow that drift as it accumulates
        for width in (3, 20):                     # widen the search if nothing is found nearby
            for j in range(max(0, i + shift - width), min(len(Xr), i + shift + width + 1)):
                if width == 20 and abs(j - i - shift) <= 3: continue
                r, t = duration(a, Xr[j])
                if r > best[0]: best = (r, t, j - i)
            if best[0] > R_MIN: break
        if best[0] > R_MIN:
            dur[i] = best[1]; shift = best[2]
    # delay between response and trial_end marker: the most common
    # difference between EEG duration and any nearby stored RT
    diffs = np.concatenate([dur[max(0, -o):n - max(0, o)] - rt[max(0, o):n - max(0, -o)] for o in OFFSETS])
    diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) < 0.5)]
    h, e = np.histogram(diffs, bins=np.arange(-0.5, 0.5, 0.01))
    delay = float(e[np.argmax(h)] + 0.005)
    # the delay is constant within a session but can differ between sessions,
    # so every clearly populated value is accepted
    delays = [float(e[b] + 0.005) for b in np.where(h >= max(8, 0.03 * n))[0]] or [delay]
    label = np.full(n, 99)                        # 99 = cannot be tested / no row fits
    for i in range(n):
        if not np.isfinite(dur[i]): continue
        for o in sorted(OFFSETS, key=abs):        # the trial's own row is tried first
            if 0 <= i + o < n and min(abs(dur[i] - dl - rt[i + o]) for dl in delays) < TOL:
                label[i] = o; break
        else:
            label[i] = 98                         # EEG duration found, but no nearby row has it
    return rt, dur - delay, label, delay


subjects = sys.argv[1:] or SUBJECTS
print(f"{'subj':<6}{'trials':>7}{'aligned':>9}{'shifted':>9}{'no match':>10}{'untestable':>12}{'% shifted':>11}{'marker delay':>14}")
print("-" * 78)
tot = np.zeros(4, int); keep = {}
for s in subjects:
    rt, dur, lab, delay = check(s); keep[s] = (rt, dur, lab)
    a = int((lab == 0).sum()); u = int((lab == 99).sum()); nm = int((lab == 98).sum()); sh = len(lab) - a - u - nm
    tot += (a, sh, nm, u)
    print(f"{s:<6}{len(lab):>7}{a:>9}{sh:>9}{nm:>10}{u:>12}{100*sh/len(lab):>10.0f}%{delay:>13.2f}s", flush=True)
print("-" * 78)
print(f"{'all':<6}{tot.sum():>7}{tot[0]:>9}{tot[1]:>9}{tot[2]:>10}{tot[3]:>12}{100*tot[1]/tot.sum():>10.0f}%")
print("\naligned    = the EEG epoch's duration equals the RT stored in its own row"
      "\nshifted    = it equals the RT stored in a neighbouring row instead"
      "\nno match   = duration measured from the EEG, but none of the 7 nearby rows has it"
      "\nuntestable = RT too long for the two epochs to overlap, or a noisy trial")

# show a stretch of the most affected participant
worst = max(keep, key=lambda s: ((keep[s][2] != 0) & (keep[s][2] < 98)).mean())
rt, dur, lab = keep[worst]
first = max(0, int(np.argmax((lab != 0) & (lab < 98))) - 3)
print(f"\nParticipant {worst}, 15 trials from EEG trial {first}:")
print(f"  {'EEG trial':>9}{'RT stored in this row':>23}{'RT measured from EEG':>22}{'found in row':>14}")
for i in range(first, min(first + 15, len(rt))):
    m = "-" if not np.isfinite(dur[i]) else f"{dur[i]:.2f}"
    w = {99: "untestable", 98: "none nearby", 0: "this row"}.get(int(lab[i]), f"{int(lab[i]):+d}")
    print(f"  {i:>9}{rt[i]:>23.2f}{m:>22}{w:>14}")
