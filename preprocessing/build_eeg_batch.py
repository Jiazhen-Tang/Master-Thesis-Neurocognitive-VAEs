# ══════════════════════════════════════════════════════════════════════════════
# build_eeg_batch.py — batch-generate 64ch _eeg.npy from 69ch raw .npy (LOCAL)
#
# Drops the 5 non-brain channels (Status + 4 EOG) from the existing raw tensors.
# No network drive needed. Bad channels were already interpolated upstream when
# the raw .npy was created, so this only removes channels.
#
# Self-validating: it derives WHICH channels to drop by comparing each subject's
# ch_names against subject 01 (which already has a correct 64ch _eeg.npy).
#
# Run:  python preprocessing/build_eeg_batch.py
# ══════════════════════════════════════════════════════════════════════════════

import os, numpy as np

TENSOR_DIR = os.path.expanduser("~/Desktop/NCVA_results/tensors")
REF_SUBJ   = "01"                      # has a known-correct 64ch _eeg.npy
# All clean subjects (22 total minus bad 03/23/24). 01/02/04 already have _eeg;
# harmless to regenerate — the script overwrites consistently.
SUBJECTS   = ["01","02","04","05","06","07","10","11","12","13","14",
              "15","16","17","18","19","20","21","22"]
SIGNALS    = ["stim", "resp"]

# Non-brain channels to drop (BioSemi: Status + 4 EOG). Matched case-insensitively
# against ch_names; anything containing these tokens is dropped.
DROP_TOKENS = ["status", "eog", "exg", "heog", "veog"]


def read_ch_names(subj):
    p = os.path.join(TENSOR_DIR, f"subj_{subj}", f"subj_{subj}_ch_names.txt")
    return [l.strip() for l in open(p) if l.strip()]


def derive_keep_mask(ch_names):
    """Return boolean mask of channels to KEEP (drop the non-brain ones)."""
    keep = []
    for n in ch_names:
        low = n.lower()
        is_bad = any(tok in low for tok in DROP_TOKENS)
        keep.append(not is_bad)
    return np.array(keep)


# ── Validate the drop logic against subject 01 first ──────────────────────────
ref_names = read_ch_names(REF_SUBJ)
ref_keep  = derive_keep_mask(ref_names)
ref_eeg   = np.load(os.path.join(TENSOR_DIR, f"subj_{REF_SUBJ}",
                                 f"subj_{REF_SUBJ}_X_stim_eeg.npy"), mmap_mode="r")

print(f"Reference subj {REF_SUBJ}: {len(ref_names)} ch → keep {ref_keep.sum()}")
print(f"  dropping: {[n for n,k in zip(ref_names, ref_keep) if not k]}")
print(f"  existing _eeg.npy has {ref_eeg.shape[1]} channels")

if ref_keep.sum() != ref_eeg.shape[1]:
    print(f"\n⚠️  MISMATCH: drop logic keeps {ref_keep.sum()} but subj 01's _eeg "
          f"has {ref_eeg.shape[1]}. The DROP_TOKENS are wrong — fix before running.")
    raise SystemExit(1)
print(f"  ✅ drop logic matches subj 01's 64ch — safe to apply\n")

# ── Generate _eeg.npy for each subject ────────────────────────────────────────
for s in SUBJECTS:
    if s == REF_SUBJ:
        print(f"subj {s}: reference subject, already has _eeg.npy — skip")
        continue
    names = read_ch_names(s)
    keep  = derive_keep_mask(names)
    if keep.sum() != ref_eeg.shape[1]:
        print(f"subj {s}: ⚠️  would keep {keep.sum()} ch (expected {ref_eeg.shape[1]}) "
              f"— channel names differ, SKIPPING")
        continue
    for sig in SIGNALS:
        src = os.path.join(TENSOR_DIR, f"subj_{s}", f"subj_{s}_X_{sig}.npy")
        dst = os.path.join(TENSOR_DIR, f"subj_{s}", f"subj_{s}_X_{sig}_eeg.npy")
        if not os.path.exists(src):
            print(f"subj {s} {sig}: raw missing, skip"); continue
        X = np.load(src)                       # (N, 69, T)
        Xeeg = X[:, keep, :].astype("float32") # (N, 64, T)
        np.save(dst, Xeeg)
        print(f"subj {s} {sig}: {X.shape} → {Xeeg.shape}  saved")
    # also write the 64ch channel names for eval scripts
    eeg_names = [n for n, k in zip(names, keep) if k]
    with open(os.path.join(TENSOR_DIR, f"subj_{s}",
                           f"subj_{s}_eeg_channels.txt"), "w") as f:
        f.write("\n".join(eeg_names))

print("\n✅  batch _eeg.npy generation complete")