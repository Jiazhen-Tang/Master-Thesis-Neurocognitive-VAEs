# ══════════════════════════════════════════════════════════════════════════════
# preprocessing/save_ch_names.py — Save EEG channel names for each subject (one-time)
#
# The .npy tensors don't store channel names. SVD needs them for sign-correction.
# This re-reads the raw EEG header (fast, no epoching) to extract channel order.
#
# Run ONCE:  python preprocessing/save_ch_names.py
# ══════════════════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
from config import *
from pipeline_utils import read_eeg_data

SUBJECT_IDS_ALL = [
    "01","02","03","04","05","06","07","10","11","12",
    "13","14","15","16","17","18","19","20","21","22",
    "23","24"
]

print("Extracting channel names...\n")

# All subjects share the same montage, so we can get names from the first available
for subj_id in SUBJECT_IDS_ALL:
    subj_dir = TENSOR_DIR / f"subj_{subj_id}"
    if not subj_dir.exists():
        continue

    out_path = subj_dir / f"subj_{subj_id}_ch_names.txt"
    if out_path.exists():
        print(f"  Subject {subj_id}: already done")
        continue

    try:
        eeg_d = read_eeg_data(DATA_PATH, subj_id)
        first_session = list(eeg_d.values())[0]
        ch_names = first_session['raw'].ch_names

        # Filter to EEG channels only (exclude stim/EOG if present)
        # Match the count in the saved tensor
        x_path = subj_dir / f"subj_{subj_id}_X_stim.npy"
        if x_path.exists():
            n_chan = np.load(x_path).shape[1]
            ch_names = ch_names[:n_chan]

        with open(out_path, "w") as f:
            f.write("\n".join(ch_names))

        print(f"  Subject {subj_id}: {len(ch_names)} channels saved")
        print(f"    First few: {ch_names[:6]}")

    except Exception as e:
        print(f"  Subject {subj_id}: ❌ {e}")

print("\n✅  Channel names saved")
