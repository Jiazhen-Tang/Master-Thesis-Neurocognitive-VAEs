# ══════════════════════════════════════════════════════════════════════════════
# preprocessing/01_erp_baseline.py — Multi-subject ERP baseline (Cells A-E)
#
# Run this ONCE to compute all ERP baselines. Results saved to SAVE_DIR.
# Resume-safe: already-completed subjects are skipped.
# ══════════════════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os, gc, pickle
import numpy as np
import mne
import pandas as pd
from scipy import signal as sp_signal
import matplotlib.pyplot as plt

from config import *
from pipeline_utils import (read_eeg_data, add_trial_index,
                             read_behavioral_data, merge_eeg_beh,
                             extract_events, epoch_data_both,
                             attach_metadata)

all_evoked_stim     = {}
all_evoked_resp     = {}
all_evoked_easy     = {}
all_evoked_hard     = {}
all_evoked_cpp_easy = {}
all_evoked_cpp_hard = {}
all_psd             = {}
failed_subjects     = []

# ── Recovery: load already-completed subjects ──────────────────────────────────
for subj_id in SUBJECT_IDS:
    path = SAVE_DIR / f"subj_{subj_id}_results.pkl"
    if path.exists():
        with open(path, "rb") as f:
            res = pickle.load(f)
        all_evoked_stim[subj_id] = res["evoked_stim"]
        all_evoked_resp[subj_id] = res["evoked_resp"]
        if res.get("evoked_easy") is not None:
            all_evoked_easy[subj_id]     = res["evoked_easy"]
            all_evoked_hard[subj_id]     = res["evoked_hard"]
            all_evoked_cpp_easy[subj_id] = res["evoked_cpp_easy"]
            all_evoked_cpp_hard[subj_id] = res["evoked_cpp_hard"]
        all_psd[subj_id] = res["psd"]
        print(f"✅  Subject {subj_id} loaded from disk")

REMAINING = [s for s in SUBJECT_IDS if s not in all_evoked_stim]
print(f"\n📋  Remaining: {REMAINING}\n")

# ── Main loop ──────────────────────────────────────────────────────────────────
for subj_id in REMAINING:
    print(f"\n{'─'*50}\n  Subject {subj_id}\n{'─'*50}")
    try:
        eeg_d    = read_eeg_data(DATA_PATH, subj_id)
        eeg_d    = add_trial_index(eeg_d)
        beh_d    = read_behavioral_data(BEH_PATH, subj_id)
        merged_d = merge_eeg_beh(eeg_d, beh_d)
        events_d = extract_events(merged_d)

        ep_stim, ep_resp = epoch_data_both(
            eeg_d, events_d,
            EVENT_ID, RESP_ID,
            TMIN_STIM, TMAX_STIM, BASELINE_STIM,
            TMIN_RESP, TMAX_RESP, BASELINE_RESP,
        )

        first_session = list(merged_d.keys())[0]
        has_condition = CONDITION_COL in merged_d[first_session].columns

        if has_condition:
            ep_stim, ep_resp = attach_metadata(
                ep_stim, ep_resp, merged_d, CONDITION_COL, HARD_CONDITION)

        stim_all = mne.concatenate_epochs(list(ep_stim.values()))
        resp_all = mne.concatenate_epochs(list(ep_resp.values()))
        stim_all.resample(sfreq=TARGET_SFREQ)
        resp_all.resample(sfreq=TARGET_SFREQ)
        del ep_stim, ep_resp; gc.collect()

        all_evoked_stim[subj_id] = stim_all.average()
        all_evoked_resp[subj_id] = resp_all.average()

        if has_condition and stim_all.metadata is not None:
            is_hard = stim_all.metadata[CONDITION_COL] == HARD_CONDITION
            all_evoked_easy[subj_id]     = stim_all[~is_hard].average()
            all_evoked_hard[subj_id]     = stim_all[is_hard].average()
            all_evoked_cpp_easy[subj_id] = resp_all[~is_hard].average()
            all_evoked_cpp_hard[subj_id] = resp_all[is_hard].average()
            print(f"  Easy: {(~is_hard).sum()} | Hard: {is_hard.sum()}")

        freqs, psd = sp_signal.welch(
            stim_all.get_data(), fs=TARGET_SFREQ,
            nperseg=int(TARGET_SFREQ*2), axis=-1)
        all_psd[subj_id] = (freqs, psd.mean(axis=0))

        # Save immediately
        with open(SAVE_DIR / f"subj_{subj_id}_results.pkl", "wb") as f:
            pickle.dump({
                "evoked_stim"    : all_evoked_stim[subj_id],
                "evoked_resp"    : all_evoked_resp[subj_id],
                "evoked_easy"    : all_evoked_easy.get(subj_id),
                "evoked_hard"    : all_evoked_hard.get(subj_id),
                "evoked_cpp_easy": all_evoked_cpp_easy.get(subj_id),
                "evoked_cpp_hard": all_evoked_cpp_hard.get(subj_id),
                "psd"            : all_psd[subj_id],
            }, f)
        print(f"  💾  Saved")

        del stim_all, resp_all, merged_d, eeg_d; gc.collect()

    except Exception as e:
        import traceback
        print(f"  ❌  {e}"); traceback.print_exc()
        failed_subjects.append(subj_id)

print(f"\n✅  Done: {len(all_evoked_stim)} subjects")
if failed_subjects: print(f"❌  Failed: {failed_subjects}")
