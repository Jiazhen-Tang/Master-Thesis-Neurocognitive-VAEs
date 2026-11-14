# ══════════════════════════════════════════════════════════════════════════════
# preprocessing/build_clean_eeg.py — Rebuild clean 64-channel EEG tensors from raw
#
# Bad channels are ALREADY interpolated upstream (epoch_data_both calls
# raw.interpolate_bads()), so the only thing left is to drop the 5 non-brain
# channels (BioSemi Status trigger + 4 EOG) → clean 64-channel scalp EEG.
#
# Output (per subject):
#   subj_XX_X_stim_eeg.npy   (N, 64, T)
#   subj_XX_X_resp_eeg.npy   (N, 64, T)
#   subj_XX_eeg_channels.txt (the 64 channel names, in order)
#
# beh.pkl is NOT touched. Resume-safe: skips subjects already at 64 channels.
#
# Run:  python preprocessing/build_clean_eeg.py
# ══════════════════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc, pickle
import numpy as np
import mne

from config import *
from pipeline_utils import (read_eeg_data, add_trial_index,
                             read_behavioral_data, merge_eeg_beh,
                             extract_events, epoch_data_both,
                             attach_metadata)

# ── Non-brain channels to drop (Status trigger + EOG) ──────────────────────────
NON_EEG = ["Status",
           "EOGL", "EOGR", "EOGT", "EOGB",
           "EXG1", "EXG2", "EXG3", "EXG4",
           "EXG5", "EXG6", "EXG7", "EXG8",
           "M1", "M2", "Mastoid1", "Mastoid2"]


def is_done(subj_id, n_expect=64):
    d = TENSOR_DIR / f"subj_{subj_id}"
    sp = d / f"subj_{subj_id}_X_stim_eeg.npy"
    rp = d / f"subj_{subj_id}_X_resp_eeg.npy"
    if not (sp.exists() and rp.exists()):
        return False
    return (np.load(sp, mmap_mode="r").shape[1] == n_expect and
            np.load(rp, mmap_mode="r").shape[1] == n_expect)


REMAINING = [s for s in SUBJECT_IDS if not is_done(s)]
print(f"📋  Already done : {len(SUBJECT_IDS) - len(REMAINING)}")
print(f"📋  Remaining    : {REMAINING}\n")

failed = []

for subj_id in REMAINING:
    print(f"\n{'─'*50}\n  Subject {subj_id}\n{'─'*50}")
    try:
        eeg_d    = read_eeg_data(DATA_PATH, subj_id)
        eeg_d    = add_trial_index(eeg_d)
        beh_d    = read_behavioral_data(BEH_PATH, subj_id)
        merged_d = merge_eeg_beh(eeg_d, beh_d)
        events_d = extract_events(merged_d)

        # epoch_data_both interpolates bads internally before epoching
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

        # ── Drop non-brain channels → clean scalp EEG ──────────────────────
        keep = [ch for ch in stim_all.ch_names if ch not in NON_EEG]
        dropped = [ch for ch in stim_all.ch_names if ch in NON_EEG]
        print(f"  Keeping {len(keep)} EEG channels, dropping {len(dropped)}: {dropped}")

        idx_stim = [stim_all.ch_names.index(ch) for ch in keep]
        idx_resp = [resp_all.ch_names.index(ch) for ch in keep]

        X_stim = stim_all.get_data()[:, idx_stim, :].astype("float32")
        X_resp = resp_all.get_data()[:, idx_resp, :].astype("float32")
        print(f"  Stim: {X_stim.shape}   Resp: {X_resp.shape}")

        # ── Save ───────────────────────────────────────────────────────────
        subj_dir = TENSOR_DIR / f"subj_{subj_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(subj_dir / f"subj_{subj_id}_X_stim_eeg.npy", X_stim)
        np.save(subj_dir / f"subj_{subj_id}_X_resp_eeg.npy", X_resp)
        with open(subj_dir / f"subj_{subj_id}_eeg_channels.txt", "w") as f:
            f.write("\n".join(keep))

        print(f"  ✅  Saved 64-channel tensors for Subject {subj_id}")

        del stim_all, resp_all, X_stim, X_resp, merged_d, eeg_d; gc.collect()

    except Exception as e:
        import traceback
        print(f"  ❌  {e}"); traceback.print_exc()
        failed.append(subj_id)

done = [s for s in SUBJECT_IDS if is_done(s)]
print(f"\n{'═'*50}")
print(f"  ✅  Complete: {len(done)} subjects")
if failed:
    print(f"  ❌  Failed : {failed}")
print(f"{'═'*50}")