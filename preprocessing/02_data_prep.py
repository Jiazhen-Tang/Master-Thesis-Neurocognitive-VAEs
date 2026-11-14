# ══════════════════════════════════════════════════════════════════════════════
# preprocessing/02_data_prep.py — Extract epoch tensors for model training
#
# Saves per subject:
#   subj_XX_X_stim.npy   (N, eeg_chan, time_stim)
#   subj_XX_X_resp.npy   (N, eeg_chan, time_resp)
#   subj_XX_beh.pkl      {rt, condition, choice, pattern}
# ══════════════════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc, pickle
import numpy as np
import pandas as pd
import mne

from config import *
from pipeline_utils import (read_eeg_data, add_trial_index,
                             read_behavioral_data, merge_eeg_beh,
                             extract_events, epoch_data_both,
                             attach_metadata)


def is_done(subj_id):
    d = TENSOR_DIR / f"subj_{subj_id}"
    return all((d / f"subj_{subj_id}_{x}").exists()
               for x in ["X_stim.npy", "X_resp.npy", "beh.pkl"])


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

        X_stim = stim_all.get_data().astype("float32")
        X_resp = resp_all.get_data().astype("float32")
        print(f"  Stim: {X_stim.shape}  Resp: {X_resp.shape}")

        if has_condition and stim_all.metadata is not None:
            meta    = stim_all.metadata.copy()
            rt_vals = pd.to_numeric(meta["rt"], errors="coerce").values.astype("float32")
            is_hard = (meta[CONDITION_COL] == HARD_CONDITION).astype(int).values
            cond_oh = np.eye(2, dtype="float32")[is_hard]
            choice  = meta["correct"].map(lambda x: 1.0 if str(x).lower() == "true" else 0.0).values.astype("float32")
            pattern = meta[CONDITION_COL].values
        else:
            n = len(stim_all)
            rt_vals = np.ones(n, dtype="float32")
            cond_oh = np.zeros((n, 2), dtype="float32"); cond_oh[:, 0] = 1.0
            choice  = np.ones(n, dtype="float32")
            pattern = None

        subj_dir = TENSOR_DIR / f"subj_{subj_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        np.save(subj_dir / f"subj_{subj_id}_X_stim.npy", X_stim)
        np.save(subj_dir / f"subj_{subj_id}_X_resp.npy", X_resp)
        with open(subj_dir / f"subj_{subj_id}_beh.pkl", "wb") as f:
            pickle.dump({"rt": rt_vals, "condition": cond_oh,
                         "choice": choice, "pattern": pattern}, f)

        print(f"  ✅  Saved — rt_mean={np.nanmean(rt_vals):.2f}s  "
              f"easy={(cond_oh[:,0]==1).sum()} hard={(cond_oh[:,1]==1).sum()}")

        del stim_all, resp_all, X_stim, X_resp, merged_d, eeg_d; gc.collect()

    except Exception as e:
        import traceback
        print(f"  ❌  {e}"); traceback.print_exc()
        failed.append(subj_id)

done = [s for s in SUBJECT_IDS if is_done(s)]
print(f"\n✅  Complete: {len(done)} subjects")
if failed: print(f"❌  Failed : {failed}")
