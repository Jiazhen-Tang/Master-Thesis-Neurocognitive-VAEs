# ══════════════════════════════════════════════════════════════════════════════
# pipeline_utils.py — EEG preprocessing pipeline functions
#
# All functions extracted from the original notebook Cells 2–8.
# Import this module in any script that needs to load or epoch EEG data.
# ══════════════════════════════════════════════════════════════════════════════

import os, gc, re, glob
import numpy as np
import pandas as pd
import mne


def read_eeg_data(data_path, subject_id):
    eeg_dict  = {}
    all_files = sorted([f for f in os.listdir(data_path) if f.endswith(".fif")])

    for file in all_files:
        match = re.search(r"subj_?(\d{2})(\d{2})", file)
        if not match:
            continue
        file_subj, session_id = match.group(1), match.group(2)
        if file_subj != subject_id:
            continue

        raw = mne.io.read_raw_fif(
            os.path.join(data_path, file), preload=False, verbose="error"
        )
        df = pd.DataFrame({
            "description": raw.annotations.description,
            "onset":       raw.annotations.onset,
        })

        trial_starts = df[df["description"] == "trial_start"].index.tolist()
        trial_ends   = df[df["description"] == "trial_end"].index.tolist()

        if len(trial_starts) > len(trial_ends):
            n_drop = len(trial_starts) - len(trial_ends)
            trial_starts = trial_starts[:len(trial_ends)]

        trial_boundaries = list(zip(trial_starts, trial_ends))
        if session_id == "01":
            trial_boundaries = trial_boundaries[3:]

        valid_indices = []
        for start, end in trial_boundaries:
            seg = df.loc[start:end]
            if "timeout" not in seg["description"].values:
                valid_indices.extend(seg.index.tolist())

        df = df.loc[valid_indices].reset_index(drop=True)
        df = df[df["description"].isin(["stim-all_stim", "trial_end"])].reset_index(drop=True)

        eeg_dict[session_id] = {"raw": raw, "df": df}
        print(f"  ✅  Session {session_id}: {len(df)//2} trials loaded")

    return eeg_dict


def add_trial_index(eeg_dict):
    for session_id, data in eeg_dict.items():
        df  = data["df"]
        raw = data["raw"]

        n_stim = df["description"].eq("stim-all_stim").sum()
        df["trial_id"] = None
        df.loc[df["description"] == "stim-all_stim", "trial_id"] = range(n_stim)
        df["trial_id"] = df["trial_id"].ffill()

        sfreq = int(raw.info["sfreq"])
        df["sample"]   = (df["onset"] * sfreq).astype(int)
        df["event_id"] = df["description"].map({"stim-all_stim": 1, "trial_end": 2})

        eeg_dict[session_id]["df"] = df[
            ["sample", "event_id", "trial_id", "onset", "description"]
        ]
        print(f"  ✅  Session {session_id}: trial index added")

    return eeg_dict


def read_behavioral_data(beh_path, subject_id):
    beh_dict  = {}
    all_paths = sorted(glob.glob(
        os.path.join(beh_path, "**/*behav.csv"), recursive=True
    ))

    for path in all_paths:
        fname = os.path.basename(path)
        if not fname.startswith(subject_id):
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"Unnamed: 0": "trial_id"})
        df = df[df["rt"] != "timeout"].reset_index(drop=True)
        session_id = fname.split("-")[1]
        beh_dict[session_id] = df
        print(f"  ✅  Session {session_id}: {len(df)} behavioral trials loaded")

    return beh_dict


def merge_eeg_beh(eeg_dict, beh_dict, merge_column="trial_id"):
    merged_dict = {}
    for session_id, data in eeg_dict.items():
        if session_id not in beh_dict:
            print(f"  ⚠️  Session {session_id}: no behavioral data, skipping")
            continue
        merged = pd.merge(data["df"], beh_dict[session_id],
                          on=merge_column, how="inner")
        merged_dict[session_id] = merged
        print(f"  ✅  Session {session_id}: {len(merged)} rows merged")
    return merged_dict


def extract_events(merged_dict):
    events_dict = {}
    for session_id, df in merged_dict.items():
        arr = df[["sample", "event_id"]].copy()
        arr.insert(1, "previous", 0)
        events_dict[session_id] = arr.values.astype(int)
    return events_dict


def epoch_data_both(eeg_dict, events_dict,
                    event_id_stim, event_id_resp,
                    tmin_stim, tmax_stim, baseline_stim,
                    tmin_resp, tmax_resp, baseline_resp,
                    min_trials=10):

    epochs_stim_dict = {}
    epochs_resp_dict = {}

    for session_id, events in events_dict.items():
        raw = eeg_dict[session_id]["raw"]

        if not raw.preload:
            raw.load_data()

        if raw.info["bads"]:
            raw.interpolate_bads(reset_bads=True)

        ep_stim = mne.Epochs(
            raw, events=events, event_id=event_id_stim,
            tmin=tmin_stim, tmax=tmax_stim, baseline=baseline_stim,
            preload=True, reject=dict(eeg=200e-6), verbose=False,
        )
        ep_resp = mne.Epochs(
            raw, events=events, event_id=event_id_resp,
            tmin=tmin_resp, tmax=tmax_resp, baseline=baseline_resp,
            preload=True, reject=dict(eeg=200e-6), verbose=False,
        )

        if len(ep_stim) < min_trials or len(ep_resp) < min_trials:
            print(f"  ⚠️  Session {session_id}: too few trials "
                  f"(stim={len(ep_stim)}, resp={len(ep_resp)}) — skipping")
            continue

        epochs_stim_dict[session_id] = ep_stim
        epochs_resp_dict[session_id] = ep_resp
        print(f"  ✅  Session {session_id}: "
              f"stim={len(ep_stim)} | resp={len(ep_resp)} trials")

    return epochs_stim_dict, epochs_resp_dict


def attach_metadata(ep_stim, ep_resp, merged_d, condition_col, hard_condition):
    """
    Attach behavioral metadata to epoch objects before concatenation.
    Must be called BEFORE mne.concatenate_epochs (sample indices get reset after).
    """
    for session_id in ep_stim.keys():
        beh_sess = merged_d[session_id]

        beh_stim_sess = beh_sess[beh_sess["event_id"] == 1].reset_index(drop=True)
        beh_resp_sess = beh_sess[beh_sess["event_id"] == 2].reset_index(drop=True)

        kept_stim = ep_stim[session_id].events[:, 0]
        kept_resp = ep_resp[session_id].events[:, 0]

        meta_stim = beh_stim_sess[
            beh_stim_sess["sample"].isin(kept_stim)
        ].reset_index(drop=True)
        meta_resp = beh_resp_sess[
            beh_resp_sess["sample"].isin(kept_resp)
        ].reset_index(drop=True)

        assert len(meta_stim) == len(ep_stim[session_id]), (
            f"Session {session_id} stim mismatch: "
            f"{len(meta_stim)} vs {len(ep_stim[session_id])}"
        )
        assert len(meta_resp) == len(ep_resp[session_id]), (
            f"Session {session_id} resp mismatch: "
            f"{len(meta_resp)} vs {len(ep_resp[session_id])}"
        )

        ep_stim[session_id].metadata = meta_stim
        ep_resp[session_id].metadata = meta_resp

    return ep_stim, ep_resp
