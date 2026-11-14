# ══════════════════════════════════════════════════════════════════════════════
# config.py — Central configuration for the NCVA pipeline
#
# Edit this file to change paths and parameters.
# All other scripts import from here — never hardcode paths elsewhere.
# ══════════════════════════════════════════════════════════════════════════════

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path("/Volumes/psychology$/Projects/2024_Pinier_FMG-7163_reasoning_EEG/Lab/data/preprocessed_eeg")
BEH_PATH  = Path("/Volumes/psychology$/Projects/2024_Pinier_FMG-7163_reasoning_EEG/Lab/data/Lab")

LOCAL_BASE   = Path(os.path.expanduser("~")) / "Desktop" / "NCVA_results"
SAVE_DIR     = LOCAL_BASE                          # ERP evoked .pkl files
TENSOR_DIR   = LOCAL_BASE / "tensors"              # .npy epoch tensors
FRP_BASE_DIR = LOCAL_BASE / "frp_outputs"          # FRP .fif files
MODEL_DIR    = LOCAL_BASE / "models"               # trained model checkpoints

for d in [SAVE_DIR, TENSOR_DIR, FRP_BASE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Subjects ───────────────────────────────────────────────────────────────────
EXCLUDE_SUBJ = {"08", "09", "25", "03", "23", "24"}
ALL_SUBJECTS = [f"{i:02d}" for i in range(1, 26)]
SUBJECT_IDS  = [s for s in ALL_SUBJECTS if s not in EXCLUDE_SUBJ]

# ── EEG Epoching ───────────────────────────────────────────────────────────────
EVENT_ID  = {"stimulus": 1}
RESP_ID   = {"response": 2}

TMIN_STIM, TMAX_STIM = -0.2,  4.5
TMIN_RESP, TMAX_RESP = -4.5,  0.2
BASELINE_STIM        = (None, 0)
BASELINE_RESP        = None
TARGET_SFREQ         = 128.0

CONDITION_COL  = "pattern"
HARD_CONDITION = "ABCDEEDC"

# ── Electrode picks (replaces SVD spatial filtering) ───────────────────────────
# P300 (stimulus-locked): Pz is the classical electrode; average with neighbours
# for a slightly more robust estimate without needing SVD.
P300_PICKS = ["Pz", "P1", "P2"]          # averaged → single channel

# Readiness Potential (response-locked): Cz is the classical electrode.
RP_PICKS   = ["Cz", "FCz", "CPz"]        # averaged → single channel

# FRP: frontal fixation-related potential picks (unchanged)
FRONTAL_PICKS = [
    "Fp1","Fpz","Fp2","AF7","AF3","AFz","AF4","AF8",
    "F7","F5","F3","F1","Fz","F2","F4","F6","F8",
]

# ── FRP Config ─────────────────────────────────────────────────────────────────
ET_RAW_PATH = BEH_PATH
FRP_TMIN    = -0.100
FRP_TMAX    =  0.600    # 0.7s total window (matches Pinier et al.)
FRP_BASELINE = (None, 0)
MIN_FIX_DUR  = 0.020   # 20ms minimum fixation duration

SCREEN_RES  = (2560, 1440)
IMG_SIZE    = (213, 213)
X_POS_SEQ   = [-1042.222, -744.444, -446.667, -148.889,
                148.889,   446.667,  744.444, 1042.222]
Y_POS_SEQ   = 213

# ── Model Hyperparameters ──────────────────────────────────────────────────────
Z_DIM  = 32
D_FEAT = 32

# ── Training Hyperparameters ───────────────────────────────────────────────────
P1_END    = 50     # Phase 1: train PosteriorDDM only
P2_END    = 120    # Phase 2: train VAE encoder + decoder
P3_END    = 250    # Phase 3: joint training
N_EPOCHS  = 320    # Phase 4: train bridgers
NDT_EPOCH = 20     # ndt warmup period

BATCH_SIZE  = 64
LR          = 5e-4
EVAL_EVERY  = 25
SAVE_EVERY  = 15

REC_AVG_W   = 1.0
REC_X_W     = 1.0
CHOICE_W    = 1.0
KL_DDM_W    = [0.1, 0.1, 0.1]     # [v, a, ndt]
KL_Y_WEIGHT = 1.0
