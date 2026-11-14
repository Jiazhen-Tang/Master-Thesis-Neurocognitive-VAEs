import pickle, glob, os, pandas as pd, numpy as np

BEH_PATH   = "/Volumes/psychology$/Projects/2024_Pinier_FMG-7163_reasoning_EEG/Lab/data/Lab"
TENSOR_DIR = os.path.expanduser("~/Desktop/NCVA_results/tensors")

FIX_SUBJECTS = SUBJECT_IDS = [
    "01","02","03","04","05","06","07","10","11","12",
    "13","14","15","16","17","18","19","20","21","22",
    "23","24"
]

for subj_id in FIX_SUBJECTS:
    beh_path = os.path.join(TENSOR_DIR, f"subj_{subj_id}",
                            f"subj_{subj_id}_beh.pkl")
    with open(beh_path, "rb") as f:
        beh = pickle.load(f)

    paths = sorted(glob.glob(
        os.path.join(BEH_PATH, f"**/{subj_id}*behav.csv"), recursive=True
    ))

    all_correct = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[df["rt"] != "timeout"].reset_index(drop=True)
        all_correct.extend(
            df["correct"].map(
                lambda x: 1.0 if str(x).lower() == "true" else 0.0
            ).values
        )

    new_choice = np.array(all_correct[:len(beh['choice'])], dtype="float32")
    old_acc    = beh['choice'].mean()
    new_acc    = new_choice.mean()
    beh['choice'] = new_choice

    with open(beh_path, "wb") as f:
        pickle.dump(beh, f)

    print(f"Subject {subj_id}: {old_acc:.3f} → {new_acc:.3f} ✅")