# ══════════════════════════════════════════════════════════════════════════════
# ez_ddm.py — EZ-DDM prior estimation for NCVA
#
# Wagenmakers et al. (2007) EZ-DDM:
#   Inputs:  accuracy (Pc), mean RT (MRT), variance of RT (VRT)
#   Outputs: drift rate (v), boundary (a), non-decision time (ndt)
#
# Usage:
#   1. Run this script once to compute priors for all subjects
#   2. Priors saved to NCVA_results/ez_ddm_priors.pkl
#   3. train.py loads priors and passes to KL loss
#
# CHANGES FROM PREVIOUS VERSION
#   - s = 1.0 instead of 0.1, to match the scaling convention used by the
#     forward simulator and by the hierarchical Bayesian EZ fit. Under s = 0.1
#     the boundary estimates came out around 0.15 and were clipped to the
#     lower bound of 0.5 for every subject.
#   - the absolute 2.0 s ceiling on ndt has been removed; ndt is now capped
#     only at 0.95 * MRT, which is a definitional constraint rather than an
#     assumption about how long non-decision processes ought to take.
#   - mean decision time and its share of the total RT are now recorded, since
#     that ratio is what shows whether the fitted parameters remain
#     cognitively interpretable.
#   - the quantity under the fourth root is checked before being raised, so a
#     non-positive value raises rather than silently producing NaN.
# ══════════════════════════════════════════════════════════════════════════════

import os, pickle
import numpy as np
from pathlib import Path

TENSOR_DIR = Path(os.path.expanduser("~")) / "Desktop" / "NCVA_results" / "tensors"
SAVE_PATH  = Path(os.path.expanduser("~")) / "Desktop" / "NCVA_results" / "ez_ddm_priors.pkl"

SUBJECT_IDS = [
    "01","02","03","04","05","06","07","10","11","12",
    "13","14","15","16","17","18","19","20","21","22",
    "23","24"
]

# Scaling convention. Must match the simulator and the Bayesian EZ fit.
S_SCALE = 1.0

# Fallback values, used only when EZ is inapplicable (accuracy at or below
# chance) or when the estimator fails. These are on the s = 1 scale.
FALLBACK = (0.5, 1.0, 0.3)


def ez_ddm(pc, mrt, vrt, s=S_SCALE):
    """
    EZ-DDM parameter estimation (Wagenmakers et al., 2007).

    Parameters
    ----------
    pc  : float  proportion correct (accuracy), e.g. 0.85
    mrt : float  mean RT of CORRECT trials (seconds)
    vrt : float  variance of RT of CORRECT trials (seconds^2)
    s   : float  scaling parameter

    Returns
    -------
    v   : float  drift rate
    a   : float  boundary separation
    ndt : float  non-decision time
    mdt : float  mean decision time (mrt - ndt)
    """
    # Clamp pc away from 0 and 1 so the logit stays finite.
    pc = float(np.clip(pc, 0.51, 0.999))

    # Step 1: logit of pc
    L = np.log(pc / (1 - pc))

    # Step 2: drift rate v
    inner = L * (L**2 * vrt - L * vrt + vrt - s**2)
    if not np.isfinite(inner) or inner <= 0:
        raise ValueError(
            f"EZ inner term non-positive (inner={inner:.4g}, pc={pc:.3f}, "
            f"vrt={vrt:.4g}) — EZ not applicable to these summary statistics"
        )
    v = np.sign(pc - 0.5) * s * inner**0.25 / vrt**0.25

    if not np.isfinite(v) or v == 0:
        raise ValueError(f"EZ produced invalid drift rate (v={v})")

    # Step 3: boundary a
    a = s**2 * L / v

    # Step 4: mean decision time, then non-decision time as the remainder
    y = np.exp(-v * a / s**2)
    mdt = (a / (2 * v)) * (1 - y) / (1 + y)
    ndt = mrt - mdt

    return float(v), float(a), float(ndt), float(mdt)


def _estimate(pc, mrt, vrt, label=""):
    """
    Run EZ with sanity bounds, falling back to fixed values if it is not
    applicable. Returns (v, a, ndt, mdt, used_fallback).
    """
    if pc < 0.55:
        print(f"    {label}: accuracy {pc:.2f} too low for EZ-DDM, using defaults")
        v, a, ndt = FALLBACK
        return v, a, ndt, mrt - ndt, True

    try:
        v, a, ndt, mdt = ez_ddm(pc, mrt, max(vrt, 1e-4))
    except Exception as e:
        print(f"    {label}: EZ-DDM failed ({e}), using defaults")
        v, a, ndt = FALLBACK
        return v, a, ndt, mrt - ndt, True

    # Sanity bounds. These are wide enough not to bind in normal cases; if one
    # of them does bind, the printout below flags it.
    v_c   = float(np.clip(v,   0.1,  10.0))
    a_c   = float(np.clip(a,   0.1,   5.0))
    # ndt is bounded below by zero and above by the mean RT itself. No
    # absolute ceiling is imposed.
    ndt_c = float(np.clip(ndt, 0.01, 0.95 * mrt))

    for name, raw, clipped in (("v", v, v_c), ("a", a, a_c), ("ndt", ndt, ndt_c)):
        if not np.isclose(raw, clipped):
            print(f"    {label}: {name} clipped {raw:.3f} -> {clipped:.3f}")

    return v_c, a_c, ndt_c, mrt - ndt_c, False


def compute_ez_priors(subj_id):
    """
    Load behavioural data for one subject and compute EZ-DDM priors.
    Returns dict with v, a, ndt estimates per condition.
    """
    beh_path = TENSOR_DIR / f"subj_{subj_id}" / f"subj_{subj_id}_beh.pkl"
    if not beh_path.exists():
        return None

    with open(beh_path, "rb") as f:
        beh = pickle.load(f)

    rt     = beh["rt"].astype(float)
    cond   = beh["condition"]       # (N, 2) one-hot [Easy, Hard]
    choice = beh["choice"].astype(float)

    # Remove NaN RT
    valid  = ~np.isnan(rt)
    rt     = rt[valid]
    cond   = cond[valid]
    choice = choice[valid]

    results = {}

    def _summary(rt_sub, choice_sub):
        pc = choice_sub.mean()
        if (choice_sub == 1).sum() > 1:
            mrt = rt_sub[choice_sub == 1].mean()
            vrt = rt_sub[choice_sub == 1].var()
        else:
            mrt = rt_sub.mean()
            vrt = rt_sub.var()
        return float(pc), float(mrt), float(vrt)

    # Per condition
    for c_idx, c_name in [(0, "easy"), (1, "hard")]:
        mask = cond[:, c_idx] == 1
        if mask.sum() < 10:
            print(f"    Subject {subj_id} {c_name}: too few trials ({mask.sum()}), skipping")
            continue

        pc, mrt, vrt = _summary(rt[mask], choice[mask])
        v, a, ndt, mdt, fb = _estimate(pc, mrt, vrt, f"Subject {subj_id} {c_name}")

        results[c_name] = {
            "v": v, "a": a, "ndt": ndt,
            "mdt": mdt, "mdt_frac": mdt / mrt if mrt > 0 else np.nan,
            "pc": pc, "mrt": mrt, "vrt": vrt,
            "n_trials": int(mask.sum()),
            "fallback": fb,
        }

    # Overall, collapsing across conditions
    pc_all, mrt_all, vrt_all = _summary(rt, choice)
    v_all, a_all, ndt_all, mdt_all, fb_all = _estimate(
        pc_all, mrt_all, vrt_all, f"Subject {subj_id} overall"
    )

    results["overall"] = {
        "v": v_all, "a": a_all, "ndt": ndt_all,
        "mdt": mdt_all, "mdt_frac": mdt_all / mrt_all if mrt_all > 0 else np.nan,
        "pc": pc_all, "mrt": mrt_all, "vrt": vrt_all,
        "n_trials": int(valid.sum()),
        "fallback": fb_all,
    }

    return results


# ── Run for all subjects ───────────────────────────────────────────────────────
all_priors = {}
failed     = []

header = (f"{'Subject':<9} {'Cond':<8} {'v':>6} {'a':>6} {'ndt':>6} "
          f"{'MDT':>6} {'MDT%':>6} {'pc':>6} {'mrt':>6} {'n':>5}")
print(header)
print("─" * len(header))

for subj_id in SUBJECT_IDS:
    priors = compute_ez_priors(subj_id)
    if priors is None:
        print(f"  Subject {subj_id}: not found")
        failed.append(subj_id)
        continue

    all_priors[subj_id] = priors

    for cond_name in ("easy", "hard", "overall"):
        if cond_name not in priors:
            continue
        vals = priors[cond_name]
        flag = " *" if vals["fallback"] else ""
        print(f"{subj_id:<9} {cond_name:<8} "
              f"{vals['v']:>6.3f} {vals['a']:>6.3f} {vals['ndt']:>6.3f} "
              f"{vals['mdt']:>6.3f} {100*vals['mdt_frac']:>5.1f}% "
              f"{vals['pc']:>6.3f} {vals['mrt']:>6.3f} {vals['n_trials']:>5}{flag}")

# ── Summary of the decision-time share ────────────────────────────────────────
fracs = [p["overall"]["mdt_frac"] for p in all_priors.values()
         if not p["overall"]["fallback"] and np.isfinite(p["overall"]["mdt_frac"])]
if fracs:
    fracs = np.array(fracs)
    print("\n" + "─" * len(header))
    print(f"Decision time as a share of total RT (EZ-applicable subjects, n={len(fracs)}):")
    print(f"  median {100*np.median(fracs):.1f}%   "
          f"range {100*fracs.min():.1f}%–{100*fracs.max():.1f}%")
    print("  The remainder is attributed to non-decision time.")

# ── Save ───────────────────────────────────────────────────────────────────────
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SAVE_PATH, "wb") as f:
    pickle.dump(all_priors, f)

print(f"\nEZ-DDM priors saved to: {SAVE_PATH}")
print(f"Scaling convention: s = {S_SCALE}")
if failed:
    print(f"Not found: {failed}")