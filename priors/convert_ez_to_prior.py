# ══════════════════════════════════════════════════════════════════════════════
# convert_ez_to_prior.py — convert EZ-DDM priors into the format train.py expects
#
# EZ format (flat):   d[subj][cond]['v'] = <float>
# train.py format:    d[subj][cond]['v'] = {'mu': <float>, 'logvar': <float>}
#
# EZ gives point estimates only (no variance), so we attach a FIXED, moderately
# wide logvar per parameter — the informed prior then acts as a soft anchor, not
# a hard constraint. Wider logvar = weaker pull.  (These match the loose regime
# that gave v its trial-to-trial spread in the single-subject runs.)
#
# Output: stan_ddm_priors.pkl  (same filename train.py already loads — no code
# change needed in train.py).
# ══════════════════════════════════════════════════════════════════════════════

import pickle, os

SRC = os.path.expanduser("~/Desktop/NCVA_results/ez_ddm_priors.pkl")
DST = os.path.expanduser("~/Desktop/NCVA_results/ddm_priors.pkl")

# fixed prior widths (logvar). Larger = weaker anchor.
# logvar = ln(sigma^2).  e.g. sigma=0.3 -> logvar = ln(0.09) ≈ -2.41
LOGVAR = {
    "v":   -2.4,     # sigma ≈ 0.30   (drift: moderate spread)
    "a":   -1.6,     # sigma ≈ 0.45
    "ndt": -3.0,     # sigma ≈ 0.22
}

def wrap(val, key):
    return {"mu": float(val), "logvar": float(LOGVAR[key])}

with open(SRC, "rb") as f:
    ez = pickle.load(f)

out = {}
for subj_id, conds in ez.items():
    out[subj_id] = {}
    for cond_name in ("overall", "easy", "hard"):
        if cond_name not in conds:
            continue
        c = conds[cond_name]
        out[subj_id][cond_name] = {
            "v":   wrap(c["v"],   "v"),
            "a":   wrap(c["a"],   "a"),
            "ndt": wrap(c["ndt"], "ndt"),
        }

with open(DST, "wb") as f:
    pickle.dump(out, f)

print(f"✅  Converted {len(out)} subjects (EZ-DDM source) → {DST}")
print(f"\n{'Subject':<9}{'v_overall':>11}{'a':>8}{'ndt':>8}")
print("-" * 36)
for s, p in out.items():
    o = p.get("overall", {})
    if o:
        print(f"{s:<9}{o['v']['mu']:>11.3f}{o['a']['mu']:>8.3f}{o['ndt']['mu']:>8.3f}")

# sanity check: confirm train.py's access pattern works
sample = out["01"]["overall"]["v"]["mu"]
_ = out["01"]["overall"]["v"]["logvar"]
print(f"\n  access check: priors['01']['overall']['v']['mu'] = {sample:.3f}  ✅")