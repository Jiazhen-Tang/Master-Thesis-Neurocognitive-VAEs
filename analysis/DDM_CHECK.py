# ══════════════════════════════════════════════════════════════════════════════
# DDM_CHECK.py — simulation-based DDM model adequacy check
#
# Tests whether a drift-diffusion model can reproduce the behaviour observed in
# this abstract-reasoning task, independently of the NCVA.
#
#   1. Parameters are estimated and tested PER PARTICIPANT. Pooling the reaction
#      times of 19 participants would produce a mixture distribution that no
#      single diffusion process could match, letting heterogeneity between
#      participants register as misfit.
#
#   2. Both parameter sets are run: the closed-form EZ-DDM (boundary fixed) and
#      the hierarchical Bayesian EZ-DDM (all three estimated freely). If the
#      misfit persists under free estimation it cannot be attributed to the
#      fixed values.
#
# Pure numpy — nothing to install.
#
# Run:  python analysis/DDM_CHECK.py
# ══════════════════════════════════════════════════════════════════════════════

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

TENSOR_DIR = os.path.expanduser("~/Desktop/NCVA_results/tensors")
OUT_DIR    = os.path.expanduser("~/Desktop/NCVA_results")

# order matters: the first entry appears in the upper block of the figure
PRIOR_SETS = [
    ("BayesEZ", os.path.join(OUT_DIR, "ddm_priors_bayes.pkl")),  # all three free
    ("EZ",      os.path.join(OUT_DIR, "ddm_priors.pkl")),        # boundary fixed
]

SUBJECTS = ["01","02","04","05","06","07","10","11","12","13",
            "14","15","16","17","18","19","20","21","22"]

N_SIM_PER_TRIAL = 20        # simulated reaction times per real trial
DT   = 0.001                # Euler step (s)
MAXT = 20.0                 # ceiling, generous — real reaction times reach ~10 s


def simulate_ddm(v, a, ndt, n, dt=DT, maxt=MAXT, rng=None):
    """
    Two-boundary diffusion process (Euler-Maruyama), unbiased start point.
    Returns rt in seconds and choice (1 = upper/correct, 0 = lower/incorrect).
    Noise sigma fixed at 1.0, the standard diffusion scaling.
    """
    rng = rng or np.random.default_rng(0)
    max_steps = int(maxt / dt)
    x = np.full(n, a / 2.0)
    finished = np.zeros(n, dtype=bool)
    rt = np.full(n, np.nan)
    choice = np.zeros(n, dtype=int)
    sqrt_dt = np.sqrt(dt)
    for step in range(max_steps):
        active = ~finished
        if not active.any():
            break
        x[active] += v * dt + rng.standard_normal(active.sum()) * sqrt_dt
        hit_up = active & (x >= a)
        hit_lo = active & (x <= 0)
        t = (step + 1) * dt + ndt
        rt[hit_up] = t; choice[hit_up] = 1; finished[hit_up] = True
        rt[hit_lo] = t; choice[hit_lo] = 0; finished[hit_lo] = True
    return rt, choice


def load_beh(subj):
    with open(os.path.join(TENSOR_DIR, f"subj_{subj}", f"subj_{subj}_beh.pkl"), "rb") as f:
        b = pickle.load(f)
    rt = np.asarray(b["rt"], dtype=float)
    ch = np.asarray(b["choice"], dtype=float)
    m = ~np.isnan(rt)
    return rt[m], ch[m]


def _param(entry, key):
    """The two prior files store parameters differently: one as bare floats, the
    other as {'mu': ..., 'logvar': ...}. Accept either."""
    x = entry[key]
    return float(x["mu"]) if isinstance(x, dict) else float(x)


def run_check(label, prior_path):
    if not os.path.exists(prior_path):
        print(f"\n⚠️  {label}: {os.path.basename(prior_path)} not found — skipping.")
        return None

    priors = pickle.load(open(prior_path, "rb"))
    rng = np.random.default_rng(42)

    print(f"\n{'='*88}")
    print(f"  DDM model check — {label} parameters   ({os.path.basename(prior_path)})")
    print(f"{'='*88}")
    print(f"  {'subj':<6}{'v':>7}{'a':>7}{'ndt':>7} | "
          f"{'realRT':>8}{'simRT':>8} | {'realAcc':>8}{'simAcc':>8} | "
          f"{'KS D':>7}{'p':>10}")
    print("  " + "-"*84)

    rows = []
    for s in SUBJECTS:
        if s not in priors:
            print(f"  {s:<6}  (no prior — skipped)")
            continue
        p = priors[s]["overall"]
        v, a, ndt = _param(p, "v"), _param(p, "a"), _param(p, "ndt")

        real_rt, real_ch = load_beh(s)
        sim_rt, sim_ch = simulate_ddm(v, a, ndt,
                                      len(real_rt) * N_SIM_PER_TRIAL, rng=rng)
        ok = ~np.isnan(sim_rt)
        sim_rt, sim_ch = sim_rt[ok], sim_ch[ok]

        # per-participant KS: this participant's real RT against their own simulated RT
        D, pval = stats.ks_2samp(real_rt, sim_rt)

        rows.append(dict(subj=s, v=v, a=a, ndt=ndt,
                         real_rt=real_rt, sim_rt=sim_rt,
                         real_acc=float(np.mean(real_ch)),
                         sim_acc=float(np.mean(sim_ch)),
                         D=float(D), p=float(pval)))

        print(f"  {s:<6}{v:>7.3f}{a:>7.3f}{ndt:>7.3f} | "
              f"{np.mean(real_rt):>8.2f}{np.mean(sim_rt):>8.2f} | "
              f"{np.mean(real_ch):>8.2f}{np.mean(sim_ch):>8.2f} | "
              f"{D:>7.3f}{pval:>10.1e}")

    Ds        = np.array([r["D"] for r in rows])
    ps        = np.array([r["p"] for r in rows])
    real_accs = np.array([r["real_acc"] for r in rows])
    sim_accs  = np.array([r["sim_acc"] for r in rows])

    print("  " + "-"*84)
    print(f"  Per-participant KS D : median {np.median(Ds):.3f}   "
          f"range {Ds.min():.3f}–{Ds.max():.3f}")
    print(f"  p < .05 in           : {int((ps < .05).sum())} / {len(rows)}")
    print(f"  Mean accuracy        : real {real_accs.mean():.3f}   "
          f"simulated {sim_accs.mean():.3f}")

    return dict(label=label, rows=rows, Ds=Ds, ps=ps,
                real_accs=real_accs, sim_accs=sim_accs)


# ── run both parameter sets ───────────────────────────────────────────────────
results = [r for r in (run_check(lbl, path) for lbl, path in PRIOR_SETS)
           if r is not None]
if not results:
    raise SystemExit("No parameter sets available — nothing to check.")

# ── comparison ────────────────────────────────────────────────────────────────
if len(results) > 1:
    print(f"\n{'='*88}")
    print("  Comparison across parameter sets")
    print(f"{'='*88}")
    print(f"  {'set':<10}{'median D':>10}{'D range':>16}{'p<.05':>8}"
          f"{'mean simAcc':>13}{'mean realAcc':>14}")
    print("  " + "-"*71)
    for r in results:
        print(f"  {r['label']:<10}{np.median(r['Ds']):>10.3f}"
              f"{r['Ds'].min():>8.2f}–{r['Ds'].max():<7.2f}"
              f"{int((r['ps'] < .05).sum()):>8}"
              f"{r['sim_accs'].mean():>13.3f}{r['real_accs'].mean():>14.3f}")

# ── Figure: per-participant KS D, both parameter sets ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.2))
w  = 0.8 / len(results)
xs = np.arange(len(results[0]["rows"]))
for k, r in enumerate(results):
    ax.bar(xs + k*w, r["Ds"], width=w, label=r["label"], alpha=.85)
ax.axhline(0.1, color="grey", ls="--", lw=1)
ax.text(len(xs) - 0.5, 0.11, "D = 0.1", fontsize=8, color="grey", ha="right")
ax.set_xticks(xs + w*(len(results) - 1)/2)
ax.set_xticklabels([r["subj"] for r in results[0]["rows"]], fontsize=8)
ax.set_xlabel("participant"); ax.set_ylabel("KS D (empirical vs simulated RT)")
ax.set_title("Per-participant discrepancy between empirical and simulated reaction times")
ax.legend(frameon=False)
fig.tight_layout()
p_ks = os.path.join(OUT_DIR, "ddm_check_ks_persubject.png")
fig.savefig(p_ks, dpi=150); plt.close(fig)
print(f"\n  saved → {p_ks}")

# ── Figure: RT distributions for two exemplar participants ───────────────────
# Only the freely estimated (BayesEZ) simulation is shown; the closed-form
# results are reported in the text and the appendix.
FIG_SUBJECTS = ["01"]
FIG_SET      = "BayesEZ"

res = next((r for r in results if r["label"] == FIG_SET), results[0])
rows_sel = [row for row in res["rows"] if row["subj"] in FIG_SUBJECTS]

if rows_sel:
    fig, axes = plt.subplots(1, len(rows_sel), figsize=(5.4*len(rows_sel), 4.0),
                             squeeze=False)
    for ax, row in zip(axes[0], rows_sel):
        edges = np.linspace(0, min(row["real_rt"].max(), 12), 40)
        ax.hist(row["real_rt"], bins=edges, density=True, alpha=.85,
                color="0.62", edgecolor="none", label="empirical")
        h, _ = np.histogram(row["sim_rt"], bins=edges, density=True)
        ax.step(edges[:-1], h, where="post", lw=1.8, color="#D1495B",
                label="DDM-simulated")
        ax.set_title(f"participant {row['subj']}   D = {row['D']:.2f}", fontsize=12)
        ax.set_xlabel("reaction time (s)", fontsize=11)
        ax.set_ylabel("density", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()

    p_rt = os.path.join(OUT_DIR, "ddm_check_rt_distributions.png")
    fig.savefig(p_rt, dpi=150)
    plt.close(fig)
    print(f"  saved → {p_rt}   ({FIG_SET}, participants {', '.join(FIG_SUBJECTS)})")
else:
    print(f"  ⚠️  none of {FIG_SUBJECTS} found in the {FIG_SET} results")

print(f"\n✅  Model check complete.")