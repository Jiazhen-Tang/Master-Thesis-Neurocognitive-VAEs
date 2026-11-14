# ══════════════════════════════════════════════════════════════════════════════
# bayesian_ez_compare.py — self-contained hierarchical Bayesian EZ-DDM (PyMC)
#
# Translates the EZ-diffusion forward equations into a hierarchical Bayesian
# model and fits them to the behavioural summary stats (accuracy, mean RT,
# var RT) per subject. Then compares the Bayesian posterior-mean drift against
# your plain closed-form EZ-DDM drift.
#
# EZ forward equations (Wagenmakers, van der Maas & Grasman, 2007):
#   y   = exp(-a * v)              (a=boundary, v=drift)
#   Pc  = 1 / (1 + y)             predicted accuracy
#   MDT = (a / 2v) * (1 - y)/(1 + y)     mean decision time
#   MRT = MDT + Ter               predicted mean RT (Ter = non-decision time)
#   VRT = (a / 2v^3)*(1 - 2*a*v*y - y^2)/(y + 1)^2   predicted variance of RT
#
# Only dependency: pymc (pip install pymc). No JAGS, no external repo.
#
# Run:  python priors/bayesian_ez_compare.py
# ══════════════════════════════════════════════════════════════════════════════

import os, sys, pickle
import numpy as np

# ── env self-check ────────────────────────────────────────────────────────────
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
except Exception as e:
    print("‼️  PyMC not available:", e)
    print("   Install with:  pip install pymc arviz")
    print("   (pure Python, no JAGS/Stan needed)")
    sys.exit(1)

TENSOR_DIR = os.path.expanduser("~/Desktop/NCVA_results/tensors")
EZ_PRIORS  = os.path.expanduser("~/Desktop/NCVA_results/ez_ddm_priors.pkl")
OUT_DIR    = os.path.expanduser("~/Desktop/NCVA_results")
SUBJECTS   = ["01","02","04","05","06","07","10","11","12","13",
              "14","15","16","17","18","19","20","21","22"]
RT_MIN, RT_MAX = 0.25, 12.0


def load_beh(subj):
    with open(os.path.join(TENSOR_DIR, f"subj_{subj}", f"subj_{subj}_beh.pkl"), "rb") as f:
        b = pickle.load(f)
    rt = np.asarray(b["rt"], dtype=float)
    ch = np.asarray(b["choice"], dtype=float)
    m = (~np.isnan(rt)) & (rt > RT_MIN) & (rt < RT_MAX)
    return rt[m], ch[m]

# ── build per-subject summary statistics ──────────────────────────────────────
meanRT, varRT, n_correct, n_trials = [], [], [], []
for s in SUBJECTS:
    rt, ch = load_beh(s)
    meanRT.append(np.mean(rt))
    varRT.append(np.var(rt))
    n_correct.append(int(np.sum(ch)))
    n_trials.append(len(ch))
meanRT = np.array(meanRT); varRT = np.array(varRT)
n_correct = np.array(n_correct); n_trials = np.array(n_trials)
S = len(SUBJECTS)
print(f"Built summary for {S} subjects.  mean RT range {meanRT.min():.2f}-{meanRT.max():.2f}s")

# ── hierarchical Bayesian EZ-DDM ──────────────────────────────────────────────
print("\nBuilding PyMC model...")
with pm.Model() as model:
    # group-level
    drift_mean = pm.Normal("drift_mean", mu=0.0, sigma=2.0)
    bound_mean = pm.TruncatedNormal("bound_mean", mu=1.0, sigma=1.0, lower=0.1, upper=3.0)
    nondt_mean = pm.TruncatedNormal("nondt_mean", mu=0.3, sigma=0.3, lower=0.05, upper=None)
    drift_sd = pm.HalfNormal("drift_sd", 1.0)
    bound_sd = pm.HalfNormal("bound_sd", 0.5)
    nondt_sd = pm.HalfNormal("nondt_sd", 0.3)

    # subject-level (non-centered)
    drift = pm.Normal("drift", mu=drift_mean, sigma=drift_sd, shape=S)
    bound = pm.TruncatedNormal("bound", mu=bound_mean, sigma=bound_sd,
                               lower=0.1, upper=3.0, shape=S)
    nondt = pm.TruncatedNormal("nondt", mu=nondt_mean, sigma=nondt_sd,
                               lower=0.05, upper=None, shape=S)

    # EZ forward equations
    y   = pt.exp(-bound * drift)
    Pc  = 1.0 / (1.0 + y)
    MDT = (bound / (2.0 * drift)) * (1.0 - y) / (1.0 + y)
    MRT = MDT + nondt
    PRT = (2.0 * drift**3 / bound) * (y + 1.0)**2 / \
          (2.0 * -bound * drift * y - y*y + 1.0)      # precision of RT (1/var)

    # likelihoods on the summary statistics (EZ sampling distributions)
    pm.Binomial("obs_correct", n=n_trials, p=Pc, observed=n_correct)
    pm.Normal("obs_meanRT", mu=MRT, sigma=pt.sqrt(1.0/(PRT*n_trials)), observed=meanRT)
    # var of RT ~ scaled around 1/PRT
    pm.Normal("obs_varRT", mu=1.0/PRT,
              sigma=pt.sqrt(2.0)/(PRT*pt.sqrt(n_trials-1.0)), observed=varRT)

    print("Sampling (this can take a few minutes)...")
    idata = pm.sample(1000, tune=1000, chains=4, cores=1,
                      target_accept=0.95, progressbar=True, random_seed=42)

# ── extract posterior drift ───────────────────────────────────────────────────
post = idata.posterior["drift"]                 # (chain, draw, subject)
drift_bayes = post.mean(dim=["chain","draw"]).values
drift_sd_post = post.std(dim=["chain","draw"]).values

# also extract boundary and ndt posteriors (Bayesian EZ estimates these too)
bound_bayes = idata.posterior["bound"].mean(dim=["chain","draw"]).values
bound_sd    = idata.posterior["bound"].std(dim=["chain","draw"]).values
nondt_bayes = idata.posterior["nondt"].mean(dim=["chain","draw"]).values
nondt_sd    = idata.posterior["nondt"].std(dim=["chain","draw"]).values

# convergence check
summ = az.summary(idata, var_names=["drift_mean","drift_sd"])
print("\nConvergence (r_hat should be ~1.00):")
print(summ[["mean","sd","r_hat"]].to_string())

# ── compare with plain EZ ─────────────────────────────────────────────────────
ez = pickle.load(open(EZ_PRIORS, "rb"))
drift_plain = np.array([ez[s]["overall"]["v"]   for s in SUBJECTS])
bound_plain = np.array([ez[s]["overall"]["a"]   for s in SUBJECTS])
nondt_plain = np.array([ez[s]["overall"]["ndt"] for s in SUBJECTS])

print(f"\n{'='*78}")
print(f"  All 3 DDM params: plain EZ (v real, a/ndt FIXED) vs Bayesian EZ (all real)")
print(f"{'='*78}")
print(f"  {'subj':<5}| {'v_plain':>8}{'v_bayes':>8} | "
      f"{'a_plain':>8}{'a_bayes':>8} | {'ndt_pln':>8}{'ndt_bay':>8}")
print("  " + "-"*72)
for i, s in enumerate(SUBJECTS):
    print(f"  {s:<5}| {drift_plain[i]:>8.3f}{drift_bayes[i]:>8.3f} | "
          f"{bound_plain[i]:>8.3f}{bound_bayes[i]:>8.3f} | "
          f"{nondt_plain[i]:>8.3f}{nondt_bayes[i]:>8.3f}")
print("  " + "-"*72)
print(f"  NOTE: plain EZ has a≈0.5 and ndt≈2.0 as FIXED/default values (not estimated).")
print(f"        Bayesian EZ estimates all three — compare especially the ndt column:")
print(f"        plain ndt mean = {nondt_plain.mean():.2f}s   Bayes ndt mean = {nondt_bayes.mean():.2f}s")
print(f"        plain a   mean = {bound_plain.mean():.2f}    Bayes a   mean = {bound_bayes.mean():.2f}")

print(f"\n  Drift comparison:")
print(f"  mean |diff| = {np.mean(np.abs(drift_bayes-drift_plain)):.3f}")
print(f"  corr(plain, Bayes) = {np.corrcoef(drift_plain, drift_bayes)[0,1]:.3f}")

# ── scatter plot ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,6))
ax.errorbar(drift_plain, drift_bayes, yerr=drift_sd_post, fmt='o', capsize=3, alpha=.7)
lim = [min(drift_plain.min(), drift_bayes.min())-0.02,
       max(drift_plain.max(), drift_bayes.max())+0.02]
ax.plot(lim, lim, 'k--', alpha=.5, label='y = x')
for i, s in enumerate(SUBJECTS):
    ax.annotate(s, (drift_plain[i], drift_bayes[i]), fontsize=7)
ax.set_xlabel("plain EZ drift (point estimate)")
ax.set_ylabel("Bayesian EZ drift (posterior mean ± SD)")
ax.set_title("Drift: plain EZ vs hierarchical Bayesian EZ")
ax.legend(); fig.tight_layout()
p = os.path.join(OUT_DIR, "ez_vs_bayes_drift.png")
fig.savefig(p, dpi=150)
print(f"\n  saved → {p}")

# save the posterior for later posterior-predictive use
with open(os.path.join(OUT_DIR, "bayes_ez_idata.pkl"), "wb") as f:
    pickle.dump({"drift_bayes": drift_bayes, "drift_sd": drift_sd_post,
                 "subjects": SUBJECTS}, f)

# ── ALSO save as a train.py-compatible prior file (nested mu/logvar) ───────────
# logvar = ln(posterior_variance) = ln(SD^2) = 2*ln(SD).  Floor SD so logvar
# doesn't blow up for very tight posteriors.
def to_logvar(sd):
    sd = max(float(sd), 1e-3)
    return float(2.0 * np.log(sd))

bayes_prior = {}
for i, s in enumerate(SUBJECTS):
    entry = {
        "v":   {"mu": float(drift_bayes[i]), "logvar": to_logvar(drift_sd_post[i])},
        "a":   {"mu": float(bound_bayes[i]), "logvar": to_logvar(bound_sd[i])},
        "ndt": {"mu": float(nondt_bayes[i]), "logvar": to_logvar(nondt_sd[i])},
    }
    # train.py reads [subj]['overall']; also fill easy/hard with the same values
    bayes_prior[s] = {"overall": entry, "easy": entry, "hard": entry}

bayes_prior_path = os.path.join(OUT_DIR, "ddm_priors_bayes.pkl")
with open(bayes_prior_path, "wb") as f:
    pickle.dump(bayes_prior, f)
print(f"  saved train.py-compatible Bayesian prior → {bayes_prior_path}")
print(f"  (point train.py's _prior_path at this file to use the Bayesian prior)")
print("✅  done. Posterior drift saved for posterior-predictive checks.")