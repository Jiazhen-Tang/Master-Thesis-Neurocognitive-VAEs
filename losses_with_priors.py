# ══════════════════════════════════════════════════════════════════════════════
# losses_with_priors.py — Updated KL loss using EZ-DDM informative priors
#
# Replace kl_standard for DDM parameters with kl_informed.
# EEG KL still uses kl_standard (N(0,I)).
# ══════════════════════════════════════════════════════════════════════════════

import torch


def kl_informed(mu, logvar, prior_mu, prior_logvar):
    """
    KL[ q(z|x) || N(prior_mu, prior_sigma²) ]
    
    Instead of pushing z toward N(0,I), pushes toward the EZ-DDM estimate.
    
    Parameters
    ----------
    mu          : (B, 1)  predicted mean from PosteriorDDM
    logvar      : (B, 1)  predicted log-variance from PosteriorDDM
    prior_mu    : float   EZ-DDM point estimate (e.g. v=1.2, a=1.5, ndt=0.3)
    prior_logvar: float   log of prior variance (e.g. log(0.1²) = -4.6)
    
    Returns
    -------
    kl : scalar
    """
    prior_mu     = torch.tensor(prior_mu,     device=mu.device, dtype=mu.dtype)
    prior_logvar = torch.tensor(prior_logvar, device=mu.device, dtype=mu.dtype)

    kl = 0.5 * (
        prior_logvar - logvar
        - 1
        + (logvar.exp() + (mu - prior_mu)**2) / prior_logvar.exp().clamp(min=1e-8)
    )
    return kl.mean()


def build_ddm_priors(ez_priors, subj_id, prior_sigma_v=0.5, prior_sigma_a=0.3, prior_sigma_ndt=0.1):
    """
    Extract EZ-DDM prior parameters for one subject.
    
    Parameters
    ----------
    ez_priors     : dict  loaded from ez_ddm_priors.pkl
    subj_id       : str   e.g. "01"
    prior_sigma_* : float uncertainty around EZ-DDM point estimate
                          larger = less informative prior
    
    Returns
    -------
    dict with prior_mu and prior_logvar for v, a, ndt
    """
    if subj_id not in ez_priors:
        # Fallback to weakly informative defaults
        return {
            "v":   {"mu": 1.0,  "logvar": torch.log(torch.tensor(prior_sigma_v**2)).item()},
            "a":   {"mu": 1.0,  "logvar": torch.log(torch.tensor(prior_sigma_a**2)).item()},
            "ndt": {"mu": 0.3,  "logvar": torch.log(torch.tensor(prior_sigma_ndt**2)).item()},
        }

    overall = ez_priors[subj_id]["overall"]

    return {
        "v":   {
            "mu":     overall["v"],
            "logvar": float(torch.log(torch.tensor(prior_sigma_v**2)).item()),
        },
        "a":   {
            "mu":     overall["a"],
            "logvar": float(torch.log(torch.tensor(prior_sigma_a**2)).item()),
        },
        "ndt": {
            "mu":     overall["ndt"],
            "logvar": float(torch.log(torch.tensor(prior_sigma_ndt**2)).item()),
        },
    }
