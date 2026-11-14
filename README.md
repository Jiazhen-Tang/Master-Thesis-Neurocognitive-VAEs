# Evaluating the Neurocognitive Variational Autoencoder in an Abstract Reasoning Task

Code for the MSc thesis *Evaluating the Neurocognitive Variational Autoencoder in an
Abstract Reasoning Task: A Conceptual Replication* (Jiazhen Tang, University of Amsterdam, 2026).

The scripts fit the Neurocognitive Variational Autoencoder (NCVA; Vo et al., 2024) to
stimulus-locked EEG and choice-reaction-time data from 19 participants, with one model
pooled across participants.

## Structure

```
config.py               paths, epoching, model and training settings
pipeline_utils.py       reading, epoching and merging EEG with behaviour
data_loader_pooled.py   loads the 64-channel tensors of all participants
model.py                network blocks
model_pooled.py         pooled NCVA with participant embedding
losses.py               reconstruction, Wiener first-passage-time and KL losses
losses_with_priors.py   KL term against the informed priors
svd_filter.py           SVD spatial filter
train_pooled.py         four-stage training

preprocessing/          from preprocessed EEG to epoch tensors
priors/                 EZ-DDM and hierarchical Bayesian EZ-DDM priors
analysis/               figures and statistics reported in the thesis
```

## Order of execution

```
# 1. Preprocessing (needs access to the raw data, see below)
python preprocessing/01_erp_baseline.py
python preprocessing/02_data_prep.py
python preprocessing/fix_choice_labels.py
python preprocessing/save_ch_names.py
python preprocessing/build_clean_eeg.py      # or build_eeg_batch.py, which works from the local tensors

# 2. Priors
python priors/ez_ddm.py                      # closed-form EZ-DDM
python priors/convert_ez_to_prior.py
python priors/bayesian_ez_compare.py         # hierarchical Bayesian EZ-DDM (PyMC), used in training

# 3. Training
python train_pooled.py

# 4. Analyses
python analysis/eval_pooled_waveforms.py     # waveform fidelity, from EEG and from reaction time
python analysis/H1.py                        # DDM parameters against reaction time
python analysis/H3.py                        # drift perturbation
python analysis/DDM_CHECK.py                 # adequacy of the drift-diffusion model
```

`check_alignment.py` checks whether each stored reaction time belongs to the EEG epoch
it is paired with (see Known issues).

## Known issues

These issues were found while preparing this repository, after the thesis had been
submitted. The code is published as it was run, so that the reported results can be
reproduced.

**1. Bridger 2 was not trained in the pooled model.**
In stage 4 of `train_pooled.py` the loss contains only the KL term for Bridger 1
(reaction time to z_C). The KL term that aligns Bridger 2 (z_C to z_N) with the
EEG-derived z_N is missing, so Bridger 2 stayed at its initial weights. Bridger 2 is used
for the waveforms generated from reaction time (Section 3.3) and for the drift-rate
perturbation analysis (Section 3.4, Figure 6). When Bridger 2 was trained afterwards with
the missing term, the P300 still did not follow the predicted pattern, and the N200-like
deflection remained and became clearer.

**2. Reaction times are paired with the wrong EEG epoch in part of the data.**
In `pipeline_utils.py` the EEG trials are renumbered after time-out trials are removed,
while the behavioural rows keep their original index. After each time-out, the remaining
trials of that session are merged with the behavioural row of a neighbouring trial.
`check_alignment.py` measures the duration of every trial from the EEG alone and compares
it with the stored reaction time. Of 5,932 trials, 848 (14%) carry the reaction time of a
neighbouring trial. Seven participants are unaffected (02, 04, 10, 11, 18, 20, 22). In
these seven, the within-participant correlation between EEG-inferred drift rate and
reaction time is also close to zero (median −.08 on held-out trials). The
participant-level result and the diffusion-model adequacy check use behavioural summaries
only and are not affected.

## Requirements

Python 3.10 and the packages in `requirements.txt`.

## Data

The EEG and behavioural data are not part of this repository. `config.py` points to the
location of the preprocessed data (`DATA_PATH`, `BEH_PATH`). All intermediate files and
results are written to `~/Desktop/NCVA_results` (`LOCAL_BASE`).

## Reference

Vo, K., Sun, Q. J., Nunez, M. D., Vandekerckhove, J., & Srinivasan, R. (2024). Deep latent
variable joint cognitive modeling of neural signals and human behavior. *NeuroImage, 291*, 120559.
