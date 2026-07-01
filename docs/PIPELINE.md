# Pipeline notes

This note maps the paper's methodology (Section IV) onto the code in this
repository, and records the specific choices used to produce the published
results.

## 1. Two dataset variants

`scripts/build_dataset.py` is run twice, producing two tensor caches from the
same raw recordings:

- **All-windows** (`--all-windows`): every 2 s window is kept, including
  silence, background noise, and other speakers. Used only for unsupervised
  reservoir pretraining.
- **Speech-only** (default): ELAN third-party annotations plus Silero VAD are
  combined to keep only windows where the child is talking. Used for feature
  extraction, classification, and evaluation.

Both variants share the same LPC source/filter decomposition and 56-channel
ERB-spaced gammatone encoding (`src/preprocessing.py`, `src/features.py`).

## 2. Unsupervised STDP pretraining

`scripts/pretrain_lsm.py`:

1. Builds the paired source/filter LSM reservoirs (`src/lsm.py`,
   `build_lsm_pair` in `src/lsm_pipeline.py`).
2. Runs 3 STDP epochs (`pretrain_lsm_pair`) over the **all-windows** split.
   STDP is enabled during this phase (`src/lsm_pipeline.py`,
   `src/stdp.py`).
3. Freezes the reservoir weights and calls `extract_features_and_spikes`
   on the **speech-only** split (STDP disabled), saving per-subject spike
   trains and population-level features.

This mixed-dataset strategy (train on everything, extract on speech-only) is
the key design choice described in Section IV of the paper: it lets the
reservoirs learn generic temporal dynamics from the full acoustic variability
of the recordings, while restricting the supervised analysis to windows
attributable to the child.

## 3. Feature extraction and classification

Population-level and layer-wise features (AFR, TB, BI, and their layer-wise
counterparts) are computed from the exported spike trains and aggregated at
the subject level. From `subject_features.pkl` and the per-subject spike
files produced by `scripts/pretrain_lsm.py`, a stratified 5-fold PCA + SVM
classification (Section IV.D of the paper) reproduces Table 1 and Fig. 3.
That classification/plotting step is not shipped as code in this repository.

## 4. What is not in this repository

Raw audio, ELAN annotations, and subject metadata/registries are clinical
data collected under IRCCS Stella Maris' ethics approval and are not
redistributed here (see the top-level README). Scripts that reproduce the
pipeline expect you to supply your own `--audio-dirs` / `--labels-csv`
pointing at data you are authorized to use.
