# Evaluating Vocal Dynamics in Children using a Neuromorphic Approach

Code accompanying the paper *"Evaluating Vocal Dynamics in Children using a
Neuromorphic Approach"* (Maselli et al.), which investigates whether
neuromorphic temporal representations of spontaneous child speech can
discriminate typically developing (TD) children from children with cerebral
palsy (CP), using audio collected during robot-mediated storytelling.

The pipeline: audio is decomposed into source/filter components (Linear
Predictive Analysis), encoded through a cochlea-inspired gammatone
filterbank, processed by a pair of unsupervised Liquid State Machines (LSM)
trained with Spike-Timing-Dependent Plasticity (STDP), summarized into
population-level and layer-wise spiking features, and classified per-subject
with PCA + SVM under stratified 5-fold cross-validation.

## Repository structure

```
src/                  Core library: I/O, preprocessing, gammatone features,
                      LSM reservoirs, STDP, spike/feature extraction
scripts/
  build_dataset.py    Windowing + source/filter + gammatone encoding
  pretrain_lsm.py     Unsupervised STDP pretraining, then frozen
                      feature/spike extraction
docs/PIPELINE.md      How the paper's methodology maps onto the code
```

## Data availability

Audio recordings, ELAN annotations, and subject metadata are clinical data
collected under IRCCS Stella Maris' ethics approval and involving minors.
**They are not included in this repository and cannot be shared.** The
scripts here operate on your own data: point `scripts/build_dataset.py` at
your own audio directories and a `subject_id,label` CSV (see
`docs/PIPELINE.md` and the docstrings in each script for the expected
layout). Model weights, extracted features, and figures derived from the
real cohort are likewise not distributed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for practical runtimes on full cohorts (LSM
simulation and STDP updates are run per audio window); everything also runs
on CPU for smaller subsets. Silero VAD is fetched on first use via
`torch.hub` (requires internet access).

## Usage

```bash
# 1. Speech-filtered windows, used for classification
python scripts/build_dataset.py \
    --audio-dirs /path/to/cohort_a /path/to/cohort_b \
    --labels-csv /path/to/labels.csv \
    --output-dir data/80_4000_Hz_56_channels

# 2. All-windows, used for unsupervised reservoir pretraining
python scripts/build_dataset.py \
    --audio-dirs /path/to/cohort_a /path/to/cohort_b \
    --labels-csv /path/to/labels.csv \
    --output-dir data/all_windows_80_4000_Hz_56_channels \
    --all-windows

# 3. Pretrain the reservoirs and export subject-level spikes/features
python scripts/pretrain_lsm.py \
    --extract-data-dir data/80_4000_Hz_56_channels \
    --train-data-dir data/all_windows_80_4000_Hz_56_channels \
    --output-dir results/stella_maris_pretrain
```

`results/stella_maris_pretrain/subject_features.pkl` and the per-subject
spike files under `results/stella_maris_pretrain/spikes/` then feed the
subject-level PCA + SVM classification (stratified 5-fold CV) described in
Section IV.D of the paper, which reproduces Table 1 and Fig. 3.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper if you use this
code or build on this pipeline.

## License

Code is released under the [MIT License](LICENSE). This does not extend to
any data, which is not part of this repository.
