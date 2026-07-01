"""High-level orchestration of the dual-reservoir LSM pipeline (Section IV.B-C of the paper).

Ties together :class:`~src.lsm.LSM`, :class:`~src.stdp.AsymmetricSTDP`, and
the per-subject window tensors produced by ``scripts/build_dataset.py``
into three steps, driven end-to-end by ``scripts/pretrain_lsm.py``:

1. :func:`build_lsm_pair` -- construct the source and filter reservoirs.
2. :func:`pretrain_lsm_pair` -- unsupervised STDP pretraining pass over a
   subject list (STDP enabled).
3. :func:`extract_features_and_spikes` -- frozen forward pass (STDP disabled)
   that exports per-subject spike trains and population-level features.
"""

import gc
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .io_utils import load_subject_windows
from .lsm import LSM

torch.backends.cudnn.benchmark = True

__all__ = [
    "build_lsm_pair",
    "extract_features_and_spikes",
    "pretrain_lsm_pair",
    "stack_windows_segment",
]

def stack_windows_segment(windows: List[torch.Tensor], start: int, end: int) -> torch.Tensor:
    """
    Stack a contiguous slice of window tensors into a batch segment.

    The input windows are assumed to share the same temporal and channel shape.
    """
    seg = windows[start:end]
    return torch.stack(seg, dim=1)

def build_lsm_pair(
    stdp_params_source: dict,
    stdp_params_filter: dict,
    device: torch.device,
    n_layers = 77,
    win_strength_source: float = 1.35,
    win_strength_filter: float = 1.30,
    th_source: float = 20.0 * 0.80,
    th_filter: float = 20.0 * 0.85,
    inhibit : bool = True
) -> Tuple[LSM, LSM]:
    """Build the paired source/filter LSM reservoirs (Section IV.B of the paper).

    Both reservoirs share the same shape (``n_layers`` layers, one per
    gammatone/ERB channel) but use independent STDP hyperparameters and
    firing thresholds, since the source (excitation) and filter (vocal
    tract) signals have different temporal characteristics.

    Args:
        stdp_params_source, stdp_params_filter: kwargs forwarded to
            :class:`~src.stdp.AsymmetricSTDP` for each reservoir (e.g.
            ``tau_plus``, ``tau_minus``, ``A_plus``, ``A_minus``).
        device: torch device to place both reservoirs on.
        n_layers: number of tonotopic layers (= number of gammatone/ERB
            channels; 56 in the paper).
        win_strength_source, win_strength_filter: input-weight scaling for
            each reservoir.
        th_source, th_filter: firing thresholds for each reservoir.
        inhibit: whether a fraction of neurons are randomly inhibitory.

    Returns:
        Tuple ``(lsm_source, lsm_filter)``, both already moved to ``device``
        and with ``use_stdp=True``.
    """
    lsm_source = LSM(
        n_layers= n_layers,
        use_stdp=True,
        stdp_params=stdp_params_source,
        Win_strength=win_strength_source,
        th=th_source,
        inhibit=inhibit
    ).to(device)

    lsm_filter = LSM(
        n_layers= n_layers,
        use_stdp=True,
        stdp_params=stdp_params_filter,
        Win_strength=win_strength_filter,
        th=th_filter,
        inhibit=inhibit
    ).to(device)

    return lsm_source, lsm_filter

def pretrain_lsm_pair(
    lsm_source: LSM,
    lsm_filter: LSM,
    df: pd.DataFrame,
    subjects: List[str],
    project_path: str,
    data_path : str,
    device: torch.device,
    max_windows_per_chunk: int = 1,
    clamp_min: Optional[float] = -1.0,
    clamp_max: Optional[float] = 1.0,
) -> None:
    """
    Run one STDP-enabled pretraining pass over a subject list.

    For each subject, windows are loaded and processed in chunks of
    ``max_windows_per_chunk`` (the paper uses 1: each window is treated as an
    independent sample). STDP traces are reset before every chunk, since
    windows are not a continuous time series. Reservoir weights (``Wlsm``)
    are updated in place as a side effect of the forward pass
    (``apply_stdp=True``); this function returns nothing.

    Args:
        lsm_source, lsm_filter: reservoirs built by :func:`build_lsm_pair`,
            updated in place.
        df: dataset manifest (``dataset_log.csv``).
        subjects: subject IDs to iterate over.
        project_path: unused by this function (kept for API compatibility
            with callers that also need it for other purposes).
        data_path: directory containing the subjects' ``.pt`` tensor files.
        device: torch device to run the forward pass on. If a chunk fails to
            move to a CUDA device (OOM), it falls back to CPU for that chunk.
        max_windows_per_chunk: number of windows processed per forward call.
        clamp_min, clamp_max: bounds applied to reservoir weights after each
            STDP update.
    """

    for subj in tqdm(subjects, desc="[PRETRAIN] Subjects"):
        try:
            erb_list, gam_list, _ = load_subject_windows(df, subj, data_path)
        except Exception as e:
            print(f"[PRETRAIN][ERROR] load_subject_features failed for {subj}: {e}")
            continue

        B_total = len(erb_list)
        if B_total == 0:
            print(f"[PRETRAIN][WARN] no windows for subject {subj}")
            continue

        for start in range(0, B_total, max_windows_per_chunk):
            end = min(start + max_windows_per_chunk, B_total)
            # (T, B, C)
            try:
                cur_erb_chunk = stack_windows_segment(erb_list, start, end)
                cur_gam_chunk = stack_windows_segment(gam_list, start, end)
            except Exception as e:
                print(f"[PRETRAIN][ERROR] stack failed for {subj} chunk {start}:{end}: {e}")
                continue

            # move to device
            use_gpu_chunk = (device.type == "cuda")
            try:
                if use_gpu_chunk:
                    cur_erb_chunk = cur_erb_chunk.to(device, non_blocking=True)
                    cur_gam_chunk = cur_gam_chunk.to(device, non_blocking=True)
            except RuntimeError as e:
                print(f"[PRETRAIN][WARN] OOM for {subj} chunk {start}:{end}: {e}. Using CPU for this chunk.")
                torch.cuda.empty_cache()
                gc.collect()
                cur_erb_chunk = cur_erb_chunk.cpu()
                cur_gam_chunk = cur_gam_chunk.cpu()
                use_gpu_chunk = False

            # Reset STDP traces per window: windows are independent samples,
            # not one continuous sequence, so traces must not leak across them.
            for model in (lsm_filter, lsm_source):
                if hasattr(model, 'stdp') and model.stdp is not None:
                    try:
                        model.stdp.reset()
                    except Exception:
                        pass

            with torch.no_grad():
                _ = lsm_filter(cur_erb_chunk, apply_stdp=True,
                               clamp_min=clamp_min, clamp_max=clamp_max)
                _ = lsm_source(cur_gam_chunk, apply_stdp=True,
                               clamp_min=clamp_min, clamp_max=clamp_max)

            torch.cuda.empty_cache()
            gc.collect()

def extract_features_and_spikes(
    lsm_source: LSM,
    lsm_filter: LSM,
    df: pd.DataFrame,
    subjects: List[str],
    project_path: str,
    data_path: str,
    device: torch.device,
    max_windows_per_chunk: int = 1,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    save_spikes_dir: Optional[str] = None,
):
    """Frozen forward pass: export per-subject spike trains and population-level features.

    Runs both reservoirs with ``apply_stdp=False`` (weights are not
    modified) over every subject's windows, and for each subject:

    - averages reservoir activations over time and windows into a single
      ``(2N,)`` feature vector (``N`` = neurons per reservoir; features are
      the concatenation of filter- then source-reservoir activations), used
      as the input to the population-level features (AFR/TB/BI, etc.)
      computed downstream for classification;
    - concatenates the raw per-timestep spike trains from both reservoirs
      into a ``(T_total, 2N)`` array and, if ``save_spikes_dir`` is given,
      saves it (plus per-timestep labels and per-window lengths) to
      ``<save_spikes_dir>/<subject>_spikes.npz``.

    Subjects with 10 or fewer windows are skipped (insufficient data for
    reliable subject-level aggregation; matches the paper's exclusion
    criterion in Section IV).

    Args:
        lsm_source, lsm_filter: reservoirs with weights already pretrained
            (e.g. by :func:`pretrain_lsm_pair`); not modified by this call.
        df: dataset manifest (``dataset_log.csv``), typically the
            speech-filtered variant for the final classification features.
        subjects: subject IDs to iterate over.
        project_path: unused by this function (kept for API compatibility
            with callers that also need it for other purposes).
        data_path: directory containing the subjects' ``.pt`` tensor files.
        device: torch device to run the forward pass on.
        max_windows_per_chunk: number of windows processed per forward call.
        clamp_min, clamp_max: bounds applied to reservoir weights (STDP is
            disabled here, so these have no effect unless re-enabled).
        save_spikes_dir: if given, per-subject spike ``.npz`` files are
            written here (created if missing).

    Returns:
        Tuple ``(subject_features, subject_labels, subject_spikes_paths)``:
        dicts keyed by subject ID, mapping to the ``(2N,)`` feature vector,
        the integer label, and (if saved) the spikes ``.npz`` path.
    """

    subject_features = {}
    subject_labels = {}
    subject_spikes_paths = {}

    if save_spikes_dir is not None:
        os.makedirs(save_spikes_dir, exist_ok=True)

    for subj in tqdm(subjects, desc="[FEATURE] Subjects"):

        try:
            erb_list, gam_list, label = load_subject_windows(df, subj, data_path)
        except Exception as e:
            print(f"[ERROR] load windows failed for {subj}: {e}")
            continue

        B_total = len(erb_list)
        if B_total == 0 or B_total <= 10:
            print(f"[INFO] skipping {subj}, insufficient windows ({B_total})")
            continue

        N = lsm_source.N   # neurons per reservoir
        window_lengths = []   # effective T per window, for later reconstruction

        # Raw per-timestep spikes, accumulated across all of the subject's windows.
        filter_spikes = []
        source_spikes = []
        labels_raw = []

        # Running sum for the subject-level (2N,) feature vector.
        accum_feat_sum = None
        accum_feat_count = 0

        for start in range(0, B_total, max_windows_per_chunk):
            end = min(start + max_windows_per_chunk, B_total)
            B_chunk = end - start

            try:
                erb_chunk = stack_windows_segment(erb_list, start, end)  # (T_eff, 1, C)
                gam_chunk = stack_windows_segment(gam_list, start, end)
            except Exception as e:
                print(f"[ERROR] stack failed for {subj}, window {start}:{end}: {e}")
                continue

            erb_chunk = erb_chunk.to(device)
            gam_chunk = gam_chunk.to(device)

            # Forward pass only, STDP disabled: weights are frozen at this stage.
            with torch.no_grad():
                out_filter = lsm_filter(
                    erb_chunk, apply_stdp=False,
                    clamp_min=clamp_min, clamp_max=clamp_max
                )  # (T_eff, 1, N)

                out_source = lsm_source(
                    gam_chunk, apply_stdp=False,
                    clamp_min=clamp_min, clamp_max=clamp_max
                )  # (T_eff, 1, N)

            T_eff = out_filter.shape[0]
            window_lengths.append(T_eff)

            # Subject-level feature: mean activation over time, concatenated
            # across filter then source reservoirs -> (2N,).
            f_mean = out_filter.mean(dim=0)  # (1, N)
            s_mean = out_source.mean(dim=0)  # (1, N)

            feat = torch.cat([f_mean, s_mean], dim=1).squeeze(0).cpu().numpy()  # (2N,)

            if accum_feat_sum is None:
                accum_feat_sum = feat.copy()
            else:
                accum_feat_sum += feat
            accum_feat_count += 1

            # Raw per-timestep spikes for this window, kept for the .npz export.
            fr = out_filter.squeeze(1).cpu().numpy()   # (T_eff, N)
            sr = out_source.squeeze(1).cpu().numpy()   # (T_eff, N)

            filter_spikes.append(fr)
            source_spikes.append(sr)

            labels_raw.extend([label] * T_eff)

        subj_feat = accum_feat_sum / accum_feat_count
        subject_features[subj] = subj_feat
        subject_labels[subj] = label

        filter_raw_full = np.vstack(filter_spikes)        # (T_tot, N)
        source_raw_full = np.vstack(source_spikes)        # (T_tot, N)
        concat_raw_full = np.concatenate([filter_raw_full, source_raw_full], axis=1)
        labels_raw_full = np.array(labels_raw, dtype=int)
        window_lengths = np.array(window_lengths, dtype=int)

        assert filter_raw_full.shape[0] == labels_raw_full.shape[0]
        assert source_raw_full.shape[0] == labels_raw_full.shape[0]

        if save_spikes_dir is not None:
            out_path = os.path.join(save_spikes_dir, f"{subj}_spikes.npz")
            np.savez_compressed(
                out_path,
                filter_raw=filter_raw_full,
                source_raw=source_raw_full,
                concat_raw=concat_raw_full,
                labels_raw=labels_raw_full,
                window_lengths=window_lengths,
            )
            subject_spikes_paths[subj] = out_path

    return subject_features, subject_labels, subject_spikes_paths
