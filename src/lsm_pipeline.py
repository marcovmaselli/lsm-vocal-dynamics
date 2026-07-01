"""High-level helpers for LSM pretraining and feature extraction."""

import gc
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .io_utils import load_subject_features
from .lsm import LSM

torch.backends.cudnn.benchmark = True

__all__ = [
    "build_lsm_pair",
    "extract_features_and_spikes",
    "load_subject_windows",
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


def load_subject_windows(
    df: pd.DataFrame,
    subj: str,
    project_path: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    """
    Load ERB and gammatone windows for a single subject.

    Args:
        df: Dataset manifest.
        subj: Subject identifier.
        project_path: Base path used to resolve stored tensors.

    Returns:
        Two lists of ``(T, C)`` tensors and the subject label.
    """
    feats, label = load_subject_features(df, subj, project_path)
    if not feats:
        return [], [], label

    erb_list, gam_list = [], []
    for erb, gam in feats:
        if not isinstance(erb, torch.Tensor):
            erb = torch.tensor(erb)
        if not isinstance(gam, torch.Tensor):
            gam = torch.tensor(gam)
        erb_list.append(erb.t().contiguous().float())   # (T, C)
        gam_list.append(gam.t().contiguous().float())   # (T, C)

    return erb_list, gam_list, label

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
    """Build the pair of LSM models used for gammatone and ERB features."""
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

    The traces are reset for each chunk because the windows are treated as
    independent samples rather than as one continuous time series.
    """

    for subj in tqdm(subjects, desc="[PRETRAIN] Subjects"):
        try:
            erb_list, gam_list, _ = load_subject_windows(df, subj, data_path)
        except Exception as e:
            print(f"[PRETRAIN][ERROR] load_subject_features failed for {subj}: {e}")
            continue

        B_total = len(erb_list)
        if B_total == 0:
            print(f"[PRETRAIN][WARN] nessuna finestra per soggetto {subj}")
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

            # reset STDP TRACE per finestra (dati non sequenziali)
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

# def extract_features_and_spikes(
#     lsm_source: LSM,
#     lsm_filter: LSM,
#     df: pd.DataFrame,
#     subjects: List[str],
#     project_path: str,
#     data_path: str,
#     device: torch.device,
#     max_windows_per_chunk: int = 1,
#     clamp_min: Optional[float] = -1.0,
#     clamp_max: Optional[float] = 1.0,
#     save_spikes_dir: Optional[str] = None,
# ) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, str]]:
#     """
#     Estrarre:
#     - feature per soggetto (media su tempo e finestre) → (2N,)
#     - spike medi nel tempo per soggetto (T, 2N), salvati anche su disco se richiesto.

#     Ritorna:
#         subject_features: subj -> (2N,) (numpy)
#         subject_labels:   subj -> label
#         subject_spikes_paths: subj -> path file npz (se save_spikes_dir non è None), altrimenti dict vuoto
#     """

#     subject_features: Dict[str, np.ndarray] = {}
#     subject_labels: Dict[str, int] = {}
#     subject_spikes_paths: Dict[str, str] = {}

#     if save_spikes_dir is not None:
#         os.makedirs(save_spikes_dir, exist_ok=True)

#     for subj in tqdm(subjects, desc="[FEATURE] Subjects"):
#         try:
#             erb_list, gam_list, label = load_subject_windows(df, subj, data_path)
#         except Exception as e:
#             print(f"[FEATURE][ERROR] load_subject_features failed for {subj}: {e}")
#             continue

#         B_total = len(erb_list)
#         if B_total <= 10:
#             print(f"[FEATURE][INFO] {subj} discarded (not enough windows: {B_total})")
#             continue
#         if B_total == 0:
#             print(f"[FEATURE][WARN] nessuna finestra per soggetto {subj}")
#             continue

#         # per feature globali
#         accum_feat_sum = None
#         accum_count = 0

#         # per spikes medi nel tempo
#         accum_spk_sum_filter = None  # (T, N)
#         accum_spk_sum_source = None  # (T, N)
#         spike_window_count = 0

#         for start in range(0, B_total, max_windows_per_chunk):
#             end = min(start + max_windows_per_chunk, B_total)
#             B_chunk = end - start

#             try:
#                 cur_erb_chunk = stack_windows_segment(erb_list, start, end)  # (T, B, C)
#                 cur_gam_chunk = stack_windows_segment(gam_list, start, end)
#             except Exception as e:
#                 print(f"[FEATURE][ERROR] stack failed for {subj} chunk {start}:{end}: {e}")
#                 continue

#             use_gpu_chunk = (device.type == "cuda")
#             try:
#                 if use_gpu_chunk:
#                     cur_erb_chunk = cur_erb_chunk.to(device, non_blocking=True)
#                     cur_gam_chunk = cur_gam_chunk.to(device, non_blocking=True)
#             except RuntimeError as e:
#                 print(f"[FEATURE][WARN] OOM for {subj} chunk {start}:{end}: {e}. Using CPU for this chunk.")
#                 torch.cuda.empty_cache()
#                 gc.collect()
#                 cur_erb_chunk = cur_erb_chunk.cpu()
#                 cur_gam_chunk = cur_gam_chunk.cpu()
#                 use_gpu_chunk = False

#             # STDP OFF per estrazione feature
#             with torch.no_grad():
#                 out_filter = lsm_filter(cur_erb_chunk, apply_stdp=False,
#                                         clamp_min=clamp_min, clamp_max=clamp_max)  # (T, B, N)
#                 out_source = lsm_source(cur_gam_chunk, apply_stdp=False,
#                                         clamp_min=clamp_min, clamp_max=clamp_max)  # (T, B, N)

#             # -------------------------
#             # Feature soggetto (2N,)
#             # -------------------------
#             # media temporale per finestra -> (B, N)
#             f_mean_chunk = out_filter.mean(dim=0)   # (B, N)
#             s_mean_chunk = out_source.mean(dim=0)   # (B, N)
#             feat_chunk = torch.cat([f_mean_chunk, s_mean_chunk], dim=1)  # (B, 2N)

#             feat_chunk_sum = feat_chunk.sum(dim=0).to('cpu').detach().numpy()  # (2N,)
#             chunk_count = int(feat_chunk.shape[0])

#             if accum_feat_sum is None:
#                 accum_feat_sum = feat_chunk_sum.copy()
#             else:
#                 accum_feat_sum += feat_chunk_sum
#             accum_count += chunk_count

#             # -------------------------
#             # Spike raw per soggetto
#             # -------------------------
#             # out_filter, out_source shape: (T, B, N)
#             spk_filter_raw_chunk = out_filter.cpu().numpy()   # (T, B, N)
#             spk_source_raw_chunk = out_source.cpu().numpy()   # (T, B, N)

#             # li appiattisco sul batch → (T*B, N)
#             spk_filter_raw_chunk = spk_filter_raw_chunk.reshape(-1, spk_filter_raw_chunk.shape[-1])
#             spk_source_raw_chunk = spk_source_raw_chunk.reshape(-1, spk_source_raw_chunk.shape[-1])

#             # accumulo
#             if 'accum_filter_raw' not in locals():
#                 accum_filter_raw = [spk_filter_raw_chunk]
#                 accum_source_raw = [spk_source_raw_chunk]
#             else:
#                 accum_filter_raw.append(spk_filter_raw_chunk)
#                 accum_source_raw.append(spk_source_raw_chunk)

#             spike_window_count += B_chunk

#             torch.cuda.empty_cache()
#             gc.collect()

#         if accum_count == 0 or spike_window_count == 0:
#             print(f"[FEATURE][WARN] no frames processed for {subj} (accum_count={accum_count}, spike_count={spike_window_count})")
#             continue

#         # feature globali per soggetto
#         subj_feat = (accum_feat_sum / float(accum_count))
#         subject_features[subj] = subj_feat
#         subject_labels[subj] = label

#         # concatena tutti i RAW spikes
#         filter_raw_full = np.concatenate(accum_filter_raw, axis=0)   # (T_tot, N)
#         source_raw_full = np.concatenate(accum_source_raw, axis=0)   # (T_tot, N)
#         concat_raw = np.concatenate([filter_raw_full, source_raw_full], axis=1)  # (T_tot, 2N)

#         if save_spikes_dir is not None:
#             out_path = os.path.join(save_spikes_dir, f"{subj}_spikes.npz")
#             np.savez_compressed(
#                 out_path,
#                 filter_raw=filter_raw_full,
#                 source_raw=source_raw_full,
#                 concat_raw=concat_raw,
#             )
#             subject_spikes_paths[subj] = out_path

#         # pulizia memoria
#         del accum_filter_raw, accum_source_raw
#         torch.cuda.empty_cache()
#         gc.collect()

#     return subject_features, subject_labels, subject_spikes_paths

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
    """
    Estrae:
        - feature per soggetto (media temporale delle attivazioni)
        - spike RAW per soggetto (T_tot, 2N)
        - label per ogni spike (T_tot,)
        - lunghezza per finestra (window_lengths)

    Salva in npz per ogni soggetto:
        filter_raw, source_raw, concat_raw, labels_raw, window_lengths

    Ritorna:
        subject_features : dict(subj -> (2N,))
        subject_labels   : dict(subj -> int)
        subject_spikes_paths : dict(subj -> path)
    """

    subject_features = {}
    subject_labels = {}
    subject_spikes_paths = {}

    if save_spikes_dir is not None:
        os.makedirs(save_spikes_dir, exist_ok=True)

    for subj in tqdm(subjects, desc="[FEATURE] Subjects"):

        # ------------------------------------------------------
        # Caricamento finestre del soggetto
        # ------------------------------------------------------
        try:
            erb_list, gam_list, label = load_subject_windows(df, subj, data_path)
        except Exception as e:
            print(f"[ERROR] load windows failed for {subj}: {e}")
            continue

        B_total = len(erb_list)
        if B_total == 0 or B_total <= 10:
            print(f"[INFO] skipping {subj}, insufficient windows ({B_total})")
            continue

        N = lsm_source.N   # numero neuroni
        window_lengths = []   # lunghezze T_eff_per_window

        # accumulo spikes
        filter_spikes = []
        source_spikes = []
        labels_raw = []

        # accumulo features globali
        accum_feat_sum = None
        accum_feat_count = 0

        # ------------------------------------------------------
        # Loop su tutte le finestre (max_windows_per_chunk=1)
        # ------------------------------------------------------
        for start in range(0, B_total, max_windows_per_chunk):
            end = min(start + max_windows_per_chunk, B_total)
            B_chunk = end - start

            try:
                erb_chunk = stack_windows_segment(erb_list, start, end)  # (T_eff, 1, C)
                gam_chunk = stack_windows_segment(gam_list, start, end)
            except Exception as e:
                print(f"[ERROR] stack failed for {subj}, window {start}:{end}: {e}")
                continue

            # move to device
            erb_chunk = erb_chunk.to(device)
            gam_chunk = gam_chunk.to(device)

            # ------------------------------------------------------
            # Forward LSM (STDP OFF → solo estrazione)
            # ------------------------------------------------------
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

            # ---------------------------
            # Feature soggetto
            # ---------------------------
            # media su T_eff → (1, N)
            f_mean = out_filter.mean(dim=0)  # (1, N)
            s_mean = out_source.mean(dim=0)  # (1, N)

            feat = torch.cat([f_mean, s_mean], dim=1).squeeze(0).cpu().numpy()  # (2N,)

            if accum_feat_sum is None:
                accum_feat_sum = feat.copy()
            else:
                accum_feat_sum += feat
            accum_feat_count += 1

            # ---------------------------
            # Spikes RAW (T_eff, N)
            # ---------------------------
            fr = out_filter.squeeze(1).cpu().numpy()   # (T_eff, N)
            sr = out_source.squeeze(1).cpu().numpy()   # (T_eff, N)

            filter_spikes.append(fr)
            source_spikes.append(sr)

            # label per ogni frame
            labels_raw.extend([label] * T_eff)

        # ------------------------------------------------------
        # Salvataggio soggetto
        # ------------------------------------------------------
        subj_feat = accum_feat_sum / accum_feat_count
        subject_features[subj] = subj_feat
        subject_labels[subj] = label

        filter_raw_full = np.vstack(filter_spikes)        # (T_tot, N)
        source_raw_full = np.vstack(source_spikes)        # (T_tot, N)
        concat_raw_full = np.concatenate([filter_raw_full, source_raw_full], axis=1)
        labels_raw_full = np.array(labels_raw, dtype=int)
        window_lengths = np.array(window_lengths, dtype=int)

        # check consistenza
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
