"""Audio I/O helpers and dataset serialization utilities."""

import os
from typing import List, Tuple

import pandas as pd
import pympi
import torch
import torchaudio
from torchaudio.transforms import Resample

__all__ = [
    "get_annotations_from_file",
    "load_audio_torch",
    "load_subject_features",
    "load_subject_windows",
    "save_chunks_with_csv_pd",
]


def load_audio_torch(audio_path: str,
                     sr: int = 16000,
                     mono: bool = True,
                     **kwargs) -> Tuple[torch.Tensor, float]:
    """
    Load an audio file with torchaudio and optionally resample it.

    Args:
        audio_path : str
            Path to audio file
        sr : int
            Target sample rate
        mono : bool
            Convert to mono if True
        **kwargs : any
            Reserved for future parameters

    Returns:
        waveform : torch.Tensor
            1D float32 tensor [-1, 1]
        duration : float
            Duration in seconds
    """
    waveform, orig_sr = torchaudio.load(audio_path)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if orig_sr != sr:
        resampler = Resample(orig_freq=orig_sr, new_freq=sr)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)
    duration = waveform.shape[-1] / sr
    return waveform, duration


def get_annotations_from_file(eaf_file: str,
                              tier_name: str = 'default',
                              sr: int = 16000,
                              **kwargs) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Read ELAN .eaf file and return annotations in seconds and sample indices.

    Args:
        eaf_file : str
            Path to ELAN .eaf file
        tier_name : str
            Tier to read
        sr : int
            Sample rate for conversion to sample indices

    Returns:
        annotations_times : List of [start_s, end_s, text]
        annotations_indexes : List of [start_idx, end_idx, text]
    """
    if not os.path.exists(eaf_file):
        return [], []

    elan = pympi.Elan.Eaf(eaf_file)
    annotations_ms = elan.get_annotation_data_for_tier(tier_name)

    annotations_times = [[a / 1000.0, b / 1000.0, text] for a, b, text in annotations_ms]
    annotations_indexes = [[int(round(a / 1000.0 * sr)), int(round(b / 1000.0 * sr)), text] for a, b, text in annotations_ms]

    return annotations_times, annotations_indexes


def save_chunks_with_csv_pd(chunks: List[torch.Tensor],
                            start_end_times: List[List[float]],
                            subject_id: str,
                            save_dir: str,
                            csv_path: str,
                            label: str = 'typical',
                            prefix: str = 'chunk',
                            overwrite: bool = False,
                            **kwargs):
    """
    Save audio chunks as .pt files and append metadata to CSV.

    Args:
        chunks : list of torch.Tensor
            Audio chunks (1D tensors)
        start_end_times : list of [start_s, end_s]
            Corresponding start/end times in seconds
        subject_id : str
            ID of the subject
        save_dir : str
            Folder to save .pt chunk files
        csv_path : str
            Path to CSV metadata file
        label : str
            Optional label for each chunk
        prefix : str
            Filename prefix
        overwrite : bool
            If True, overwrite CSV; if False, append removing duplicates
        **kwargs : any
            Reserved for future use (e.g., additional metadata fields)
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for i, (chunk, se) in enumerate(zip(chunks, start_end_times)):
        filename = f"{prefix}_{subject_id}_{i:03d}.pt"
        path = os.path.join(save_dir, filename)
        torch.save({'audio_chunk': chunk, 'start_end': se, 'subject_id': subject_id}, path)
        rows.append([subject_id, path, label, se[0], se[1]])

    df_new = pd.DataFrame(rows, columns=['subject_id', 'file_path', 'label', 'start_s', 'end_s'])

    if not overwrite and os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
        # Rimuove duplicati basandosi sul percorso del file
        df_new = df_new.drop_duplicates(subset='file_path', keep='first')

    df_new.to_csv(csv_path, index=False)


def load_subject_features(df: pd.DataFrame, subject: str, data_path: str):
    """Load ERB and gammatone tensors for all windows of one subject."""
    features = []
    subj_rows = df[df['subject_id'] == subject]
    for _, row in subj_rows.iterrows():
        gammatone = torch.load(os.path.join(data_path, row['tensor_gammatone_path'].split('/')[-1]))
        erb = torch.load(os.path.join(data_path, row['tensor_erb_path'].split('/')[-1]))
        features.append([gammatone, erb])
    return features, subj_rows['label'].iloc[0]

def load_subject_windows(
    df: pd.DataFrame,
    subj: str,
    project_path: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    """
    Load ERB and gammatone windows for one subject.

    Returns:
        erb_list: List of tensors shaped ``(T, C)``.
        gam_list: List of tensors shaped ``(T, C)``.
        label: Subject-level label.
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