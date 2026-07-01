"""Signal preprocessing utilities used before feature extraction."""

from typing import List, Tuple

import torch

torch.set_num_threads(1)
import torchaudio.functional as F
import numpy as np

__all__ = [
    "extract_windows_with_overlap",
    "find_non_silent_vad_silero",
    "levinson_durbin",
    "linear_predictive_analysis",
    "lowpass_filter_torch",
    "remove_other_people_from_segments",
]

def lowpass_filter_torch(waveform: torch.Tensor, sr: int = 16000, highcut: float = 7900.0, **kwargs) -> torch.Tensor:
    """Apply a low-pass filter using ``torchaudio.functional.lowpass_biquad``."""
    if waveform.ndim == 1:
        w = waveform.unsqueeze(0)
    else:
        w = waveform
    filtered = F.lowpass_biquad(w, sample_rate=sr, cutoff_freq=highcut)
    return filtered.squeeze(0) if waveform.ndim == 1 else filtered

def find_non_silent_vad_silero(sig: torch.Tensor,
                               sr: int = 16000,
                               min_silence_duration: float = 0.05,
                              **kwargs) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Detect speech segments with the Silero VAD model.

    Args:
        sig: 1D waveform tensor in ``[-1, 1]``.
        sr: Sample rate.
        min_silence_duration: Minimum silent gap, in seconds, used to merge
            adjacent speech regions.

    Returns:
        A tuple containing time intervals and sample-index intervals.
    """
    if sig.ndim > 1:
        sig = sig.mean(dim=0)
    sig = sig.contiguous().float()

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils
    speech_timestamps = get_speech_timestamps(sig, model, return_seconds=True)
    non_silent_time = []
    non_silent_idx = []
    for ts in speech_timestamps:
        start_s = ts.get('start_time', ts.get('start', None))
        end_s   = ts.get('end_time', ts.get('end', None))
        if start_s is None or end_s is None:
            continue
        non_silent_time.append([float(start_s), float(end_s)])
        non_silent_idx.append([int(round(float(start_s) * sr)), int(round(float(end_s) * sr))])
    merged_time = []
    merged_idx = []
    for s, e in non_silent_time:
        if not merged_time:
            merged_time.append([s, e])
        else:
            if s - merged_time[-1][1] <= min_silence_duration:
                merged_time[-1][1] = e
            else:
                merged_time.append([s, e])
    merged_idx = [[int(round(s * sr)), int(round(e * sr))] for s, e in merged_time]
    return merged_time, merged_idx

def remove_other_people_from_segments(non_silent_time: List[List[float]], annotations_time: List[List[float]], sr: int = 16000):
    """Remove annotated overlap from VAD speech regions.

    Args:
        non_silent_time: VAD regions in seconds.
        annotations_time: ELAN regions to subtract from the VAD regions.
        sr: Sample rate used to derive sample indices.

    Returns:
        Remaining regions expressed both in seconds and sample indices.
    """
    remaining_times = []
    remaining_idx = []
    ann_sorted = sorted([[a, b] for a, b, _ in annotations_time], key=lambda x: x[0])
    for ns_time in non_silent_time: 
        ns_start, ns_end = ns_time
        residuals = [[ns_start, ns_end]]
        for ann_start, ann_end in ann_sorted:
            new_residuals = []
            for r_start, r_end in residuals:
                if ann_end <= r_start or ann_start >= r_end:
                    new_residuals.append([r_start, r_end])
                else:
                    if ann_start > r_start:
                        new_residuals.append([r_start, min(ann_start, r_end)])
                    if ann_end < r_end:
                        new_residuals.append([max(ann_end, r_start), r_end])
            residuals = new_residuals
            if not residuals:
                break
        for r_start, r_end in residuals:
            if r_end - r_start > 1e-4:
                remaining_times.append([r_start, r_end])
                remaining_idx.append([int(round(r_start * sr)), int(round(r_end * sr))])
    return remaining_times, remaining_idx

def extract_windows_with_overlap(signal: torch.Tensor,
                                 remaining_idxs: List[Tuple[int,int]],
                                 sr: int = 16000,
                                 win_s: float = 2,
                                 hop_s: float = 0.5,
                                 min_fraction: float = 0.8,
                                 **kwargs):
    """
    Parameters
    ----------
    signal : torch.Tensor
        1D tensor with the signal samples.
    remaining_idxs : list of (start_idx, end_idx)
        intervals to check overlap against (in samples, inclusive start, exclusive end recommended).
    sr : int
        sample rate (samples per second) (default 16k).    
    win_s : float
        window length in seconds (default 0.03).
    hop_s : float
        hop length in seconds (default 0.05).
    min_fraction : float
        minimum fraction of the window that must overlap an interval to keep it (default 0.8).
    
    Returns
    -------
    signal_windows : List[torch.Tensor]
        list of window tensors kept.
    windows_meta : List[dict]
        list of metadata dicts for each kept window: {'start': int, 'end': int, 'overlap_frac': float}
    """
    win_len = int(round(win_s * sr))
    hop = int(round(hop_s * sr))
    L = signal.shape[0]
    rem = []
    for a, b in remaining_idxs:
        s = max(0, int(a))
        e = min(L, int(b))
        if e > s:
            rem.append((s, e))
    if not rem:
        return [], []
    signal_windows = []
    windows_meta = []
    for win_start in range(0, L - win_len + 1, hop):
        win_end = win_start + win_len
        best_overlap_frac = 0.0
        for rem_start, rem_end in rem:
            inter_start = max(win_start, rem_start)
            inter_end = min(win_end, rem_end)
            inter = max(0, inter_end - inter_start)
            frac = inter / win_len
            if frac > best_overlap_frac:
                best_overlap_frac = frac
            if best_overlap_frac >= min_fraction:
                break
        if best_overlap_frac >= min_fraction:
            window_tensor = signal[win_start:win_end].clone()
            signal_windows.append(window_tensor)
            windows_meta.append({
                'start': win_start,
                'end': win_end,
                'overlap_frac': best_overlap_frac
            })
    return signal_windows, windows_meta


def levinson_durbin(r: torch.Tensor, order: int, eps = 1e-9, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Solve the Levinson-Durbin recursion for one autocorrelation vector.
    """
    assert r.ndim == 1
    assert r.shape[0] >= order + 1

    a = torch.zeros(order + 1, dtype=r.dtype, device=r.device)
    a[0] = 1.0
    e = r[0].clone()
    if e == 0:
        return a, e
    for i in range(1, order + 1):
        acc = torch.dot(a[1:i],  torch.flip(r[1:i], dims=[0]))
        k = -(r[i] + acc) / e
        a_prev = a[:i].clone()
        if i > 1:
            a[1:i] = a_prev[1:i] + k * torch.flip(a_prev[1:i], dims=[0])
        a[i] = k
        e = e * (1.0 - k * k)
        e = torch.clamp(e, min=eps)
        if e <= 0:
            e = torch.tensor(eps, dtype=r.dtype, device=r.device)
    return a, e


def linear_predictive_analysis(signal: torch.Tensor,
                                       sr: int = 16000,
                                       order: int = 16,
                                       win_ms: int = 30,
                                       hop_ms: int = 5,
                                       apply_window: bool = True,
                                       window_type: str = 'hamming',
                                       eps: float = 1e-8, 
                                       **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run frame-wise Linear Predictive Analysis (LPA) on a waveform, decomposing it into
    an LPC filter (spectral envelope) and a residual/source signal (Section IV.A of
    the paper).

    For each ``win_ms``-long frame (default 30 ms, ``hop_ms`` hop, default 5 ms),
    LPC coefficients are estimated via the Levinson-Durbin recursion
    (:func:`levinson_durbin`) on the frame's autocorrelation, and the
    prediction residual (source) is computed as the difference between the
    frame and its linear prediction.

    Args:
        signal: 1D waveform tensor.
        sr: sample rate.
        order: LPC order.
        win_ms: LPC analysis window length, in ms.
        hop_ms: hop between windows, in ms.
        apply_window: whether to apply a window function before analysis.
        window_type: window type to apply if ``apply_window`` is True
            (only ``'hamming'`` is implemented; anything else is a no-op
            rectangular window).
        eps: small value to avoid division by zero in the recursion.

    Returns:
        Tuple of ``(lpc_coeffs, residuals, preds)``: LPC coefficients shaped
        ``(n_frames, order+1)``, and the residual/predicted signal per frame,
        both shaped ``(n_frames, win_len)``.
    """
    device = signal.device
    dtype = signal.dtype

    win_len = int(sr * win_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = 1 + (len(signal) - win_len) // hop_len

    if apply_window:
        if window_type == 'hamming':
            window = torch.from_numpy(np.hamming(win_len).astype(np.float32)).to(device=device, dtype=dtype)
        else:
            window = torch.ones(win_len, device=device, dtype=dtype)
    else:
        window = torch.ones(win_len, device=device, dtype=dtype)

    lpc_coeffs = torch.zeros((n_frames, order+1), dtype=dtype, device=device)
    residuals = torch.zeros((n_frames, win_len), dtype=dtype, device=device)
    preds = torch.zeros((n_frames, win_len), dtype=dtype, device=device)
    for i in range(n_frames):
        start = i * hop_len
        frame = signal[start:start+win_len] * window

        r = torch.zeros(order+1, dtype=dtype, device=device)
        for k in range(order+1):
            r[k] = (frame[:win_len-k] * frame[k:]).sum()

        a, e = levinson_durbin(r, order)
        lpc_coeffs[i] = a

        x_padded = torch.nn.functional.pad(frame, (order,0))
        pred = torch.zeros_like(frame)
        if order > 0:
            a_pos = a[1:]
            for n in range(win_len):
                past = x_padded[n + order - order : n + order]
                pred[n] = - torch.dot(a_pos, past.flip(0))
        residuals[i] = frame - pred
        preds[i] = pred

    return lpc_coeffs, residuals, preds