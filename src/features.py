"""Feature extraction utilities for the audio-to-spikes pipeline."""

import torch
from torchaudio.transforms import MelSpectrogram
import numpy as np
from scipy.signal import freqz

__all__ = [
    "MinMaxNormalizer",
    "compute_mel_spectrogram",
    "erb2hz",
    "hz2erb",
    "lpc_filters_to_erb",
    "make_erb_centers",
    "residuals_to_gammatone_energy",
]

def compute_mel_spectrogram(
        waveform: torch.Tensor,
        sr: int = 16000,
        n_mels: int = 128,
        win_length_ms: int = 64,
        hop_length_ms: int = 16,
        f_min: float = 0.0,
        f_max: float = None,
        log_compression: bool = True
) -> torch.Tensor:
    """
    Compute log-Mel spectrogram from a waveform.

    Args:
        waveform: 1D torch.Tensor
        sr: sample rate
        n_mels: number of Mel bands
        win_length_ms: window length in ms
        hop_length_ms: hop length in ms
        f_min: min frequency
        f_max: max frequency (default sr/2)
        log_compression: apply log(1 + x) compression if True

    Returns:
        Tensor of shape [n_mels, num_frames]
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)  # mono

    f_max = f_max or sr / 2
    win_length = int(sr * win_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)

    mel_spec = MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        power=1.0  # amplitude
    )(waveform.unsqueeze(0))  # add channel dim
    mel_spec = mel_spec.squeeze(0)

    if log_compression:
        mel_spec = torch.log1p(mel_spec)

    return mel_spec


class MinMaxNormalizer:
    """Fit a scalar min-max range and reuse it on subsequent tensors."""
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, tensor: torch.Tensor):
        """Record ``tensor``'s min/max as the scaling range for future calls to :meth:`transform`."""
        self.min_val = tensor.min().item()
        self.max_val = tensor.max().item()

    def transform(self, tensor: torch.Tensor):
        """Scale ``tensor`` to ``[0, 1]`` using the range recorded by :meth:`fit`, clamped at the edges."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer not fitted yet")
        norm = (tensor - self.min_val) / (self.max_val - self.min_val)
        return torch.clamp(norm, 0.0, 1.0)

    def fit_transform(self, tensor: torch.Tensor):
        """Equivalent to calling :meth:`fit` then :meth:`transform` on the same tensor."""
        self.fit(tensor)
        return self.transform(tensor)


def hz2erb(f):
    """Convert Hertz to the ERB-rate scale."""
    return 21.4 * np.log10(4.37e-3 * f + 1.0)

def erb2hz(e):
    """Convert ERB-rate values back to Hertz."""
    return (10**(e / 21.4) - 1.0) / 4.37e-3

def make_erb_centers(n_filters, f_min, f_max):
    """Create evenly spaced ERB center frequencies between two limits."""
    erb_min = hz2erb(f_min)
    erb_max = hz2erb(f_max)
    erb_points = np.linspace(erb_min, erb_max, n_filters)
    return erb2hz(erb_points)

def lpc_filters_to_erb(lpc_coeffs: np.ndarray,
                       sr: int = 16000,
                       n_filters: int = 77,
                       f_min: float = 80.0,
                       f_max: float = None,
                       n_freqs: int = 512,
                       log_compression: bool = True,
                       pool: str = 'mean',
                       device: torch.device = None) -> torch.Tensor:
    """
    Convert per-frame LPC coefficients into an ERB-banded spectral-envelope
    representation (the "filter" branch of the paper's source-filter
    decomposition, Section IV.A): for each frame, the LPC magnitude response
    ``H(f)`` is evaluated on a linear frequency grid and pooled within each
    of ``n_filters`` ERB-spaced bands between ``f_min`` and ``f_max``.

    Args:
      lpc_coeffs: np.ndarray shape (n_frames, order+1), as returned by
          :func:`~src.preprocessing.linear_predictive_analysis` (a[0]=1, a[1..order]).
      sr: sample rate.
      n_filters: number of ERB channels (56 in the paper).
      f_min, f_max: band edges in Hz (``f_max`` defaults to ``sr/2``).
      n_freqs: number of frequency bins to evaluate ``H(f)`` on.
      log_compression: apply log compression to the pooled magnitude matrix,
          then shift it to be non-negative (spiking neuron inputs cannot be
          negative; see Section IV.A of the paper).
      pool: 'mean' or 'max' to aggregate magnitude inside each ERB band.
      device: torch device (optional).

    Returns:
      erb_tensor: torch.Tensor shape (n_filters, n_frames)
    """
    if f_max is None:
        f_max = sr / 2.0
    n_frames, p1 = lpc_coeffs.shape

    # Frequency grid (Hz)
    freqs = np.linspace(0.0, f_max, n_freqs)

    # ERB centers & edges
    erb_centers_hz = make_erb_centers(n_filters, f_min, f_max)
    erb_centers_erb = hz2erb(erb_centers_hz)
    edges_erb = np.concatenate((
        [erb_centers_erb[0] - (erb_centers_erb[1]-erb_centers_erb[0])/2],
        (erb_centers_erb[:-1] + erb_centers_erb[1:]) / 2,
        [erb_centers_erb[-1] + (erb_centers_erb[-1]-erb_centers_erb[-2])/2]
    ))
    edges_hz = erb2hz(edges_erb)

    # Precompute masks for ERB bands
    band_masks = []
    for i in range(n_filters):
        lo, hi = edges_hz[i], edges_hz[i+1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            idx = np.argmin(np.abs(freqs - (lo+hi)/2.0))
            tmp = np.zeros_like(freqs, dtype=bool)
            tmp[idx] = True
            mask = tmp
        band_masks.append(mask)

    erb_mat = np.zeros((n_filters, n_frames), dtype=np.float32)

    for t in range(n_frames):
        a = lpc_coeffs[t]  # shape (order+1,)
        _, h = freqz([1.0], a, worN=n_freqs, fs=sr)
        h_mag = np.abs(h)
        for i in range(n_filters):
            vals = h_mag[band_masks[i]]
            erb_mat[i, t] = vals.mean() if pool == 'mean' else vals.max()

    # Log compression over the whole matrix, then shift to non-negative
    # (spiking neuron inputs cannot be negative).
    if log_compression:
        epsilon = 1e-12
        erb_mat = np.log(erb_mat + epsilon)
        erb_mat -= erb_mat.min()

    erb_mat = torch.from_numpy(erb_mat).to(device or 'cpu', dtype=torch.float32)

    return erb_mat

def residuals_to_gammatone_energy(residuals: torch.Tensor,
                                  sr: int = 16000,
                                  n_filters: int = 77,
                                  f_min: float = 80.0,
                                  f_max: float = None,
                                  segment_length : float = 5,
                                  log_compression: bool = True,
                                  device: torch.device = None) -> torch.Tensor:
    """
    Convert a batch of LPC residuals (the "source"/excitation branch of the
    paper's source-filter decomposition, Section IV.A) into per-frame energy
    across ``n_filters`` ERB-spaced gammatone channels: each residual frame
    is convolved with a gammatone impulse response centered on each ERB
    channel, and the mean squared output is taken as that channel's energy.

    Args:
        residuals: LPC residual frames, shape ``(n_frames, frame_len)``.
        sr: sample rate.
        n_filters: number of ERB/gammatone channels (56 in the paper).
        f_min, f_max: band edges in Hz (``f_max`` defaults to ``sr/2``).
        segment_length: gammatone impulse response length, **in milliseconds**
            despite the name (matches how it is used below: ``sr * segment_length / 1000``).
        log_compression: apply log compression then shift to non-negative
            (see :func:`lpc_filters_to_erb`), per channel.
        device: torch device (defaults to ``residuals.device``).

    Returns:
        gt_energy: torch.Tensor shape ``(n_filters, n_frames)``.
    """
    device = device or residuals.device
    residuals = residuals.to(device).float()
    n_frames, frame_len = residuals.shape
    f_max = f_max or sr/2.0

    # ERB-spaced channel center frequencies (local Hz<->ERB helpers, equivalent
    # to hz2erb/erb2hz above but kept self-contained here).
    def hz_to_erb(f): return 21.4 * np.log10(4.37e-3 * f + 1.0)
    def erb_to_hz(e): return (10**(e / 21.4) - 1.0) / 4.37e-3
    erb_centers = np.linspace(hz_to_erb(f_min), hz_to_erb(f_max), n_filters)
    centers = torch.tensor(erb_to_hz(erb_centers), dtype=torch.float32, device=device)

    gt_energy = []
    filt_len = min(int(sr * segment_length / 1000.0), frame_len)

    for fc in centers:
        # Gammatone impulse response
        t = torch.arange(0, filt_len, device=device).float()/sr
        erb = 24.7*(4.37e-3*fc+1.0)
        b = 1.019*erb
        n = 4
        env = (t**(n-1)) * torch.exp(-2*np.pi*b*t)
        carrier = torch.cos(2*np.pi*fc*t)
        h = env*carrier
        h = h / torch.sqrt(torch.clamp((h**2).sum(), min=1e-12))

        # Frame-wise convolution with the gammatone impulse response.
        conv = torch.nn.functional.conv1d(
            residuals.unsqueeze(1),  # shape (n_frames,1,frame_len)
            h.view(1,1,-1),
            padding=0
        ).squeeze(1)  # shape (n_frames, frame_len - filt_len +1)

        # Per-frame energy: mean squared output over the convolution window.
        energy = (conv**2).mean(dim=1)  # shape (n_frames,)

        if log_compression:
            epsilon = 1e-12
            energy = torch.log(energy + epsilon)
            energy -= energy.min()

        gt_energy.append(energy)

    gt_energy = torch.stack(gt_energy, dim=0)  # shape (n_filters, n_frames)

    return gt_energy