"""Asymmetric Spike-Timing-Dependent Plasticity (STDP), shared across LSM reservoirs.

Implements Eq. (4) of the paper:

    dw = A+ * exp(-dt/tau+)   if dt > 0  (pre before post -> potentiation)
    dw = -A- * exp(dt/tau-)   if dt < 0  (pre after post -> depression)

where ``dt = t_post - t_pre``. Both an exponential-trace formulation (fast,
used by the paper's pipeline) and an explicit finite-time-window formulation
are provided; see ``mode`` below.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = ["AsymmetricSTDP"]


class AsymmetricSTDP:
    """Batched, GPU-friendly asymmetric STDP over an ``(N, N)`` recurrent weight matrix.

    Two update modes are supported:

    - ``mode='trace'`` (default, recommended): maintains exponentially-decaying
      pre-/post-synaptic traces and updates weights in O(N^2) per step. This
      is what the paper's reservoirs use.
    - ``mode='window'``: an explicit finite time-window formulation that
      looks back over the last ``window_ms`` milliseconds of spike history
      instead of a continuous trace. More memory/compute per step, but keeps
      strictly bounded temporal support.

    A per-neuron refractory period (default 5 ms) can additionally suppress a
    neuron's contribution to plasticity right after it has spiked.

    ``C_pot``/``C_dep`` accumulate, respectively, the total potentiation and
    (absolute) depression applied to every synapse since the last
    :meth:`reset`, purely as a diagnostic (see :meth:`get_causality`); they do
    not feed back into the update rule.
    """

    def __init__(self, N: int,
                 dt_ms: float = 1.0,
                 window_ms: float = 100.0,
                 A_plus: float = 0.01,
                 A_minus: float = 0.0033,
                 tau_plus: float = 20.0,
                 tau_minus: float = 60.0,
                 lr: float = 1.0,
                 refractory: bool = True,
                 mask: Optional[torch.Tensor] = None,
                 mode: str = 'trace',
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            N: number of neurons (weight matrix is ``(N, N)``).
            dt_ms: simulation timestep in milliseconds.
            window_ms: time window considered in ``mode='window'`` (ignored
                in ``'trace'`` mode, where the exponential decay has
                unbounded but vanishing support instead).
            A_plus, A_minus: potentiation/depression amplitudes (Eq. 4).
            tau_plus, tau_minus: potentiation/depression time constants in ms (Eq. 4).
            lr: global learning-rate multiplier applied to every update.
            refractory: if True, a neuron that spiked within the last 5 ms
                does not contribute to that step's trace/window update.
            mask: optional ``(N, N)`` 0/1 tensor restricting which synapses
                are plastic (e.g. for sparse connectivity); ``None`` updates
                all synapses.
            mode: ``'trace'`` or ``'window'``, see class docstring.
            device, dtype: torch device/dtype for internal buffers.
        """
        self.N = N
        self.dt_ms = float(dt_ms)
        self.window_ms = float(window_ms)
        self.window_steps = max(1, int(math.ceil(self.window_ms / self.dt_ms)))
        self.A_plus = float(A_plus)
        self.A_minus = float(A_minus)
        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.lr = float(lr)
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.mode = mode.lower()
        self.refractory = refractory
        if self.refractory:
            self.refractory_ms = 5.0
            self.refractory_steps = max(1, int(self.refractory_ms / self.dt_ms))
            self.last_spike_step = torch.full((N,), -1e9, device=self.device, dtype=torch.float32)
            self.global_step = 0
        assert self.mode in ('trace', 'window'), "mode must be 'trace' or 'window'"

        # Optional (N, N) 0/1 mask restricting which synapses are plastic.
        self.mask = None
        if mask is not None:
            self.mask = mask.to(self.device).to(self.dtype)

        if self.mode == 'trace':
            # Per-step multiplicative decay of the pre-/post-synaptic traces,
            # derived from tau_plus/tau_minus (Eq. 4).
            self.decay_pre = math.exp(- (self.dt_ms / self.tau_plus))
            self.decay_post = math.exp(- (self.dt_ms / self.tau_minus))
            self.batch_size = None
            self.pre_trace = None
            self.post_trace = None

        if self.mode == 'window':
            # Precompute the per-offset potentiation/depression kernel for
            # offsets k = 1..window_steps-1 (steps in the past), so step_update
            # can apply them as a single weighted sum over the spike buffer.
            self.window_steps = max(1, int(math.ceil(self.window_ms / self.dt_ms)))
            if self.window_steps > 1:
                ks = torch.arange(self.window_steps - 1, 0, -1, device=self.device, dtype=self.dtype)
                # Pre-synaptic spike k steps in the past relative to "now": dt = -k*dt_ms.
                delta_pre_ms = - (ks * self.dt_ms)
                self.f_plus_offsets = (self.A_plus * torch.exp(delta_pre_ms / self.tau_plus)).to(self.device).to(self.dtype)
                # Post-synaptic spike k steps in the past relative to "now": dt = +k*dt_ms.
                delta_post_ms = (ks * self.dt_ms)
                self.f_minus_offsets = (self.A_minus * torch.exp(- delta_post_ms / self.tau_minus)).to(self.device).to(self.dtype)
            else:
                # A one-step window has no history to look back on.
                self.f_plus_offsets = torch.zeros((0,), device=self.device, dtype=self.dtype)
                self.f_minus_offsets = torch.zeros((0,), device=self.device, dtype=self.dtype)

            # (batch, window_steps, N) ring buffer of recent spikes, allocated
            # lazily once the batch size of the first step_update call is known.
            self.batch_size = None
            self.buffer = None

        # Diagnostics only (see get_causality); do not affect the update rule.
        self.C_pot = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)
        self.C_dep = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)

    def reset(self):
        """Reset traces/buffers and causality stats (call once per new input sequence)."""
        if self.mode == 'trace':
            self.pre_trace = None
            self.post_trace = None
            self.batch_size = None
        else:
            self.buffer = None
            self.batch_size = None
        self.C_pot.zero_()
        self.C_dep.zero_()

    def _ensure_batch_trace(self, batch: int):
        """(Re)allocate the pre-/post-synaptic trace buffers if the batch size changed."""
        if self.batch_size != batch:
            self.batch_size = batch
            self.pre_trace = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
            self.post_trace = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)

    def _ensure_batch_buffer(self, batch: int):
        """(Re)allocate the spike-history ring buffer if the batch size changed."""
        if self.batch_size != batch:
            self.batch_size = batch
            self.buffer = torch.zeros((batch, self.window_steps, self.N), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def step_update(self, W: torch.Tensor,
                    spk_t: torch.Tensor,
                    t_step: int = 0,
                    clamp_min: Optional[float] = None,
                    clamp_max: Optional[float] = None):
        """Apply one STDP update to ``W`` in place, given the current step's spikes.

        Args:
            W: ``(N, N)`` recurrent weight tensor, updated in place.
            spk_t: current spikes, shaped ``(batch, N)`` or ``(N,)`` (0/1).
            t_step: current timestep index (only used for logging/compatibility;
                the refractory counter tracks its own internal step count).
            clamp_min, clamp_max: if given, ``W`` is clamped to this range
                after the update.

        Returns:
            The same tensor ``W``, updated in place.
        """
        if spk_t.dim() == 1:
            spk_t = spk_t.unsqueeze(0)
        batch, N = spk_t.shape
        assert N == self.N, f"N mismatch: {N} vs {self.N}"

        spk = spk_t.to(self.device, dtype=self.dtype)

        if self.refractory:
            t_now = self.global_step
            mask_ref = (t_now - self.last_spike_step) >= self.refractory_steps
            spk = spk * mask_ref.unsqueeze(0).to(self.dtype)
            new_spikes = (spk > 0).any(dim=0)
            self.last_spike_step[new_spikes] = t_now
            self.global_step += 1

        if self.mode == 'trace':
            self._ensure_batch_trace(batch)
            # Decay existing traces before adding the current step's spikes,
            # so contrib_plus/contrib_minus only reflect *past* activity.
            self.pre_trace.mul_(self.decay_pre)
            self.post_trace.mul_(self.decay_post)

            # Potentiation: past pre-synaptic trace paired with current post spikes.
            contrib_plus = self.pre_trace.transpose(0, 1) @ spk        # (N, N)
            # Depression: current pre spikes paired with past post-synaptic trace.
            contrib_minus = spk.transpose(0, 1) @ self.post_trace      # (N, N)

            deltaW = (self.lr * (self.A_plus * contrib_plus - self.A_minus * contrib_minus)).to(W.device)

            if self.C_pot is not None:
                pot = (self.lr * self.A_plus * contrib_plus).to(self.device)
                self.C_pot += pot
            if self.C_dep is not None:
                # A_minus is a positive constant; accumulate magnitude for diagnostics.
                dep = (self.lr * self.A_minus * contrib_minus).to(self.device)
                self.C_dep += dep.abs()

            self.pre_trace.add_(spk)
            self.post_trace.add_(spk)

        else:
            self._ensure_batch_buffer(batch)
            if self.window_steps > 1:
                # Shift the ring buffer left and append the current spikes.
                self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
                self.buffer[:, -1, :] = spk
                pre_history = self.buffer[:, :-1, :]   # (batch, T-1, N), older steps
                post_history = self.buffer[:, :-1, :]  # same buffer, used symmetrically below
                if pre_history.shape[1] > 0:
                    w_plus = self.f_plus_offsets.view(1, -1, 1)
                    pre_weighted = (pre_history * w_plus).sum(dim=1)  # (batch, N)
                    w_minus = self.f_minus_offsets.view(1, -1, 1)
                    post_weighted = (post_history * w_minus).sum(dim=1)  # (batch, N)
                else:
                    pre_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
                    post_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
            else:
                self.buffer[:, -1, :] = spk
                pre_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
                post_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)

            contrib_plus = torch.einsum('bn,bm->nm', pre_weighted, spk)     # weighted pre-history outer current post
            contrib_minus = torch.einsum('bn,bm->nm', spk, post_weighted)  # current pre outer weighted post-history

            deltaW = (self.lr * (contrib_plus - contrib_minus)).to(W.device)

            if self.C_pot is not None:
                pot = (self.lr * contrib_plus).to(self.device)
                self.C_pot += pot
            if self.C_dep is not None:
                dep = (self.lr * contrib_minus).to(self.device)
                self.C_dep += dep.abs()

        if self.mask is not None:
            deltaW = deltaW * self.mask.to(deltaW.device)

        with torch.no_grad():
            W.add_(deltaW)
            if clamp_min is not None or clamp_max is not None:
                W.clamp_(min=clamp_min, max=clamp_max)

        return W

    def get_causality(self, normalize: bool = True, eps: float = 1e-9):
        """Return the accumulated net (potentiation - depression) per synapse since the last reset.

        Args:
            normalize: if True, divide by ``C_pot + C_dep`` so the result is
                in ``[-1, 1]`` (a diagnostic "causality score" per synapse)
                instead of raw accumulated weight change.
            eps: numerical stability term for the normalized denominator.
        """
        if self.C_pot is None or self.C_dep is None:
            return None
        net = (self.C_pot - self.C_dep).to(self.device)
        if not normalize:
            return net
        denom = (self.C_pot + self.C_dep + eps).to(self.device)
        return net / denom
