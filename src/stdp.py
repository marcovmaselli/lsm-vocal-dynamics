"""Standalone STDP implementation shared across spiking modules."""

from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = ["AsymmetricSTDP"]

class AsymmetricSTDP:
    """
    GPU-friendly Asymmetric STDP.
    - mode='trace'  : trace-based (fast, recommended)
    - mode='window' : windowed vectorized (keeps explicit time-window behavior)
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

        # Mask delle connessioni (N,N) float 0/1 — utile per sparse connectivity
        self.mask = None
        if mask is not None:
            self.mask = mask.to(self.device).to(self.dtype)

        # trace mode params
        if self.mode == 'trace':
            self.decay_pre = math.exp(- (self.dt_ms / self.tau_plus))
            self.decay_post = math.exp(- (self.dt_ms / self.tau_minus))
            self.batch_size = None
            self.pre_trace = None
            self.post_trace = None

        # window mode params
        if self.mode == 'window':
            # precompute per-offset f_plus (for pre in past, Δ < 0 -> potentiation) and f_minus (Δ>0 -> depression)
            # We will build arrays for offsets k = 1..window_steps-1 (past steps)
            self.window_steps = max(1, int(math.ceil(self.window_ms / self.dt_ms)))
            # offsets array of length window_steps-1 representing how far in the past (1..T-1)
            if self.window_steps > 1:
                ks = torch.arange(self.window_steps-1, 0, -1, device=self.device, dtype=self.dtype)  # newest past =1
                # delta_ms for pre in the past relative to current step: negative values -k*dt
                delta_pre_ms = - (ks * self.dt_ms)
                # f_plus(delta_ms) = A_plus * exp(delta_ms / tau_plus)
                self.f_plus_offsets = (self.A_plus * torch.exp(delta_pre_ms / self.tau_plus)).to(self.device).to(self.dtype)
                # for depression: when pre AFTER post (pre now, post in past): weights for past post
                delta_post_ms = (ks * self.dt_ms)  # positive
                self.f_minus_offsets = ( self.A_minus * torch.exp(- delta_post_ms / self.tau_minus)).to(self.device).to(self.dtype)
            else:
                # window of 1 -> effectively no history
                self.f_plus_offsets = torch.zeros((0,), device=self.device, dtype=self.dtype)
                self.f_minus_offsets = torch.zeros((0,), device=self.device, dtype=self.dtype)

            # buffer shape (batch, window_steps, N) of recent spikes
            self.batch_size = None
            self.buffer = None  # allocate on first batch

        # diagnostics / causality
        self.C_pot = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)
        self.C_dep = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)

    def reset(self):
        """Reset internal traces/buffers and causality stats."""
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
        if self.batch_size != batch:
            self.batch_size = batch
            self.pre_trace = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
            self.post_trace = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)

    def _ensure_batch_buffer(self, batch: int):
        if self.batch_size != batch:
            self.batch_size = batch
            self.buffer = torch.zeros((batch, self.window_steps, self.N), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def step_update(self, W: torch.Tensor,
                    spk_t: torch.Tensor,
                    t_step: int = 0,
                    clamp_min: Optional[float] = None,
                    clamp_max: Optional[float] = None):
        """
        Aggiorna W in-place.
        - W: (N,N) tensor sul device desiderato
        - spk_t: (batch, N) o (N,) tensor (0/1) degli spike correnti, deve essere sul device preferito o verrà spostato
        - t_step: indice temporale (solo per compatibilità)
        """
        # normalize input
        if spk_t.dim() == 1:
            spk_t = spk_t.unsqueeze(0)
        batch, N = spk_t.shape
        assert N == self.N, f"N mismatch: {N} vs {self.N}"

        spk = spk_t.to(self.device, dtype=self.dtype)
        
        if self.refractory:
            t_now = self.global_step
            mask_ref = (t_now - self.last_spike_step) >= self.refractory_steps
            spk = spk_t.to(self.device, dtype=self.dtype)
            spk = spk * mask_ref.unsqueeze(0).to(self.dtype)
            new_spikes = (spk > 0).any(dim = 0)
            self.last_spike_step[new_spikes] = t_now
            self.global_step += 1

        if self.mode == 'trace':
            # trace update
            self._ensure_batch_trace(batch)
            # decay + add spikes
            # self.pre_trace.mul_(self.decay_pre)
            # self.pre_trace.add_(spk)   # pre_trace now includes current pre spikes
            # self.post_trace.mul_(self.decay_post)
            # self.post_trace.add_(spk)
            self.pre_trace.mul_(self.decay_pre)
            self.post_trace.mul_(self.decay_post)
            # contribuzioni (somma su batch)
            # contrib_plus: pre_trace (bn) outer post_spk (bm) -> nm via einsum
            # contrib_plus = torch.einsum('bn,bm->nm', self.pre_trace, spk)
            # # contrib_minus: pre_spk (bn) outer post_trace (bm)
            # contrib_minus = torch.einsum('bn,bm->nm', spk, self.post_trace)
            contrib_plus = self.pre_trace.transpose(0,1) @ spk        # (N,N)
            contrib_minus = spk.transpose(0,1) @ self.post_trace      # (N,N)


            deltaW = (self.lr * (self.A_plus * contrib_plus - self.A_minus * contrib_minus)).to(W.device)

            # separiamo potenziamento/depressione per statistiche:
            if self.C_pot is not None:
                pot = (self.lr * self.A_plus * contrib_plus).to(self.device)
                self.C_pot += pot
            if self.C_dep is not None:
                dep = (self.lr * self.A_minus * contrib_minus).to(self.device)
                # note: A_minus is positive constant but contrib_minus uses post_trace -> this term is typically positive in magnitude
                self.C_dep += dep.abs()
            self.pre_trace.add_(spk)
            self.post_trace.add_(spk)


        else:
            # window mode vectorized
            self._ensure_batch_buffer(batch)
            # shift buffer left and append current spikes at last index
            if self.window_steps > 1:
                # roll is fine: moves memory but implemented in C
                self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
                self.buffer[:, -1, :] = spk
                # pre-history: all except last index (older steps)
                pre_history = self.buffer[:, :-1, :]  # (batch, T-1, N)
                post_history = self.buffer[:, :-1, :]  # same
                # weigh along time dimension by precomputed offset weights (length T-1)
                if pre_history.shape[1] > 0:
                    # shape (T-1) -> (1, T-1, 1) broadcast
                    w_plus = self.f_plus_offsets.view(1, -1, 1)  # (1, T-1, 1)
                    pre_weighted = (pre_history * w_plus).sum(dim=1)  # (batch, N)
                    w_minus = self.f_minus_offsets.view(1, -1, 1)
                    post_weighted = (post_history * w_minus).sum(dim=1)  # (batch, N)
                else:
                    pre_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
                    post_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
            else:
                # no history
                self.buffer[:, -1, :] = spk
                pre_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)
                post_weighted = torch.zeros((batch, self.N), device=self.device, dtype=self.dtype)

            # contribuzioni (sum over batch)
            contrib_plus = torch.einsum('bn,bm->nm', pre_weighted, spk)    # pre_history weighted outer post_now
            contrib_minus = torch.einsum('bn,bm->nm', spk, post_weighted) # pre_now outer post_history weighted

            deltaW = (self.lr * (contrib_plus - contrib_minus)).to(W.device)

            # update stats: separate pot/dep using signs of offsets
            if self.C_pot is not None:
                # pot stems from contrib_plus (f_plus offsets are positive)
                pot = (self.lr * contrib_plus).to(self.device)
                self.C_pot += pot
            if self.C_dep is not None:
                # contrib_minus may be negative if f_minus_offsets negative; accumulate absolute
                dep = (self.lr * contrib_minus).to(self.device)
                self.C_dep += dep.abs()

        # apply mask if present
        if self.mask is not None:
            deltaW = deltaW * self.mask.to(deltaW.device)

        # apply update in-place
        with torch.no_grad():
            W.add_(deltaW)
            if clamp_min is not None or clamp_max is not None:
                W.clamp_(min=clamp_min, max=clamp_max)

        return W

    def get_causality(self, normalize: bool = True, eps: float = 1e-9):
        if self.C_pot is None or self.C_dep is None:
            return None
        net = (self.C_pot - self.C_dep).to(self.device)
        if not normalize:
            return net
        denom = (self.C_pot + self.C_dep + eps).to(self.device)
        return net / denom