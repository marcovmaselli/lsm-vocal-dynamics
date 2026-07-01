"""Liquid State Machine layers and connectivity builders.

The module contains helper functions to create the reservoir topology, build the
input and recurrent connectivity matrices, and run online STDP during forward
passes when requested.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import snntorch as snn

__all__ = [
    "LSM",
    "LSM_GPU",
    "build_3d_positions",
    "build_weight_in_layered",
    "build_weight_in_layered_one_to_one",
    "build_weight_lsm_probabilistic",
    "pairwise_squared_distances",
]

def build_3d_positions(n_layers = 77, layer_h = 3, layer_w = 3):
    """Return 3D coordinates for every neuron in the layered reservoir."""
    coords = []
    for z in range(n_layers):
        for y in range(layer_h):
            for x in range(layer_w):
                coords.append((z,y,x))
    return torch.tensor(coords, dtype=torch.float32)

def pairwise_squared_distances(positions: torch.Tensor):
    """Compute all pairwise squared Euclidean distances between positions."""
    p = positions
    norms = (p * p).sum(dim = 1, keepdim = True)
    d2 = norms + norms.t() - 2.0 * (p @ p.t())
    d2 = torch.clamp(d2, min = 0.0)
    return d2

def build_weight_lsm_probabilistic(positions : torch.Tensor, C = 1.0, l = 3.4,
                                   w_mean = 0.08, w_std = 0.03,
                                   self_connection = False,
                                   inhibit = True,
                                   inh_ratio: float = 0.2,  # 20% neuroni inibitori
                                   seed: Optional[int] = 48,
                                   device: Optional[torch.device] = None):
    """Sample recurrent reservoir weights from a distance-based probability map."""
    if seed is not None:
        torch.manual_seed(seed)
    pos = positions.to(device) if device is not None else positions
    d2 = pairwise_squared_distances(pos)
    prob = C * torch.exp(- d2 / (l * l))
    prob = torch.clamp(prob, max = 1.0)
    bern = torch.distributions.Bernoulli(probs = prob)
    mask = bern.sample()
    N = prob.shape[0]
    weights = torch.normal(mean = w_mean, std = w_std, size = (N, N), device = pos.device)

    # selezione neuroni inibitori
    if inhibit:
        n_inh = int(N * inh_ratio)
        inh_idx = torch.randperm(N)[:n_inh]
        weights[inh_idx, :] *= -1.0

    W = mask * weights
    if not self_connection:
        W.fill_diagonal_(0.0)
    return W, prob

def build_weight_in_layered(n_layers = 77, layer_h = 3, layer_w = 3, n_inputs = 77,
                            win_strength = 0.5, inhibit = True, inh_ratio: float = 0.2,
                            device: Optional[torch.device]=None):
    """Create a layered input matrix where each input channel drives one layer."""
    N = n_layers * layer_h * layer_w
    assert n_inputs == n_layers, "LSM ASSERT: n_inputs must equal n_layers"
    Win = torch.zeros((N, n_inputs), dtype=torch.float32, device=device)
    for layer in range(n_layers):
        start = layer * (layer_h * layer_w)
        end = start + (layer_h * layer_w)
        Win[start:end, layer] = win_strength

    # neuroni inibitori
    if inhibit:
        n_inh = int(N * inh_ratio)
        inh_idx = torch.randperm(N)[:n_inh]
        Win[inh_idx, :] *= -1.0
    return Win

def build_weight_in_layered_one_to_one(n_layers, layer_h, layer_w, n_inputs,
                                       win_strength=0.5,
                                       inhibit = True,
                                       inh_ratio: float = 0.2,
                                       device: Optional[torch.device] = None):
    """Create a one-to-one layered input matrix with randomized per-neuron gains."""
    assert n_inputs == n_layers
    N = n_layers * layer_h * layer_w
    Win = torch.zeros((N, n_inputs), device=device)
    for i in range(n_layers):
        start = i * (layer_h * layer_w)
        end = start + (layer_h * layer_w)
        Win[start:end, i] = win_strength * torch.rand(layer_h * layer_w, device=device)
    
    # neuroni inibitori
    if inhibit:
        n_inh = int(N * inh_ratio)
        inh_idx = torch.randperm(N)[:n_inh]
        Win[inh_idx, :] *= -1.0
    return Win

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

class LSM(nn.Module):
    """
    LSM che supporta costruzione 3D del reservoir e STDP pairwise-exact online.

    Args:
        n_layers, layer_h, layer_w: definiscono N = n_layers*layer_h*layer_w
        in_ch: numero di input channels 
        Win, Wlsm: opzionali; se None vengono costruite automaticamente secondo i builder
        use_stdp: se True, applica aggiornamenti STDP su Wlsm durante forward
        stdp_params: dizionario con parametri per AsymmetricSTDP (es. dt_ms, window_ms, A_plus, ...)
        alpha, beta, th: parametri del RSynaptic
    """
    def __init__(self,
                 n_layers: int = 77,
                 layer_h: int = 3,
                 layer_w: int = 3,
                #  in_ch: int = 77,
                 Win: Optional[torch.Tensor] = None,
                 Win_strength: float = 1.0,
                 Wlsm: Optional[torch.Tensor] = None,
                 C: float = 1.0,
                 sigma: float = 3.4,
                 use_stdp: bool = False,
                 stdp_params: Optional[dict] = None,
                 alpha: float = 0.9,
                 beta: float = 0.9,
                 th: float = 20.0,
                 neuron_type : str = 'RSynaptic',
                 one_to_one : bool = True,
                 inhibit : bool = True):
        super().__init__()
        self.n_layers = n_layers
        self.layer_h = layer_h
        self.layer_w = layer_w
        self.N = n_layers * layer_h * layer_w
        in_ch = n_layers
        # self.in_ch = in_ch
        # assert in_ch == n_layers, "Per il design richiesto, in_ch deve essere uguale a n_layers"

        # coordinate 3D
        self.positions = build_3d_positions(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w)

        # Win (input->reservoir)
        if Win is None:
            if one_to_one:
                Win = build_weight_in_layered_one_to_one(
                    n_layers=n_layers,
                    layer_h=layer_h,
                    layer_w=layer_w,
                    n_inputs=in_ch,
                    win_strength=0.5,
                    inhibit = inhibit
                )
            else:
                Win = build_weight_in_layered(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w,
                                            n_inputs=in_ch, win_strength=0.5, inhibit=inhibit)
        Win = Win*Win_strength
        assert Win.shape == (self.N, in_ch)
        self.register_buffer("Win", Win)

        # Wlsm (ricorrente)
        if Wlsm is None:
            Wlsm_init, prob = build_weight_lsm_probabilistic(self.positions, C=C, l=sigma,
                                                            w_mean=0.08, w_std=0.03, self_connection=False, inhibit = inhibit)
            Wlsm = Wlsm_init
            self.register_buffer("W_prob", prob)
        assert Wlsm.shape == (self.N, self.N)
        # registriamo Wlsm come buffer che verrà aggiornato in-place dallo STDP
        self.register_buffer("Wlsm", Wlsm)

        # mapping input->reservoir (copia Win)
        self.fc1 = nn.Linear(in_ch, self.N, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(self.Win)

        # RSynaptic (snntorch)
        # nota: RSynaptic con R maiuscola
        self.neuron_type = neuron_type
        if self.neuron_type == 'RSynaptic':
            self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True,
                                    linear_features=self.N, threshold=th)
        elif self.neuron_type == 'Leaky':
            self.lsm = snn.RLeaky(beta=beta, all_to_all=True,
                                    linear_features=self.N, threshold=th)
        with torch.no_grad():
            self.lsm.recurrent.weight.copy_(self.Wlsm)

        # STDP
        self.use_stdp = bool(use_stdp)
        if self.use_stdp:
            stdp_args = stdp_params.copy() if stdp_params is not None else {}
            # assicurati che stdp riceva device: preferibilmente il device di Wlsm (buffer)
            device_for_stdp = self.Wlsm.device if hasattr(self, "Wlsm") else None
            stdp_args["device"] = device_for_stdp
            # Inizializza la tua classe AsymmetricSTDP (che prende N come primo argomento)
            self.stdp = AsymmetricSTDP(self.N, **stdp_args)

    def forward(self, x: torch.Tensor, apply_stdp: bool = False,
                clamp_min: Optional[float] = -1.0, clamp_max: Optional[float] = 1.0):
        """
        x: (time_steps, batch, in_ch)
        if apply_stdp True e use_stdp True -> verrà applicata la step_update pairwise-exact su Wlsm
        Nota: dopo ogni aggiornamento STDP copiamos i pesi aggiornati dentro self.lsm.recurrent.weight
        in modo che il modulo snntorch li usi nel passo successivo.
        """
        device = x.device
        num_steps, batch_size, channels = x.size()

        # inizializza stato del RSynaptic sul device corretto
        if self.neuron_type == 'RSynaptic':
            spk, syn, mem = self.lsm.init_rsynaptic()
        elif self.neuron_type == 'Leaky':
            spk,  mem = self.lsm.init_rleaky()
        spk_rec = []

        # reset STDP internamente prima di una nuova sequenza
        if self.use_stdp and apply_stdp:
            self.stdp.reset()

        for step in range(num_steps):
            curr = self.fc1(x[step])           # (batch, N) presupponendo x[step] shape (batch, in_ch)
            if self.neuron_type == 'RSynaptic':
                spk, syn, mem = self.lsm(curr, spk, syn, mem)
            elif self.neuron_type == 'Leaky':
                spk,  mem = self.lsm(curr, spk, mem)
            spk_rec.append(spk)

            if self.use_stdp and apply_stdp:
                # spk può essere (batch, N) o (N,) a seconda del batch size; passalo direttamente
                spk_now = spk.detach()
                # La tua AsymmetricSTDP.step_update si occupa di batch internamente (somma per elemento)
                # e mantiene deques separati per ciascun elemento del batch.
                # Passa self.Wlsm (buffer), spk_now e l'indice temporale corrente
                self.stdp.step_update(self.Wlsm, spk_now, t_step=step, clamp_min=clamp_min, clamp_max=clamp_max)

                # Dopo l'aggiornamento, copia i pesi aggiornati dentro snntorch
                with torch.no_grad():
                    self.lsm.recurrent.weight.copy_(self.Wlsm)

        spk_rec_out = torch.stack(spk_rec)  # (time, batch, N)
        # batch_size = x.size(1)
        # outputs = []
        # for b in range(batch_size):
        #     spk_b, syn_b, mem_b = self.lsm.init_rsynaptic()
        #     spk_rec_b = []
        #     for step in range(num_steps):
        #         curr = self.fc1(x[step, b, :].unsqueeze(0))  # (1, in_ch)
        #         spk_b, syn_b, mem_b = self.lsm(curr, spk_b, syn_b, mem_b)
        #         spk_rec_b.append(spk_b)
        #     outputs.append(torch.stack(spk_rec_b, dim=0))  # (T, 1, N)
        # spk_rec_out = torch.cat(outputs, dim=1)  # (T, B, N)
        return spk_rec_out
    
class LSM_GPU(nn.Module):
    """
    LSM ottimizzata per GPU con STDP pairwise-exact online e supporto batch.
    Replica le funzionalità di LSM classico ma sfrutta parallelizzazione GPU.
    """
    def __init__(self,
                 n_layers: int = 77,
                 layer_h: int = 3,
                 layer_w: int = 3,
                 in_ch: int = 77,
                 Win: Optional[torch.Tensor] = None,
                 Wlsm: Optional[torch.Tensor] = None,
                 C: float = 1.0,
                 sigma: float = 3.4,
                 use_stdp: bool = False,
                 stdp_params: Optional[dict] = None,
                 alpha: float = 0.9,
                 beta: float = 0.9,
                 th: float = 20.0,
                 one_to_one: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_layers = n_layers
        self.layer_h = layer_h
        self.layer_w = layer_w
        self.N = n_layers * layer_h * layer_w
        self.in_ch = in_ch
        self.use_stdp = use_stdp
        assert in_ch == n_layers, "Per il design richiesto, in_ch deve essere uguale a n_layers"

        # coordinate 3D
        self.positions = build_3d_positions(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w)

        # Win (input -> reservoir)
        if Win is None:
            if one_to_one:
                Win = build_weight_in_layered_one_to_one(
                    n_layers=n_layers, layer_h=layer_h, layer_w=layer_w,
                    n_inputs=in_ch, win_strength=0.5
                )
            else:
                Win = build_weight_in_layered(
                    n_layers=n_layers, layer_h=layer_h, layer_w=layer_w,
                    n_inputs=in_ch, win_strength=0.5
                )
        self.register_buffer("Win", Win.to(self.device))

        # Wlsm (ricorrente)
        if Wlsm is None:
            Wlsm_init, prob = build_weight_lsm_probabilistic(
                self.positions, C=C, l=sigma,
                w_mean=0.08, w_std=0.03, self_connection=False
            )
            Wlsm = Wlsm_init
            self.register_buffer("W_prob", prob.to(self.device))
        self.register_buffer("Wlsm", Wlsm.to(self.device))

        # mapping input -> reservoir
        self.fc1 = nn.Linear(in_ch, self.N, bias=False).to(self.device)
        with torch.no_grad():
            self.fc1.weight.copy_(self.Win)

        # RSynaptic snntorch
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True,
                                 linear_features=self.N, threshold=th).to(self.device)
        with torch.no_grad():
            self.lsm.recurrent.weight.copy_(self.Wlsm)

        # STDP
        self.stdp = None
        if self.use_stdp:
            stdp_args = stdp_params.copy() if stdp_params is not None else {}
            stdp_args["device"] = self.device
            self.stdp = AsymmetricSTDP(self.N, **stdp_args)

    def forward(self, x: torch.Tensor, apply_stdp: bool = False,
                clamp_min: Optional[float] = -1.0, clamp_max: Optional[float] = 1.0):
        """
        x: (T, batch, in_ch)
        """
        T, batch_size, _ = x.shape
        spk, syn, mem = self.lsm.init_rsynaptic()
        spk = spk.expand(batch_size, -1)
        syn = syn.expand(batch_size, -1)
        mem = mem.expand(batch_size, -1)
        spk_rec = []

        if self.use_stdp and apply_stdp:
            self.stdp.reset()

        for t in range(T):
            curr = self.fc1(x[t])  # (batch, N)
            spk, syn, mem = self.lsm(curr, spk, syn, mem)
            spk_rec.append(spk)

            if self.use_stdp and apply_stdp:
                # STDP batch-parallel
                self.stdp.step_update(self.Wlsm, spk.detach(), t_step=t,
                                      clamp_min=clamp_min, clamp_max=clamp_max)
                # copia pesi aggiornati in RSynaptic
                with torch.no_grad():
                    self.lsm.recurrent.weight.copy_(self.Wlsm)

        return torch.stack(spk_rec)  # (T, batch, N)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    n_layers = 77
    layer_h = 3
    layer_w = 3
    in_ch = 77
    N = n_layers * layer_h * layer_w

    # Costruzione input weights e recurrent weights
    positions = build_3d_positions(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w)
    Wlsm_init, prob = build_weight_lsm_probabilistic(positions, C=1.0, l=3.4, w_mean=0.08, w_std=0.03, seed=42)
    Win = build_weight_in_layered(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w, n_inputs=in_ch, win_strength=0.5)

    # Creazione modello LSM con STDP
    lsm = LSM(Win=Win, Wlsm=Wlsm_init.clone(), th=0.5, alpha=0.8, beta=0.8, use_stdp=True)
    stdp = lsm.stdp  # usa lo stesso oggetto STDP interno

    # Il tuo input: 77 canali x 395 time step -> batch=1
    # Supponiamo che tu abbia un tensor torch chiamato `input_data` shape (77,395)
    # Lo trasformiamo in (time, batch, in_ch) = (395, 1, 77)
    input_data = torch.rand((77,395))  # esempio casuale; sostituire con la tua matrice reale
    inputs = input_data.t().unsqueeze(1)  # shape (395, 1, 77)

    # Resetta STDP prima della simulazione
    stdp.reset()

    # Forward passo-passo con STDP
    start = time.time()
    spikes_rec = lsm.forward(inputs, apply_stdp=True, clamp_min=-1.0, clamp_max=1.0)
    elapsed = time.time() - start
    print('Simulazione completata in {:.2f} s'.format(elapsed))

    W_final = lsm.Wlsm  # i pesi aggiornati in-place

    print("Spikes tensor shape:", spikes_rec.shape)  # (395, 1, 231)
    
    # Plot raster batch 0, primi 22 neuroni
    neurons_to_plot = min(22, N)
    fig, axes = plt.subplots(2,1, figsize=(9,6))
    axes[0].set_title(f"Raster spikes (batch 0) primi {neurons_to_plot} neuroni")
    axes[0].imshow(spikes_rec[:,0,:neurons_to_plot].t().detach().numpy() , aspect='auto')
    axes[0].set_ylabel("Neuron index")
    axes[0].set_xlabel("Time step")

    # Causality heatmap normalizzata
    causality = stdp.get_causality(normalize=True)
    axes[1].set_title(f"Causalità normalizzata ({N}x{N})")
    axes[1].imshow(causality.detach().numpy() , aspect='auto')
    axes[1].set_xlabel("post neuron index")
    axes[1].set_ylabel("pre neuron index")

    plt.tight_layout()
    plt.show()

    # Top 10 connessioni più causali (assoluto)
    net = (stdp.C_pot - stdp.C_dep).abs().numpy()
    pairs = [(net[i,j], i, j) for i in range(N) for j in range(N)]
    top10 = sorted(pairs, key=lambda x: -x[0])[:10]
    print("Top 10 (abs net causal influence):")
    for val,i,j in top10:
        print(f"pre {i} -> post {j}: score {val:.6f}")

    # Statistiche variazione pesi
    with torch.no_grad():
        deltaW = W_final - Wlsm_init
        print("W change stats: mean {:.6f}, std {:.6f}, min {:.6f}, max {:.6f}".format(
            deltaW.mean().item(),
            deltaW.std().item(),
            deltaW.min().item(),
            deltaW.max().item()
        ))