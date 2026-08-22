"""Liquid State Machine (LSM) reservoir and its connectivity builders.

This module implements the dual-reservoir LSM architecture described in
Section IV.B of the paper: a 3D grid of spiking neurons organized into
``n_layers`` tonotopic layers (one per gammatone/ERB channel), each a
``layer_h`` x ``layer_w`` block of neurons. Two independent instances of
:class:`LSM` (source and filter) are built by ``build_lsm_pair`` in
``src/lsm_pipeline.py``.

Connectivity:
    - Input weights (``Win``): one-to-one, layer-wise mapping from each input
      channel to its own layer (see ``build_weight_in_layered_one_to_one``),
      preserving tonotopic organization.
    - Recurrent weights (``Wlsm``): sampled probabilistically, with
      connection probability decaying with squared Euclidean distance
      between neurons (Eq. 1 of the paper; see
      ``build_weight_lsm_probabilistic``). A fraction of neurons are
      randomly flagged inhibitory.

Index convention:
    ``Wlsm`` and ``Win`` are copied straight into ``nn.Linear`` weights
    (``self.lsm.recurrent`` and ``self.fc1``), which compute
    ``y[i] = sum_j W[i, j] * x[j]``. Rows are therefore **post-synaptic**
    targets and columns **pre-synaptic** sources: ``Wlsm[i, j]`` is the
    synapse from neuron ``j`` onto neuron ``i``, and ``Win[i, c]`` the one
    from input channel ``c`` onto neuron ``i``. Making a neuron inhibitory
    means negating its *column* (all of its outgoing synapses); negating a
    row would instead make that neuron receive only inhibition, silencing it.

Neuron dynamics follow the second-order RSynaptic leaky integrate-and-fire
model from snnTorch (Eq. 2-3 of the paper), and recurrent weights are
updated online via asymmetric STDP (:class:`~src.stdp.AsymmetricSTDP`) when
``use_stdp=True`` and ``forward(..., apply_stdp=True)``.
"""

from typing import Optional

import torch
import torch.nn as nn
import snntorch as snn

from .stdp import AsymmetricSTDP

__all__ = [
    "LSM",
    "build_3d_positions",
    "build_weight_in_layered",
    "build_weight_in_layered_one_to_one",
    "build_weight_lsm_probabilistic",
    "pairwise_squared_distances",
]


def build_3d_positions(n_layers=77, layer_h=3, layer_w=3):
    """Return integer 3D coordinates ``(layer, row, col)`` for every neuron.

    Neurons are ordered layer-major, so index ``i`` maps to
    ``i // (layer_h*layer_w)`` for the layer and the remainder for the
    within-layer row/col. This ordering is assumed throughout the module
    (e.g. by the layer-wise input builders below).
    """
    coords = []
    for z in range(n_layers):
        for y in range(layer_h):
            for x in range(layer_w):
                coords.append((z, y, x))
    return torch.tensor(coords, dtype=torch.float32)


def pairwise_squared_distances(positions: torch.Tensor):
    """Compute the full ``(N, N)`` matrix of pairwise squared Euclidean distances."""
    p = positions
    norms = (p * p).sum(dim=1, keepdim=True)
    d2 = norms + norms.t() - 2.0 * (p @ p.t())
    d2 = torch.clamp(d2, min=0.0)
    return d2


def build_weight_lsm_probabilistic(positions: torch.Tensor, C=1.0, l=3.4,
                                   w_mean=0.08, w_std=0.03,
                                   self_connection=False,
                                   inhibit=True,
                                   inh_ratio: float = 0.2,
                                   seed: Optional[int] = 48,
                                   device: Optional[torch.device] = None):
    """Sample recurrent reservoir weights from a distance-decaying connection probability.

    Implements Eq. (1) of the paper: ``P(n1, n2) = C * exp(-D^2(n1, n2) / l^2)``,
    where ``D`` is the Euclidean distance between neuron positions and ``l``
    is the spatial decay parameter (``sigma`` elsewhere in this codebase;
    ``l=3.4`` in the paper). A Bernoulli draw with that probability decides
    which pairs are connected; connected weights are drawn from
    ``Normal(w_mean, w_std)``, and a random ``inh_ratio`` fraction of
    neurons act as inhibitory units: their **column** of ``W`` is negated,
    i.e. all of their outgoing synapses become negative (Dale's principle).
    Their incoming synapses are left untouched, so they are driven like any
    other neuron and can fire.

    ``seed`` seeds a local :class:`torch.Generator` rather than the global
    RNG, so building one reservoir does not perturb (or replicate itself
    into) any other; pass distinct seeds to obtain independent reservoirs.

    Returns:
        Tuple of ``(W, prob)``: the ``(N, N)`` sampled weight matrix and the
        underlying connection-probability matrix (useful for diagnostics).
    """
    pos = positions.to(device) if device is not None else positions
    gen = torch.Generator(device=pos.device)
    if seed is not None:
        gen.manual_seed(int(seed))
    d2 = pairwise_squared_distances(pos)
    prob = C * torch.exp(- d2 / (l * l))
    prob = torch.clamp(prob, max=1.0)
    mask = torch.bernoulli(prob, generator=gen)
    N = prob.shape[0]
    weights = torch.normal(mean=w_mean, std=w_std, size=(N, N),
                           generator=gen, device=pos.device)

    if inhibit:
        # Column-wise: neuron `j` is inhibitory, so every synapse *leaving* it
        # (column j) is negated. Negating rows instead would make the neuron
        # receive only inhibition and never fire.
        n_inh = int(N * inh_ratio)
        inh_idx = torch.randperm(N, generator=gen, device=pos.device)[:n_inh]
        weights[:, inh_idx] *= -1.0

    W = mask * weights
    if not self_connection:
        W.fill_diagonal_(0.0)
    return W, prob


def build_weight_in_layered(n_layers=77, layer_h=3, layer_w=3, n_inputs=77,
                            win_strength=0.5,
                            device: Optional[torch.device] = None):
    """Build layered input weights: each input channel drives every neuron in one layer.

    All neurons within a layer receive the same fixed ``win_strength`` from
    their corresponding input channel (``n_inputs`` must equal ``n_layers``).
    Input drive is purely excitatory: inhibition lives in the recurrent
    matrix, where it is expressed per pre-synaptic neuron (see
    :func:`build_weight_lsm_probabilistic`).
    """
    N = n_layers * layer_h * layer_w
    assert n_inputs == n_layers, "LSM ASSERT: n_inputs must equal n_layers"
    Win = torch.zeros((N, n_inputs), dtype=torch.float32, device=device)
    for layer in range(n_layers):
        start = layer * (layer_h * layer_w)
        end = start + (layer_h * layer_w)
        Win[start:end, layer] = win_strength
    return Win


def build_weight_in_layered_one_to_one(n_layers, layer_h, layer_w, n_inputs,
                                       win_strength=0.5,
                                       device: Optional[torch.device] = None):
    """Build layered input weights with per-neuron randomized gain (default LSM input mapping).

    Like :func:`build_weight_in_layered`, but instead of a fixed
    ``win_strength`` for every neuron in a layer, each neuron gets an
    independent gain drawn uniformly from ``[0, win_strength)``. This is the
    input connectivity used by :class:`LSM` by default (``one_to_one=True``).
    Input drive is purely excitatory, as above.
    """
    assert n_inputs == n_layers
    N = n_layers * layer_h * layer_w
    Win = torch.zeros((N, n_inputs), device=device)
    for i in range(n_layers):
        start = i * (layer_h * layer_w)
        end = start + (layer_h * layer_w)
        Win[start:end, i] = win_strength * torch.rand(layer_h * layer_w, device=device)
    return Win


class LSM(nn.Module):
    """Single reservoir: a 3D grid of recurrently-connected spiking neurons.

    Wraps a snnTorch ``RSynaptic`` (or ``RLeaky``) recurrent layer with the
    tonotopic input/recurrent connectivity built above, and optionally drives
    online STDP updates of the recurrent weights during ``forward``.

    Args:
        n_layers, layer_h, layer_w: reservoir shape; ``N = n_layers * layer_h * layer_w``
            neurons total. The number of input channels always equals
            ``n_layers`` (one channel per layer).
        Win, Wlsm: optional pre-built input/recurrent weight matrices; if
            ``None`` they are constructed with the builders above.
        Win_strength: multiplier applied to ``Win`` after construction.
        C, sigma: passed to :func:`build_weight_lsm_probabilistic` as
            ``C`` and ``l`` when ``Wlsm`` is not supplied.
        use_stdp: if True, allocate an internal :class:`AsymmetricSTDP`
            instance and allow ``forward(..., apply_stdp=True)`` to update
            ``Wlsm`` in place.
        stdp_params: kwargs forwarded to :class:`AsymmetricSTDP` (e.g.
            ``tau_plus``, ``tau_minus``, ``A_plus``, ``A_minus``).
        alpha, beta, th: RSynaptic/RLeaky synaptic and membrane decay factors
            and firing threshold.
        neuron_type: ``'RSynaptic'`` (default, used in the paper) or
            ``'Leaky'``.
        one_to_one: whether to use :func:`build_weight_in_layered_one_to_one`
            (default) instead of :func:`build_weight_in_layered` for ``Win``.
        inhibit: whether a random fraction of neurons are inhibitory, i.e.
            have all of their outgoing recurrent synapses negated in
            ``Wlsm``. Input weights are always excitatory.
        wlsm_seed: seed for the local RNG used to sample ``Wlsm``. Two
            reservoirs must be given different seeds to be independent.
    """

    def __init__(self,
                 n_layers: int = 77,
                 layer_h: int = 3,
                 layer_w: int = 3,
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
                 neuron_type: str = 'RSynaptic',
                 one_to_one: bool = True,
                 inhibit: bool = True,
                 wlsm_seed: Optional[int] = 48):
        super().__init__()
        self.n_layers = n_layers
        self.layer_h = layer_h
        self.layer_w = layer_w
        self.N = n_layers * layer_h * layer_w
        in_ch = n_layers

        # 3D neuron coordinates, used only to build Wlsm's distance-based
        # connection probabilities.
        self.positions = build_3d_positions(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w)

        # Win: input -> reservoir (tonotopic, one layer per input channel).
        if Win is None:
            if one_to_one:
                Win = build_weight_in_layered_one_to_one(
                    n_layers=n_layers,
                    layer_h=layer_h,
                    layer_w=layer_w,
                    n_inputs=in_ch,
                    win_strength=0.5,
                )
            else:
                Win = build_weight_in_layered(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w,
                                            n_inputs=in_ch, win_strength=0.5)
        Win = Win * Win_strength
        assert Win.shape == (self.N, in_ch)
        self.register_buffer("Win", Win)

        # Wlsm: recurrent reservoir connectivity, distance-decaying probability (Eq. 1).
        if Wlsm is None:
            Wlsm_init, prob = build_weight_lsm_probabilistic(self.positions, C=C, l=sigma,
                                                            w_mean=0.08, w_std=0.03, self_connection=False,
                                                            inhibit=inhibit, seed=wlsm_seed)
            Wlsm = Wlsm_init
            self.register_buffer("W_prob", prob)
        assert Wlsm.shape == (self.N, self.N)
        # Registered as a buffer since STDP updates it in place during forward().
        self.register_buffer("Wlsm", Wlsm)

        # Linear layer used only as a convenient container for Win inside nn.Module.
        self.fc1 = nn.Linear(in_ch, self.N, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(self.Win)

        # Recurrent spiking neuron layer (Eq. 2-3 of the paper: second-order
        # leaky integrate-and-fire with separate synaptic/membrane dynamics).
        self.neuron_type = neuron_type
        if self.neuron_type == 'RSynaptic':
            self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True,
                                    linear_features=self.N, threshold=th)
        elif self.neuron_type == 'Leaky':
            self.lsm = snn.RLeaky(beta=beta, all_to_all=True,
                                    linear_features=self.N, threshold=th)
        with torch.no_grad():
            self.lsm.recurrent.weight.copy_(self.Wlsm)

        # Online STDP (Eq. 4 of the paper), applied to Wlsm during forward()
        # when apply_stdp=True.
        self.use_stdp = bool(use_stdp)
        if self.use_stdp:
            stdp_args = stdp_params.copy() if stdp_params is not None else {}
            device_for_stdp = self.Wlsm.device if hasattr(self, "Wlsm") else None
            stdp_args["device"] = device_for_stdp
            self.stdp = AsymmetricSTDP(self.N, **stdp_args)

    def forward(self, x: torch.Tensor, apply_stdp: bool = False,
                clamp_min: Optional[float] = -1.0, clamp_max: Optional[float] = 1.0):
        """Run the reservoir over a batch of input sequences.

        Args:
            x: input tensor shaped ``(time_steps, batch, in_ch)``.
            apply_stdp: if True (and ``use_stdp=True`` at construction time),
                update ``Wlsm`` in place with :class:`AsymmetricSTDP` after
                every timestep, then copy the updated weights into the
                snnTorch recurrent layer so they take effect on the next
                step. STDP traces are reset at the start of each call.
            clamp_min, clamp_max: bounds applied to ``Wlsm`` after each STDP
                update.

        Returns:
            Spike tensor shaped ``(time_steps, batch, N)``.
        """
        num_steps, batch_size, channels = x.size()

        if self.neuron_type == 'RSynaptic':
            spk, syn, mem = self.lsm.init_rsynaptic()
        elif self.neuron_type == 'Leaky':
            spk, mem = self.lsm.init_rleaky()
        spk_rec = []

        if self.use_stdp and apply_stdp:
            self.stdp.reset()

        for step in range(num_steps):
            curr = self.fc1(x[step])  # (batch, N)
            if self.neuron_type == 'RSynaptic':
                spk, syn, mem = self.lsm(curr, spk, syn, mem)
            elif self.neuron_type == 'Leaky':
                spk, mem = self.lsm(curr, spk, mem)
            spk_rec.append(spk)

            if self.use_stdp and apply_stdp:
                spk_now = spk.detach()
                self.stdp.step_update(self.Wlsm, spk_now, t_step=step, clamp_min=clamp_min, clamp_max=clamp_max)
                with torch.no_grad():
                    self.lsm.recurrent.weight.copy_(self.Wlsm)

        return torch.stack(spk_rec)  # (time, batch, N)


if __name__ == '__main__':
    # Minimal usage example: build one reservoir, run a random input through
    # it with STDP enabled, and inspect the resulting spikes/weight changes.
    import matplotlib.pyplot as plt
    import time

    n_layers = 77
    layer_h = 3
    layer_w = 3
    in_ch = 77
    N = n_layers * layer_h * layer_w

    positions = build_3d_positions(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w)
    Wlsm_init, prob = build_weight_lsm_probabilistic(positions, C=1.0, l=3.4, w_mean=0.08, w_std=0.03, seed=42)
    Win = build_weight_in_layered(n_layers=n_layers, layer_h=layer_h, layer_w=layer_w, n_inputs=in_ch, win_strength=0.5)

    lsm = LSM(Win=Win, Wlsm=Wlsm_init.clone(), th=0.5, alpha=0.8, beta=0.8, use_stdp=True)
    stdp = lsm.stdp

    # Random example input; replace with a real (in_ch, time) tensor.
    input_data = torch.rand((77, 395))
    inputs = input_data.t().unsqueeze(1)  # (time, batch=1, in_ch)

    stdp.reset()
    start = time.time()
    spikes_rec = lsm.forward(inputs, apply_stdp=True, clamp_min=-1.0, clamp_max=1.0)
    elapsed = time.time() - start
    print('Simulation completed in {:.2f} s'.format(elapsed))

    W_final = lsm.Wlsm
    print("Spikes tensor shape:", spikes_rec.shape)  # (395, 1, 231)

    neurons_to_plot = min(22, N)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].set_title(f"Raster spikes (batch 0), first {neurons_to_plot} neurons")
    axes[0].imshow(spikes_rec[:, 0, :neurons_to_plot].t().detach().numpy(), aspect='auto')
    axes[0].set_ylabel("Neuron index")
    axes[0].set_xlabel("Time step")

    causality = stdp.get_causality(normalize=True)
    axes[1].set_title(f"Normalized causality ({N}x{N})")
    axes[1].imshow(causality.detach().numpy(), aspect='auto')
    axes[1].set_xlabel("post neuron index")
    axes[1].set_ylabel("pre neuron index")

    plt.tight_layout()
    plt.show()

    net = (stdp.C_pot - stdp.C_dep).abs().numpy()
    pairs = [(net[i, j], i, j) for i in range(N) for j in range(N)]
    top10 = sorted(pairs, key=lambda x: -x[0])[:10]
    print("Top 10 (abs net causal influence):")
    for val, i, j in top10:
        print(f"pre {i} -> post {j}: score {val:.6f}")

    with torch.no_grad():
        deltaW = W_final - Wlsm_init
        print("W change stats: mean {:.6f}, std {:.6f}, min {:.6f}, max {:.6f}".format(
            deltaW.mean().item(),
            deltaW.std().item(),
            deltaW.min().item(),
            deltaW.max().item()
        ))
