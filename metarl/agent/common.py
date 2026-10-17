"""Devices, shared constants and small utilities of the PEARL agent."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import copy
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

# Small networks on small batches: the autotuner's warm-up costs more than it
# saves.
torch.backends.cudnn.enabled = False
if hasattr(torch.backends, 'miopen'):
    torch.backends.miopen.enabled = False

# PLANT_DEVICE forces a device (rollout workers set "cpu"); otherwise the GPU
# when one is available.
device = torch.device(os.environ.get("PLANT_DEVICE")
                      or ("cuda" if torch.cuda.is_available() else "cpu"))

# Acting and the online posterior run batch-1 forward passes every control
# step; on a GPU kernel launches and syncs cost more than that arithmetic, so
# rollouts always run on the CPU (see PEARLAgent._actor_r).
ROLLOUT_DEVICE = torch.device("cpu")


def _on_rollout_device(module):
    """Return ``module`` if it already lives on ROLLOUT_DEVICE, else a copy there."""
    if module is None:
        return None
    if next(module.parameters()).device.type == ROLLOUT_DEVICE.type:
        return module
    return copy.deepcopy(module).to(ROLLOUT_DEVICE)


# The lean study used 5 latent dimensions for a 6-entry task descriptor; the
# plant's descriptor has 15 (three reactors' gains, leaks, kinetic activity
# and contamination, the feed and the downstream rate), and the latent must
# also say WHICH reactor a fault is in.
LATENT_DIM = 8
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
PRE_TANH_CLIP = 9.0
LOG_ALPHA_MIN, LOG_ALPHA_MAX = -5.0, 3.0
ENTROPY_TERM_CLIP = 50.0

# Network-architecture identifier stamped in every checkpoint.
NET_ARCH = "plant_token_v1"

# Length of one pre-drawn colored exploration sequence, in control steps. An
# episode is ~120-135 (600 h at 5 h, plus an early end at every batch
# boundary of any reactor).
EXPLORE_NOISE_LEN = 256

# Hard bound on the encoder's posterior means (see the lean agent).
MU_CLAMP = 6.0


class RunningMeanStd:
    """Bounded-window running mean/variance estimator for reward normalisation.

    Parallel variance merge with the sample count capped at ``max_count``, so
    the estimate tracks the current policy. Update it only from freshly
    collected steps. Only the std is consumed (std-only scaling preserves the
    sign and ranking of rewards).
    """

    def __init__(self, eps=1e-4, max_count=1e5):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.max_count = float(max_count)

    def update(self, x):
        """Fold a batch of scalars into the estimate."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        b_mean, b_var, b_count = float(x.mean()), float(x.var()), int(x.size)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta * delta * self.count * b_count / tot) / tot
        # Plain floats: numpy scalars make torch.load(weights_only=True)
        # reject the checkpoint.
        self.count = float(min(tot, self.max_count))

    @property
    def std(self):
        """Standard deviation with a floor, safe as a divisor."""
        return float(np.sqrt(self.var)) + 1e-6


class RunningObsNorm(nn.Module):
    """Per-channel running standardisation of observations.

    Statistics are buffers, so they travel with every ``state_dict``. One
    instance is shared by the actor, both critics and the encoder, and moves
    only through ``update`` from freshly collected observations. Constant
    channels (masked concentrations) read 0.
    """

    def __init__(self, dim, clip=10.0, eps=1e-4, max_count=1e5):
        super().__init__()
        self.clip = float(clip)
        self.eps = float(eps)
        self.max_count = float(max_count)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        """Fold a batch of observations, shape [N, dim], into the statistics."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float32),
                            device=self.mean.device).reshape(-1, self.mean.numel())
        n = x.shape[0]
        if n == 0:
            return
        b_mean = x.mean(dim=0)
        b_var = x.var(dim=0, unbiased=False)
        tot = self.count + n
        delta = b_mean - self.mean
        self.var.copy_((self.var * self.count + b_var * n
                        + delta.pow(2) * self.count * n / tot) / tot)
        self.mean.add_(delta * n / tot)
        self.count.fill_(min(float(tot), self.max_count))

    def forward(self, x):
        """Standardise ``x`` (any leading shape, last dimension ``dim``)."""
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps),
                           -self.clip, self.clip)


def _adamw(module, lr, weight_decay):
    """AdamW that decays only weight matrices, never FiLM projections.

    Biases, norms and the FiLM projections are exempt: decaying FiLM would
    pull every layer back toward ignoring z. With ``weight_decay=0`` this is
    exactly Adam.
    """
    decay, keep = [], []
    for name, p in module.named_parameters():
        if p.ndim >= 2 and "films" not in name.split("."):
            decay.append(p)
        else:
            keep.append(p)
    groups = [{"params": ps, "weight_decay": wd}
              for ps, wd in ((decay, float(weight_decay)), (keep, 0.0)) if ps]
    return torch.optim.AdamW(groups, lr=lr)


def powerlaw_noise(beta, n, dims, rng=None):
    """Gaussian noise with a 1 / f**beta power spectrum along time.

    Timmer & Koenig's method (as in ``colorednoise``); beta = 0 is white, 1
    pink. Colored-noise exploration (Eberhard et al., ICLR 2023).

    Returns:
        Float32 array of shape (n, dims), zero mean and unit variance in
        expectation.
    """
    rng = np.random if rng is None else rng
    n = int(n)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0 / n
    s = f ** (-float(beta) / 2.0)
    w = s[1:].copy()
    w[-1] *= (1 + (n % 2)) / 2.0
    sigma = 2.0 * np.sqrt(np.sum(w ** 2)) / n
    sr = rng.normal(size=(dims, f.size)) * s
    si = rng.normal(size=(dims, f.size)) * s
    if n % 2 == 0:
        si[:, -1] = 0.0
        sr[:, -1] *= np.sqrt(2.0)
    si[:, 0] = 0.0
    sr[:, 0] *= np.sqrt(2.0)
    y = np.fft.irfft(sr + 1j * si, n=n, axis=-1) / sigma
    return np.ascontiguousarray(y.T, dtype=np.float32)
