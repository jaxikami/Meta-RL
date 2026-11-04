"""Training of the APN's learned heads (apn.KineticNet).

The APN's exact heads are the plant's own arithmetic and are not trained.  The
learned heads regress what one reactor does if its action is held
(metarl/apn/data.py): its heat-load profile over the next HEAT_HORIZON_H hours
and its titer when unload begins.  Both losses are Huber losses on outputs
scaled by apn.HEAT_REF / apn.TITER_REF; the titer loss is weighted towards the
purity spec, where the purity head's sign is decided.

Usage:
    python -m metarl.apn.data       # simulate and label the dataset
    python -m metarl.apn.train      # fit KineticNet -> policy/apn_kinetic.pth
    python -m metarl.apn.validate   # check the full APN
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.apn.data import DATA_PATH
from metarl.apn.constants import HEAT_REF, KINETIC_PATH, TITER_REF
from metarl.apn.kinetic import KineticNet
from metarl.env.config import MIN_PRODUCT_TITER

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Titer samples within a few g/L of the purity spec weigh up to 3x more.
SPEC_EMPHASIS = 2.0
SPEC_WIDTH = 5.0


def to_tensors(data, device):
    """Dataset dict -> (obs, action, heat, titer, has_titer) tensors on ``device``."""
    titer = np.nan_to_num(data["titer"], nan=0.0)
    return (torch.as_tensor(data["obs"], device=device),
            torch.as_tensor(data["action"], device=device),
            torch.as_tensor(data["heat"], device=device),
            torch.as_tensor(titer, device=device),
            torch.as_tensor(~np.isnan(data["titer"]), device=device))


def kinetic_loss(model, obs, action, heat, titer, has_titer):
    """Huber loss of KineticNet on one batch (scaled targets)."""
    heat_p, titer_p = model(obs, action)
    l_heat = F.smooth_l1_loss(heat_p / HEAT_REF, heat / HEAT_REF, beta=0.02)
    w = (1.0 + SPEC_EMPHASIS * torch.exp(-(titer - MIN_PRODUCT_TITER).abs() / SPEC_WIDTH)) * has_titer
    l_titer = (F.smooth_l1_loss(titer_p / TITER_REF, titer / TITER_REF, beta=0.02, reduction="none")
               * w).sum() / w.sum().clamp(min=1.0)
    return l_heat + l_titer


@torch.no_grad()
def evaluate(model, obs, action, heat, titer, has_titer, batch=16384):
    """Validation metrics: heat MAE/R2, titer MAE and purity-sign accuracy."""
    model.eval()
    hp, tp = [], []
    for i in range(0, len(obs), batch):
        h, t = model(obs[i:i + batch], action[i:i + batch])
        hp.append(h)
        tp.append(t)
    hp, tp = torch.cat(hp), torch.cat(tp)
    res = (hp - heat).pow(2).sum()
    tot = (heat - heat.mean()).pow(2).sum()
    t_true, t_pred = titer[has_titer], tp[has_titer]
    near = (t_true - MIN_PRODUCT_TITER).abs() < SPEC_WIDTH
    sign = (t_pred >= MIN_PRODUCT_TITER) == (t_true >= MIN_PRODUCT_TITER)
    return {
        "heat_mae": float((hp - heat).abs().mean()),
        "heat_r2": float(1.0 - res / tot),
        "titer_mae": float((t_pred - t_true).abs().mean()),
        "purity_sign_acc": float(sign.float().mean()),
        "purity_sign_acc_near_spec": float(sign[near].float().mean()) if near.any() else None,
    }


def fit(model, train, steps, lr, batch=4096, weight_decay=1e-5, val=None, log_every=0):
    """Fit KineticNet for ``steps`` minibatch steps (cosine-decayed AdamW).

    Args:
        model: KineticNet (trained in place).
        train: Tuple from to_tensors().
        steps: Minibatch steps.
        lr: Peak learning rate.
        batch: Minibatch size.
        weight_decay: AdamW weight decay.
        val: Optional validation tuple, evaluated every ``log_every`` steps.
        log_every: Logging period (0: never).

    Returns:
        The last validation metrics (or None).
    """
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    n = len(train[0])
    metrics = None
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch,), device=train[0].device)
        loss = kinetic_loss(model, *(t[idx] for t in train))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and val is not None and (step % log_every == 0 or step == steps):
            metrics = evaluate(model, *val)
            print(f"  step {step:6d} | loss {loss.item():.5f} | heat MAE {metrics['heat_mae']:.1f} kg/h "
                  f"R2 {metrics['heat_r2']:.4f} | titer MAE {metrics['titer_mae']:.2f} g/L | "
                  f"purity sign {metrics['purity_sign_acc']:.4f} "
                  f"(near spec {metrics['purity_sign_acc_near_spec']:.3f})", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return metrics

def run_pretraining(data_path=DATA_PATH, out_path=KINETIC_PATH, steps=20000, lr=1e-3,
                    val_frac=0.1, seed=0):
    """Train KineticNet on the generated dataset and save it.

    Args:
        data_path: Dataset from metarl.apn.data.generate().
        out_path: Checkpoint path.
        steps: Minibatch steps.
        lr: Peak learning rate.
        val_frac: Held-out share.
        seed: Seed.

    Returns:
        Validation metrics of the saved model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    data = dict(np.load(data_path))
    n = len(data["obs"])
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_frac)
    val = to_tensors({k: v[perm[:n_val]] for k, v in data.items()}, device)
    train = to_tensors({k: v[perm[n_val:]] for k, v in data.items()}, device)
    print(f"Training KineticNet on {n - n_val:,} pairs ({n_val:,} held out), device {device}")

    model = KineticNet().to(device)
    metrics = fit(model, train, steps, lr, val=val, log_every=max(steps // 20, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, out_path)
    print(f"Saved {out_path}")
    return metrics


if __name__ == "__main__":
    run_pretraining()
