"""Offline pre-training and validation for the plant's action projection network.

Trained once on synthetic data and frozen. It depends on no policy, seed or
run, so every arm and seed shares one checkpoint (``network.APN_PATH``).

Usage::

    python -m metarl.apn.train            # train with the defaults below
    python -m metarl.apn.train 12000      # train for up to 12000 epochs
    python -m metarl.apn.train validate   # validate the existing checkpoint

The loss is the lean study's: a logistic term on the margin's SIGN, weighted
toward the boundary, plus a regression term on its VALUE.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from metarl.apn.data import (assert_plant_constants_match, plant_constants,
                             sample_actions, sample_labeled_batch,
                             sample_states, safety_margin, shut_action)
from metarl.apn.network import (APN_PATH, ActionProjectionNetwork,
                                MARGIN_SCALE, PROJ_REPAIR_MARGIN,
                                PROJ_REPAIR_THRESHOLD, PROJ_TARGET_MARGIN,
                                PROJ_THRESHOLD, load_apn, project_to_safe,
                                save_apn)
from metarl.env.layout import (SHIELD_OBS_DIM, ACTION_DIM, TIME_ACTION_DIMS,
                               SHIELD_IGNORED_CHANNELS)
from metarl.env.adapter import PlantFermentationEnv

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

device = torch.device(os.environ.get("PLANT_DEVICE")
                      or ("cuda" if torch.cuda.is_available() else "cpu"))

# ─── Schedule ─────────────────────────────────────────────────────────────────
SEED = int(os.environ.get("PLANT_SEED", 0))
# Longer than the lean study's 6000: seven margins over three reactors and the
# tank, on 96 inputs instead of 30.
EPOCHS = 12000
BATCH_SIZE = 4096
BUFFER_SIZE = 400000
REFRESH_INTERVAL = 50
LR = 8e-4
WEIGHT_DECAY = 1e-4
BCE_MIX = 0.6
APN_LATENT_DIM = 128
APN_ENCODER_DIM = 128

# ─── Early stop ───────────────────────────────────────────────────────────────
TARGET_SIGN_ACC = 0.965
MIN_EPOCHS = 2000
VALIDATE_EVERY = 500
# Longer than the cosine schedule's last cycle (epochs 4000-12000), so a
# stop never catches the shield at a high learning rate: at 3000 the plant's
# shield stopped at epoch 6500 with repair 0.87; run through, it reached sign
# 0.992 / boundary 0.971 / repair 0.90 (0.97 on the hard constraints).
PATIENCE = 9000

# ─── Validation gates (the lean study's) ─────────────────────────────────────
VAL_MIN_SIGN_ACC = 0.95
VAL_MIN_BOUNDARY_ACC = 0.85
VAL_MIN_REPAIR_RATE = 0.90
VAL_MIN_NOOP_RATE = 0.85
VAL_MIN_SAFE_PRESERVED = 0.98
VAL_MAX_MASK_DRIFT = 1e-6
NOOP_MARGIN = PROJ_TARGET_MARGIN + 0.05


def _loss(pred_margin, true_margin):
    """Composite boundary-weighted classification plus value regression."""
    sign_target = (true_margin >= 0.0).float()
    w = 1.0 / (true_margin.abs() * MARGIN_SCALE + 0.5)
    w = w / (w.mean() + 1e-8)
    l_bce = F.binary_cross_entropy_with_logits(pred_margin, sign_target, weight=w)
    l_reg = F.smooth_l1_loss(pred_margin, true_margin * MARGIN_SCALE)
    return BCE_MIX * l_bce + (1.0 - BCE_MIX) * l_reg, l_bce, l_reg


def _refresh_buffer(n, rng, const):
    """Generate a fresh stratified training buffer on ``device``."""
    o, a, m = sample_labeled_batch(n, rng, const=const)
    return o.to(device), a.to(device), m.to(device)


def new_apn():
    """An untrained shield of the plant's shape."""
    return ActionProjectionNetwork(SHIELD_OBS_DIM, ACTION_DIM,
                                   latent_dim=APN_LATENT_DIM,
                                   encoder_dim=APN_ENCODER_DIM,
                                   inert_dims=TIME_ACTION_DIMS,
                                   ignored_obs=SHIELD_IGNORED_CHANNELS)


# =============================================================================
# VALIDATION
# =============================================================================

def validate(net, rng, const, n=20000, verbose=True):
    """Run the five-part validation suite on a frozen network.

    Sign and boundary accuracy grade the regressor; repair, no-op /
    safe-preserved and mask drift grade the projection operator the agent
    actually calls (inert clocks included in the drift test).

    Returns:
        Dict of test results plus a boolean ``passed``.
    """
    was_training = net.training
    net.eval()
    res = {}
    dev = next(net.parameters()).device

    o, a, m = sample_labeled_batch(n, rng, const=const)
    o, a, m = o.to(dev), a.to(dev), m.to(dev)
    with torch.no_grad():
        pred = net(o, a)
    correct = (pred >= 0.0) == (m >= 0.0)
    res["sign_acc"] = float(correct.float().mean())
    near = m.abs() < 0.05
    res["boundary_acc"] = float(correct[near].float().mean()) if near.any() else 1.0

    o2 = torch.from_numpy(sample_states(n, rng)).to(dev)
    a2 = torch.from_numpy(sample_actions(n, rng)).to(dev)
    shut = shut_action(n, device=dev)
    with torch.no_grad():
        true_now = safety_margin(o2.cpu(), a2.cpu(), const=const).to(dev)
        recoverable = safety_margin(o2.cpu(), shut.cpu(), const=const).to(dev) > 0.0
    sel = (true_now < 0.0) & recoverable
    if sel.any():
        mask = PlantFermentationEnv.get_action_mask(o2[sel])
        proj, _ = project_to_safe(net, o2[sel], a2[sel], stage_mask=mask)
        with torch.no_grad():
            after = safety_margin(o2[sel].cpu(), proj.cpu(), const=const)
        res["repair_rate"] = float((after > 0.0).float().mean())
        res["n_repair"] = int(sel.sum())
    else:                                                   # pragma: no cover
        res["repair_rate"], res["n_repair"] = 1.0, 0

    sel_safe = true_now > NOOP_MARGIN
    if sel_safe.any():
        mask = PlantFermentationEnv.get_action_mask(o2[sel_safe])
        proj, _ = project_to_safe(net, o2[sel_safe], a2[sel_safe], stage_mask=mask)
        moved = (proj - a2[sel_safe]).abs().max(dim=-1).values
        res["noop_rate"] = float((moved < 1e-8).float().mean())
        res["mean_disturb"] = float(moved.mean())
        with torch.no_grad():
            after_safe = safety_margin(o2[sel_safe].cpu(), proj.cpu(), const=const)
        res["safe_preserved"] = float((after_safe > 0.0).float().mean())
    else:                                                   # pragma: no cover
        res["noop_rate"] = res["safe_preserved"] = 1.0
        res["mean_disturb"] = 0.0

    # Locked dimensions AND the inert stage clocks must never move.
    mask_all = PlantFermentationEnv.get_action_mask(o2)
    frozen = 1.0 - mask_all
    frozen[:, list(TIME_ACTION_DIMS)] = 1.0
    proj_all, _ = project_to_safe(net, o2, a2, stage_mask=mask_all)
    res["mask_drift"] = float(((proj_all - a2).abs() * frozen).max())

    rows = [("sign accuracy", res["sign_acc"], VAL_MIN_SIGN_ACC, ">="),
            ("boundary accuracy", res["boundary_acc"], VAL_MIN_BOUNDARY_ACC, ">="),
            ("repair rate", res["repair_rate"], VAL_MIN_REPAIR_RATE, ">="),
            ("no-op rate", res["noop_rate"], VAL_MIN_NOOP_RATE, ">="),
            ("safe preserved", res["safe_preserved"], VAL_MIN_SAFE_PRESERVED, ">="),
            ("mask drift", res["mask_drift"], VAL_MAX_MASK_DRIFT, "<=")]
    res["passed"] = all((v >= g) if op == ">=" else (v <= g)
                        for _, v, g, op in rows)
    if verbose:
        print(f"\n{'test':<22}{'value':>12}{'gate':>12}{'':>8}")
        for name, val, gate, op in rows:
            ok = (val >= gate) if op == ">=" else (val <= gate)
            print(f"{name:<22}{val:>12.4f}{gate:>12.4f}"
                  f"{'PASS' if ok else 'FAIL':>8}")
        print(f"{'':<22}{'':>12}{'':>12}"
              f"{'ALL PASS' if res['passed'] else 'FAILED':>8}")
        print(f"\n  target margin {PROJ_TARGET_MARGIN} (sigmoid "
              f"{PROJ_THRESHOLD:.4f}); repaired actions pushed to "
              f"{PROJ_REPAIR_MARGIN} (sigmoid {PROJ_REPAIR_THRESHOLD:.4f}); "
              f"mean disturbance on comfortably-safe inputs "
              f"{res['mean_disturb']:.4f}; repair tested on "
              f"{res.get('n_repair', 0)} recoverable unsafe draws")
    if was_training:
        net.train()
    return res


# =============================================================================
# TRAINING
# =============================================================================

def train(epochs=EPOCHS, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE,
          path=None, seed=SEED):
    """Train the APN from scratch and save it.

    Returns:
        Tuple of (trained network, final validation dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = PlantFermentationEnv()
    assert_plant_constants_match(env)
    const = plant_constants()
    net = new_apn().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=100, T_mult=3, eta_min=1e-5)

    print(f"\n{'=' * 78}\nTRAIN ACTION PROJECTION NETWORK  "
          f"(obs={SHIELD_OBS_DIM}, act={ACTION_DIM}, device={device})\n{'=' * 78}")
    print("  margins: reactor overflow / underflow (x3) and storage overflow")
    print(f"  epochs<={epochs}  batch={batch_size}  buffer={buffer_size}  seed={seed}")

    buf_o, buf_a, buf_m = _refresh_buffer(buffer_size, rng, const)
    # Ranked on (passed every gate, score): a check that clears the gates
    # always outranks one that does not, so the restored checkpoint is never a
    # failing one with a slightly higher score (the lean trainer's ranking on
    # score alone restored such a check after an early stop on a passing one).
    best_key, best_state, best_epoch = None, None, 0
    t0 = time.time()
    ep = 0
    for ep in range(1, epochs + 1):
        if ep % REFRESH_INTERVAL == 0:
            buf_o, buf_a, buf_m = _refresh_buffer(buffer_size, rng, const)
        idx = torch.randint(0, buf_o.shape[0], (batch_size,), device=device)
        o, a, m = buf_o[idx], buf_a[idx], buf_m[idx]
        pred = net(o, a)
        loss, l_bce, l_reg = _loss(pred, m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()

        if ep % 250 == 0 or ep == 1:
            with torch.no_grad():
                acc = float((((pred >= 0.0) == (m >= 0.0)).float()).mean())
            print(f"  ep {ep:>6}  loss {loss.item():>7.4f}  bce {l_bce.item():>7.4f}"
                  f"  reg {l_reg.item():>7.4f}  sign-acc {acc:>6.4f}"
                  f"  lr {sched.get_last_lr()[0]:.2e}  [{time.time() - t0:>5.0f}s]")

        if ep >= MIN_EPOCHS and ep % VALIDATE_EVERY == 0:
            val = validate(net, rng, const, n=20000, verbose=False)
            score = val["repair_rate"] + val["sign_acc"] + val["boundary_acc"]
            key = (bool(val["passed"]), score)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, ep
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
            print(f"  [val ep {ep}] sign {val['sign_acc']:.4f}  boundary "
                  f"{val['boundary_acc']:.4f}  repair {val['repair_rate']:.4f}  "
                  f"noop {val['noop_rate']:.4f}  safe {val['safe_preserved']:.4f}"
                  f"  {'PASS' if val['passed'] else '----'}")
            if val["passed"] and val["sign_acc"] >= TARGET_SIGN_ACC:
                print(f"  [early stop] gates cleared at epoch {ep}")
                break
            if ep - best_epoch >= PATIENCE:
                print(f"  [early stop] no validation improvement in {PATIENCE} "
                      f"epochs (best at {best_epoch})")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\n  restored best checkpoint from epoch {best_epoch}")
    print(f"\n{'=' * 78}\nFINAL VALIDATION\n{'=' * 78}")
    final = validate(net, rng, const, n=50000, verbose=True)
    p = save_apn(net, path, meta={
        "seed": seed, "epochs_run": ep, "best_epoch": best_epoch,
        "validation": {k: v for k, v in final.items()},
    })
    print(f"\nwrote {p}")
    if not final["passed"]:
        print("  [!] the APN did not clear its validation gates. The shielded "
              "arms will still run, but the projection is not doing what it "
              "claims to; train longer before reporting a comparison.")
    return net, final


def ensure_apn(path=None, obs_dim=SHIELD_OBS_DIM, action_dim=ACTION_DIM,
               epochs=EPOCHS, seed=SEED):
    """Load the APN, training it first if no valid checkpoint exists.

    Returns:
        A frozen ``ActionProjectionNetwork``.
    """
    p = APN_PATH if path is None else path
    if os.path.exists(p):
        try:
            return load_apn(p, obs_dim, action_dim, device=device)
        except SystemExit as err:
            print(f"[apn] existing checkpoint at {p!r} is invalid for the "
                  f"current configuration:\n      {err}\n      retraining ...")
    print(f"[apn] no valid checkpoint at {p!r} -- training one now (once, "
          f"shared by every arm and seed).")
    train(epochs=epochs, path=p, seed=seed)
    return load_apn(p, obs_dim, action_dim, device=device)


def main(argv):
    """CLI entry point; returns the process exit code."""
    arg = argv[1] if len(argv) > 1 else None
    if arg == "validate":
        net = load_apn(device=device)
        rng = np.random.default_rng(SEED + 1)
        res = validate(net, rng, plant_constants(), n=50000, verbose=True)
        return 0 if res["passed"] else 1
    epochs = int(arg) if arg and arg.isdigit() else EPOCHS
    _, final = train(epochs=epochs)
    return 0 if final["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
