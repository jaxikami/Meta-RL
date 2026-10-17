"""The three arms (``ARMS``), their checkpoints and the agent each one runs."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.layout import SHIELD_OBS_DIM, PRIVILEGED_STATE_DIM
from metarl.agent.common import device
from metarl.agent.pearl import PEARLAgent
from metarl.training.config import (APN_PENALTY_WEIGHT, SEED, INCLUDE_BASELINE,
                                    NORMALIZE_REWARDS, REWARD_SCALE, USE_FILM,
                                    AUX_MODE, AUX_WEIGHT, AUX_SAMPLE_FRAC,
                                    META_AUX_STATE_TARGETS, NET_ACTOR_HIDDEN,
                                    NET_CRITIC_HIDDEN, NET_CRITIC_TOKEN_HIDDEN,
                                    NET_MIXER_LAYERS, NET_MIXER_HEADS,
                                    NET_DEPTH, NET_LAYER_NORM, NET_RESIDUAL,
                                    NET_OBS_NORM, NET_ENCODER_LAYER_NORM,
                                    NET_ENCODER_HIDDEN, NET_WEIGHT_DECAY,
                                    POLICY_DIR, ALPHA_MIN, META_KL_LAMBDA,
                                    META_ENCODER_LR, GATE_LOSS_WEIGHT,
                                    GATE_SHARPNESS, GATE_INIT_BIAS,
                                    META_EXPLORE_NOISE_BETA)


class ArmSpec:
    """Static configuration for one trained arm (see the lean study's)."""

    def __init__(self, name, label, multitask, use_context, lagrangian,
                 action_mask, apn, blurb, apn_soft=False, warm_start=None,
                 train_sensor_noise=True):
        """Record one arm's configuration.

        Args:
            name: Arm key, used for checkpoint and log filenames.
            label: Display label.
            multitask: Train on the disturbed multi-task pool rather than the
                single nominal task.
            use_context: Enable the context encoder and latent z.
            lagrangian: Enable the environment's dual multipliers.
            action_mask: Enable the structural stage mask.
            apn: Hard shield: executed actions are projected by the APN.
            blurb: One-line description.
            apn_soft: Soft shield: mapping penalty only, never projected.
            warm_start: Key of an arm whose trained checkpoint this arm starts
                from; it is then anchored to that policy, selected against it
                (NO_REGRESS_*) and, with context, gated (GATE_*).
            train_sensor_noise: Train on the noisy instrument model (False:
                exact observation; evaluation always adds the noise).
        """
        self.name = name
        self.label = label
        self.multitask = bool(multitask)
        self.use_context = bool(use_context)
        self.lagrangian = bool(lagrangian)
        self.action_mask = bool(action_mask)
        self.apn = bool(apn)
        self.apn_soft = bool(apn_soft)
        self.warm_start = warm_start
        self.train_sensor_noise = bool(train_sensor_noise)
        self.blurb = blurb

    def summary(self):
        """Render the arm's switches as a one-line string."""
        on = lambda b: "on" if b else "off"                # noqa: E731
        s = (f"multitask={on(self.multitask)} z={on(self.use_context)} "
             f"lagrangian={on(self.lagrangian)} mask={on(self.action_mask)} "
             f"apn={on(self.apn)}")
        if self.apn_soft and not self.apn:
            s += " apn_soft=on"
        if self.warm_start:
            s += f" warm={self.warm_start}"
        s += f" train_obs={'noisy' if self.train_sensor_noise else 'exact'}"
        return s


ARMS = {
    "baseline": ArmSpec(
        "baseline", "BASELINE", multitask=False, use_context=False,
        lagrangian=False, action_mask=False, apn=False, apn_soft=False,
        train_sensor_noise=False,
        blurb="unconstrained RL on the nominal regime: no duals, no mask, "
              "no shield"),
    "saferl": ArmSpec(
        "saferl", "SAFERL", multitask=False, use_context=False,
        lagrangian=True, action_mask=True, apn=True, apn_soft=False,
        train_sensor_noise=False,
        blurb="constrained RL on the nominal regime: duals + stage mask + "
              "action projection"),
    "meta": ArmSpec(
        "meta", "META", multitask=True, use_context=True,
        lagrangian=True, action_mask=True, apn=True, apn_soft=False,
        warm_start="saferl", train_sensor_noise=True,
        blurb="meta-RL across fresh disturbed regimes: duals + stage mask + "
              "inventory shield + latent task inference, from saferl's "
              "policy, which acts until the latent detects a fault"),
}

ARM_ORDER = ("baseline", "saferl", "meta")

CONTRASTS = (
    ("SAFETY", "saferl", "baseline",
     "saferl - baseline : the safe-RL stack (duals + mask + projection)"),
    ("ADAPT", "meta", "saferl",
     "meta - saferl : meta-RL on top of safe RL, across regimes"),
)


def active_arms():
    """Arms that train under ``all`` and run in ``evaluate``."""
    return tuple(a for a in ARM_ORDER if INCLUDE_BASELINE or a != "baseline")


def arm_paths(arm):
    """Checkpoint paths (final, best) for one arm."""
    name = arm.name if isinstance(arm, ArmSpec) else arm
    return (os.path.join(POLICY_DIR, f"{name}.pth"),
            os.path.join(POLICY_DIR, f"{name}_best.pth"))


META_PATH, BEST_META_PATH = arm_paths("meta")


def get_arm(arm):
    """Resolve an arm key or spec to an ``ArmSpec``."""
    if isinstance(arm, ArmSpec):
        return arm
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; expected one of "
                         f"{', '.join(ARM_ORDER)}")
    return ARMS[arm]


def _load_apn_for(spec, obs_dim, action_dim, train_if_missing=False):
    """Load the shared projection network when the arm calls for one."""
    if not (spec.apn or spec.apn_soft):
        return None
    if train_if_missing:
        from metarl.apn.train import ensure_apn
        return ensure_apn(obs_dim=obs_dim, action_dim=action_dim,
                          seed=SEED).to(device)
    from metarl.apn.network import load_apn
    return load_apn(obs_dim=obs_dim, action_dim=action_dim, device=device,
                    strict=True)


def _build_agent(env, arm, train_apn_if_missing=False):
    """Construct the agent for one arm."""
    spec = get_arm(arm)
    apn = _load_apn_for(spec, SHIELD_OBS_DIM, env.actionDim,
                        train_if_missing=train_apn_if_missing)
    kwargs = dict(reward_scale=REWARD_SCALE, normalize_rewards=NORMALIZE_REWARDS,
                  alpha_min=ALPHA_MIN,
                  use_film=USE_FILM, use_safeguard=spec.action_mask,
                  apn=apn, apn_penalty_weight=APN_PENALTY_WEIGHT,
                  apn_hard_project=spec.apn,
                  actor_hidden=NET_ACTOR_HIDDEN, critic_hidden=NET_CRITIC_HIDDEN,
                  critic_token_hidden=NET_CRITIC_TOKEN_HIDDEN,
                  mixer_layers=NET_MIXER_LAYERS, mixer_heads=NET_MIXER_HEADS,
                  net_depth=NET_DEPTH, layer_norm=NET_LAYER_NORM,
                  residual=NET_RESIDUAL, obs_norm=NET_OBS_NORM,
                  encoder_layer_norm=NET_ENCODER_LAYER_NORM,
                  encoder_hidden=NET_ENCODER_HIDDEN,
                  weight_decay=NET_WEIGHT_DECAY)
    if spec.use_context:
        kwargs.update(kl_lambda=META_KL_LAMBDA, encoder_lr=META_ENCODER_LR,
                      aux_mode=AUX_MODE, aux_weight=AUX_WEIGHT,
                      aux_sample_frac=AUX_SAMPLE_FRAC,
                      critic_task_context=True,
                      critic_privileged_dim=PRIVILEGED_STATE_DIM,
                      adapt_gate=arm_gated(spec),
                      gate_weight=GATE_LOSS_WEIGHT,
                      gate_sharpness=GATE_SHARPNESS,
                      gate_init_bias=GATE_INIT_BIAS,
                      explore_noise_beta=META_EXPLORE_NOISE_BETA,
                      aux_state_targets=(META_AUX_STATE_TARGETS
                                         if AUX_MODE == "task" else None))
    return PEARLAgent(env.stateDim, env.actionDim,
                      use_context=spec.use_context, **kwargs)


def arm_gated(arm):
    """Whether an arm acts through the adaptation gate (warm-started context arm)."""
    spec = get_arm(arm)
    return bool(spec.use_context and spec.warm_start)


def _load_preferring_best(agent, best_path, final_path, label, strict=False):
    """Load the best checkpoint if present, else the final one."""
    for p in (best_path, final_path):
        if os.path.exists(p):
            try:
                agent.load(p)
            except ValueError as exc:
                raise SystemExit(f"[!] {label}: {exc}") from None
            print(f"[load] {label}: {p}")
            return
    msg = f"[!] {label}: no checkpoint at {best_path} or {final_path}"
    if strict:
        raise SystemExit(msg)
    print(msg + " -- using random weights")
