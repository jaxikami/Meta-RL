# Meta-RL for Uncertainty

An uncertainty-aware meta-reinforcement-learning study for **safe control of a
multi-reactor lignocellulosic fermentation plant**. The central question is
whether a meta-RL agent that infers the operating regime online can hold
production and stay within safety constraints under conditions a conventional
safe-RL policy was never trained for — new feedstocks, drifting kinetics, and
abrupt mid-run disturbances.

## The idea

Real fermentation plants never run at their nominal set point. Feedstock
composition varies batch to batch, enzyme lots differ in efficacy, strain
toxin-tolerance shifts, and pH excursions move the reaction kinetics. A policy
tuned to one nominal regime degrades — and can violate physical safety limits —
when the true process drifts away from it.

This project frames each operating regime as a **task**: one draw of latent
process parameters `theta ~ p(theta)`, held fixed within an episode and never
observed directly by the agent. The agent must instead *infer* the regime from
the transitions it sees and adapt its control accordingly. We then ask whether
that adaptation actually buys safety and performance over a strong non-adaptive
baseline, on regimes inside and beyond the training distribution.

## The comparison

Two agents share an **identical safety stack**, so meta-learning is the only
variable between them:

- **Meta agent (PEARL)** — infers a latent task variable `z` from recent
  transitions and conditions its policy on it.
- **Non-adaptive baseline** — the same architecture with `z ≡ 0`, trained on the
  nominal task only. This is ordinary safe RL with no task inference.

Both use the same two-layer safety mechanism: a **SERL stage mask** that
structurally locks action dimensions that are illegal in the current process
stage, and the environment's **Lagrangian constraint penalties**, which grow
when overflow/underflow limits are approached and shape them out of the policy.

The hypothesis: on in-distribution regimes the two should be close (the baseline
is not disadvantaged there), while on out-of-distribution regimes and
mid-episode disturbances the non-adaptive baseline should lose reward and/or
incur more violations, because it cannot recognise that the process has changed.

## What's in the repository

- `env_core.py` — the fermentation ODE physics and reaction kinetics
  (`calculate_rates_numba`, `kinetics_from_pH`), parameterised by a length-12
  latent vector whose nominal values reproduce the original physics exactly.
- `env_config.py` — structural plant limits, feed compositions, and the task
  distribution `TASK_PRIORS`: `sample_task` / `nominal_task` draw a regime, and
  `build_param_vector` assembles the physics vector. Four bounded,
  physically-motivated randomization groups (kinetics via operating pH,
  toxin sensitivity, enzyme adsorption, feedstock composition).
- `env_safe.py` — `FermentationEnv`, the heterogeneous 3-reactor environment
  with the SERL stage mask, Lagrangian constraint shaping, per-episode task
  installation (`set_task`), and mid-episode disturbance injection
  (`schedule_disturbance`: inlet step, enzyme drop, mass drift).
- `agent_pearl.py` — the PEARL meta-RL agent: a permutation-invariant
  probabilistic context encoder `q(z|c)`, a tanh-Gaussian SAC actor with
  cross-reactor attention, twin soft critics, an off-policy SAC update with a
  KL information bottleneck on the encoder, and automatic entropy tuning. The
  non-adaptive baseline is the same class built with `use_context=False`.
- `agent_safe.py` — the shared Safe-PPO network components (including the
  cross-reactor attention block reused by the PEARL actor).
- `train_pearl_single_run.py` — the meta-training and evaluation entrypoint;
  runs ONE seed through train_meta -> train_baseline -> evaluate.
- `train_pearl_seed_sweep.py` — runs `train_pearl_single_run`'s train_meta/
  train_baseline/evaluate under several shared seeds and reports the paired
  meta-vs-baseline difference across seeds (mean ± std, "meta wins k/n"). The
  training pipeline is high-variance run-to-run, so a single run of
  `train_pearl_single_run.py` is NOT sufficient evidence for a meta-vs-baseline
  verdict — use this sweep for that.
- `smoke_test_pearl.py` — a fast CPU check that exercises every code path
  (shapes, imports, wiring) without doing real training.

## How to run

```bash
# 1. fast CPU sanity check of all code paths (no training)
python smoke_test_pearl.py     # prints "ALL SMOKE CHECKS PASSED"

# 2. meta-train PEARL, train the non-adaptive baseline, then compare (one seed)
python train_pearl_single_run.py all
#   or individually:
python train_pearl_single_run.py train_meta
python train_pearl_single_run.py train_baseline
python train_pearl_single_run.py evaluate

# 3. for an actual meta-vs-baseline verdict, sweep several seeds instead of
#    trusting a single run (see note above on run-to-run variance):
python train_pearl_seed_sweep.py            # seeds 0,1,2
python train_pearl_seed_sweep.py 0 1 2 3 4  # explicit seed list
```

Hyperparameters (task-pool size, meta-iterations, batch/context sizes, reward
scale, warm-up, priors, disturbance magnitudes) are at the top of
`train_pearl_single_run.py` and in `env_config.TASK_PRIORS`.

## What `evaluate` produces

`train_pearl_single_run.py evaluate` runs both agents on **in-distribution** tasks (unseen
draws from the training prior), genuinely **out-of-distribution** tasks (widened
priors, beyond the training ranges), and three **mid-episode disturbances**
(inlet step, enzyme drop, mass drift).

It prints and saves to `policy/pearl_eval.json`:

- a reward / violations / feasible-rate table (mean ± std) per scenario, per agent;
- a per-scenario verdict (Δreward and fewer-violations, meta − baseline);
- disturbance-recovery metrics (post-onset mean reward, post-onset violation
  rate, recovery time in control steps),

and renders two figures under `policy/`: `pearl_compare_bars.png` (grouped
reward/violation bars per scenario) and `pearl_disturbance_response.png`
(per-step reward trajectories around each disturbance).

**How to read it.** The in-distribution row is the control; the OOD and
disturbance rows are the test. The claim "basic safe RL can't handle
uncertainty" is supported when the baseline shows markedly lower reward and/or
more violations than the meta agent on the test rows, while the two stay close
on the control row.

## Notes and caveats

- Both agents share the identical safety stack, so the meta-vs-baseline
  comparison is fair by construction.
- The safety guarantee is two-layered but soft on physical limits: the SERL mask
  makes structurally-illegal actions impossible, while overflow/underflow limits
  are enforced through Lagrangian reward penalties rather than a hard projection.
- The critic is trained on the executed (stage-masked) action while the actor
  optimizes its own differentiable raw action; both agents share this.
- Run `smoke_test_pearl.py` first on any new machine to confirm shapes and
  imports before committing to a full training run.
