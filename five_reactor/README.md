# Five-reactor sub-study: meta-RL under a known mid-episode change

The question:

> On a plant whose shared utilities cannot keep five vessels busy, can a
> meta-RL agent infer the size of a known change from the plant's response
> and re-pace production to it? Does it keep doing so when the change is
> larger or smaller than anything it trained on?

There is one arm, `meta5`. It is the parent study's `meta` arm trained from
scratch, with no saferl and no T1–T3 tiers. Everything not listed here is
the parent study's (`../README.md`): the plant model, kinetics, observation,
reward, constraints, APN, networks, and training and selection machinery.

## The plant

| unit | spec |
|---|---|
| reactors R0–R4 | 220 t each, the three-reactor plant's vessel; feed pumps 1× / 4× / 0.5× / 1× / 0.5× (its three pump types) |
| feedstock pool, storage tank, downstream draw, cooling loop | the three-reactor plant's, unchanged: 2,000 t LF and 600 t C5; 400 t tank, 5 t floor, starts at 350 t; 1.5 t/h downstream; CO₂ evolution ≤ 540 kg/h |

It is selected with `PLANT_LAYOUT=five` (`metarl/env/config.py`, META half).
Importing `five_reactor` sets that before `metarl` loads, and worker
processes inherit it.
- **Observation, action, task descriptor:** the observation is (140,), five
  of the parent study's 28-channel rows. The action is (20,). The task
  descriptor has 26 entries, with the same per-reactor groups for five
  reactors.
- **What stays exact:** each vessel's hardware is one of the three-reactor
  plant's types. KineticNet (`policy/apn_kinetic.pth`, shared with the
  parent study) is used in distribution. The APN's plant-wide constants
  (`S_MAX`, `MAX_HEAT_LOAD`, `S_OUT_RATE`) stay true, and no ported file
  changes.
- **The coupling is tighter than with three vessels.** Five vessels running
  freely make far more than the tank absorbs, and more fed-batch heat than
  the cooling loop removes. Production is bounded by the utilities, not the
  vessels. Some vessels must wait, and how many can run depends on the
  downstream draw.
- **A first design was rejected.** It used five 132 t vessels, the same 660 t
  in total. The fixed 10 t charge diluted every lot to ~74 g/L, under the
  75 g/L purity spec, for every scripted recipe. The study would have been
  about purity, not the change.

## The known change

At hour 300 of every 600 h episode, the tank's downstream draw steps to
**m × 1.5 t/h** (the `downstream_rate` fault, a step, plant-wide). The kind
and the onset are known; m is not.

| | m | |
|---|---|---|
| training | N(1.30, 0.08), cut at ±3 SD | every training episode |
| evaluation | N(1.30, 0.25), cut to [0.5, 2.1] | ~11% of draws fall below 1.0: the draw drops instead |

`five_reactor/config.py` holds all of these. Evaluation rows are *in
support* when |m − 1.30| ≤ 2 × 0.08, and *outside* otherwise.

Why the draw, and not, say, +40% heat release:
- **It is observable.** The storage level is a channel and the agent knows its
  own discharges, so m is in the level's slope within a few steps. A heat
  change shows only through violations of id 9. The APN's L3 relocates the
  cap after the first alarm for any policy, which is why the parent study's
  invisible-limit faults tied.
- **The right response is a schedule the APN cannot supply.** A larger draw
  needs more vessels running and earlier batches. A smaller one needs
  finished batches held. The storage heads only clip discharge, and
  `AdaptiveResidual` already corrects their draw estimate.

## Calibration (`python -m five_reactor calibrate`)

This uses scripted recipes on the five-reactor plant
(`five_reactor/calibrate.py`) and writes `results/log/calibration.json`.
- **The recipe:** the parent smoke test's staggered closed-loop recipe, with
  clocks from 1.0× down to 0.6×. It adds one rule: at most three vessels
  charge or co-feed at once. With all five free (they all start at hour 0),
  every setting ran 65–70 h over the cooling cap.
- **Headroom:** the recipe's post-change settings (clock pace, mass target,
  drain level) are retuned for each m and compared with the settings tuned
  at m = 1.30. The difference is the most a policy that infers m can win
  over one that acts on the mean.

On the nominal plant with no change:

| scripted policy | violation events | lots released | on-spec ethanol | Π |
|---|---|---|---|---|
| idle | 370 storage underflow + 1 purity | 0 | 0 | −2.96e9 |
| clocks in lockstep, drain whenever unloading | 391 storage overflow, 130 heat, 4 purity | 6 / 10 | 6.79e7 | −2.45e9 |
| staggered, all five feeding | 67 heat, 54 storage overflow, 1 purity | 5 / 6 | 5.86e7 | −1.56e8 |
| staggered, at most three feeding | 44 storage overflow, 3 purity | 3 / 6 | 3.80e7 | +1.46e7 |

Headroom is averaged over four prior tasks, 84 settings each. The pre-change
recipe tuned on the unchanged plant is pace 0.75, target 0.80, drain below
70% (Π 1.49e7, 11.5 events):

| m | retuned settings | Π retuned | Π tuned at 1.30 | Π not retuned | gap |
|---|---|---|---|---|---|
| 0.60 | (0.6, 0.95, 0.70) | 1.32e7 | 1.32e7 | 1.27e7 | +1.7e4 |
| 0.80 | (0.6, 0.80, 0.70) | 1.37e7 | 1.37e7 | 1.32e7 | +2.0e4 |
| 1.00 | (0.6, 0.95, 0.70) | 1.57e7 | 1.55e7 | 1.49e7 | +1.2e5 |
| 1.15 | (0.6, 0.80, 0.70) | 1.83e7 | 1.81e7 | 1.76e7 | +2.2e5 |
| 1.30 | (0.6, 0.90, 0.70) | 1.99e7 | 1.99e7 | 1.85e7 | 0 |
| 1.45 | (0.9, 0.90, 0.85) | 2.25e7 | 1.98e7 | 1.89e7 | +2.7e6 |
| 1.60 | (1.15, 0.90, 0.85) | 2.44e7 | 0.19e7 | 1.88e7 | +2.3e7 |
| 1.80 | (1.15, 0.90, 0.85) | 2.59e7 | −15.3e7 | −6.3e7 | +1.8e8 |
| 2.00 | (1.3, 0.80, 0.85) | 2.63e7 | −34.6e7 | −27.8e7 | +3.7e8 |

- **The plant can serve every m:** the retuned Π rises with the draw, since
  idle vessels can come online.
- **Headroom is one-sided.** Below the mean, holding batches is cheap and
  knowing m is worth ≤ 1.2%. Above about 1.45, the settings tuned on the
  mean (slow pace, low drain level) let the tank run dry. Retuning is worth
  12% at 1.45 and the whole production at 1.6 and beyond.
- **What to expect from INFER:** near zero in support and below it, and
  large on the wide draws' upper tail (P(m > 1.6) ≈ 11%). That tail lies
  outside the training support (1.14–1.46).

## Training (`python -m five_reactor train`)

This is the parent study's `meta` loop (`five_reactor/train.py`) without the
warm start's machinery: no actor warm-up, anchor, L2-SP decay, adaptation
gate or no-regression selection.
- **Episodes:** every training episode draws a fresh prior task and a
  training m.
- **Held-out checks:** each held-out task branches at the onset into m =
  1.15, 1.30 and 1.45, all inside the training distribution, so selection
  never sees the evaluation's tails. The unchanged branch (m = 1) is logged,
  never selected on.
- **Selection gate:** `SELECT_MAX_VIOL = 5` and `SELECT_MAX_WORST_VIOL = 20`,
  the parent's 3 / 12 scaled by 5/3. These are first guesses.
- **Logged with the score:** the aux head's decoded draw error (`m_mae`).

## Evaluation (`python -m five_reactor evaluate`)

There are 100 prior tasks (`PLANT_EVAL_EPISODES`). Each runs once to the
onset, and every branch continues from that snapshot:

| branch | m |
|---|---|
| `unchanged` | 1.0 |
| `train` | the task's training draw |
| `wide` | the task's evaluation draw |
| `sweep@x` | x = 0.6, 0.7, …, 2.0 |

- **Frozen branches.** Every changed branch also runs *frozen*: the same
  weights with z held at the onset's posterior mean, so the policy never
  infers the change. Its recurrent history still sees the plant.
- **INFER.** `INFER = adaptive − frozen` is paired per task. It measures
  what online inference of m is worth, without a second trained arm.
- **Reported:** Π and B (`../docs/evaluation_metrics.md`), events by
  constraint, on-spec ethanol, flow-change wear, the post-change feed and
  discharge, and the decoded draw 25, 50 and 100 h after the onset and at
  the end. The wide rows are also split in and outside the training support.

The figures are written to `results/plot/`:
- `training_curves.png`
- `change_response.png`: Π, B, post-change feed and discharge against m,
  adaptive vs frozen, with the support shaded and the wide draws' histogram.
- `change_inference.png`: decoded against true m, and the decoding error
  against time since the onset.

## How to run

From the repository root:

```bash
python five_reactor/smoke_test.py
```

```bash
python -m five_reactor calibrate
```

```bash
python -m five_reactor all
```

Outputs go to `five_reactor/policy/` and `five_reactor/results/`. The parent
study's environment variables apply (`PLANT_SEED`, `PLANT_RUN_TAG`,
`PLANT_META_ITERS`, `PLANT_EVAL_EPISODES`, `PLANT_HELDOUT_WORKERS`, ...).

| file | role |
|---|---|
| `__init__.py` | selects `PLANT_LAYOUT=five` and this folder as `PLANT_OUTPUT_ROOT` |
| `config.py` | the change's distributions, held-out magnitudes, selection gate, evaluation sweep |
| `change.py` | the magnitude samplers and the change's fault schedule |
| `arms.py` | `meta5`, registered in `metarl`'s `ARMS` (here and in workers) |
| `calibrate.py` | scripted-recipe coupling and headroom checks |
| `train.py` / `evaluate.py` / `plots.py` | training, evaluation, figures |
| `smoke_test.py` | fast CPU check of all of it |

Cost: a five-reactor episode takes ~5/3 of a three-reactor one, so expect
~8 h of training at the parent study's iteration budget. The critics' trunk
reads five tokens: ~11.4M parameters for the pair, against ~9.5M.
