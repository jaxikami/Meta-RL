# Meta-RL for Uncertainty — Three-Reactor Plant with Central Storage

The lean study's question (`../MetaRL_Lean`), asked of the full plant:

> Can a meta-RL agent that infers the operating regime online hold **production
> and product quality** while staying inside safety constraints, under flow,
> biochemical and storage faults a conventional safe-RL policy was never
> trained for?

Everything the lean study settled is kept: the constraints and quality
criteria, the instrument model (POMDP), the flow-only task distribution, the
fault kinds, the three arms, the warm-started and gated meta agent, and the
selection and evaluation protocol. What changes is the plant: **three
heterogeneous reactors and one central storage tank**, the multi-reactor
study's setup, with the storage tank's two inventory constraints added to the
lean study's six. The agent's networks are rebuilt for it (reactor tokens with
cross-reactor attention) and enlarged.

The chemistry is the Fermentation-PPO project's, pinned on every task and in
every reactor. The fast integrator now reads one kinetic-parameter row per
reactor, so a biochemical fault can act on one vessel alone; with identical
rows it matches the reference integrator to 1.8e-16 relative error
(`tests/smoke_test.py`).

The previous multi-reactor implementation (product-of-Gaussians PEARL,
inventory-only constraints) is in git history (commit `88452a4` and before).
Its leftover modules (`env_safe.py`, `agent_safe.py`, `main.py`,
`train_pearl_single_run.py`, `train_pearl_seed_sweep.py`,
`smoke_test_pearl.py`) are kept in `legacy/` for reference; they are
superseded and do not import against the `metarl` package.

---

## The three arms

| arm | training tasks | latent z | Lagrangian duals | stage mask | action projection (APN) |
|---|---|---|---|---|---|
| `baseline` | nominal only | no | no | no | no |
| `saferl` | nominal only | no | yes | yes | **hard**: reactor overflow / underflow, storage overflow |
| `meta` | fresh prior task per episode + mid-episode faults, from `saferl`'s policy, which acts until z detects a fault | yes | yes | yes | **hard**: the same shield |

- `SAFETY = saferl − baseline`, `ADAPT = meta − saferl`, both paired.
- `INCLUDE_BASELINE = False` by default (as in the lean study), so `all` trains
  `saferl` then `meta` and reports `ADAPT`.
- `meta` is the lean study's arm unchanged in kind: warm start from `saferl`
  (FiLM weights at zero, so it acts as `saferl` for every z), a frozen copy of
  `saferl` as the adaptation gate's base, `a = base(s) + g(z)(actor(s, z) −
  base(s))`, the per-fault-kind anchor, L2-SP decay toward `saferl`, pink
  exploration noise, no-regression selection against `saferl` as it runs. See
  the lean README for the reasons behind each.

---

## The plant

| unit | spec |
|---|---|
| reactors R0 / R1 / R2 | 220 t each; feed pumps 1× / **4×** / 0.5× the lean vessel's (LF 2487 kg/h, C5 628 kg/h at 1×) |
| feedstock pool | 2,000 t LF and 600 t C5, shared, drawn in reactor order each hour |
| central storage | 400 t (`S_MAX`), floor 5 t (`S_MIN`), starts at 350 t, feeds downstream at 1.5 t/h |

Each reactor runs the lean recipe (charge, co-feed, ferment, unload, clean;
262 h at a unit clock); an episode is 600 h, acted on every 5 h.

**The storage tank is the bottleneck, and it couples the reactors.**
Downstream takes 900 t over an episode, the tank holds 400 t and starts at
350 t, so the plant can discharge at most ~950 t (~4.5 full batches) without
overflowing it, against ~7 batches three free-running vessels would make. It
also runs dry by ~230 h without inflow. So unloads have to be staggered, a
finished batch sometimes held, and a batch started early enough to keep
downstream fed. A scripted scheduler shows both sides
(`tests/smoke_test.py`, nominal plant, exact instruments):

| scripted policy | violations | lots released | on-spec ethanol | reward |
|---|---|---|---|---|
| idle | 370 h storage underflow + purity (nothing shipped) | 0 | 0 | −3816 |
| clocks in lockstep, drain whenever unloading | 160 h storage overflow | — | — | — |
| clocks staggered 1.0 / 0.8 / 0.6× in charge and co-feed, drain only below 85% storage | **none** | 6 / 6 | 6.8e7 | +55.8 |

- **Action** `(12,)` in `[-1, 1]`, four per reactor: time multiplier
  (0.5–2×), LF feed, C5 feed, discharge.
- **Stage mask** per reactor (the lean mask): clock free except cleaning; LF in
  charge and co-feed; C5 in co-feed; discharge in unload.
- **Observation** `(78,)`:

  | channels | content |
  |---|---|
  | `24·i + 0..14` | reactor i species (zeroed when `MASK_INTERNAL_CONCENTRATIONS`) |
  | `24·i + 15` | reactor i mass / Mmax |
  | `24·i + 16..20` | reactor i stage one-hot |
  | `24·i + 21` | reactor i stage credit remaining |
  | `24·i + 22..23` | reactor i lot in progress: mass, mass-weighted mean titer |
  | `72` | storage mass / S_MAX |
  | `73..74` | LF / C5 pool remaining |
  | `75..77` | each reactor's off-gas CO₂ evolution rate |

  The CO₂ rates sit at the tail so the shield reads a prefix (`SHIELD_OBS_DIM
  = 75`) that never contains them, as in the lean study.
- **Lots** are per reactor batch, released or rejected when that reactor's
  batch completes (and at episode end). A rejected lot earns nothing; the
  others' released lots still count. The tank is a physical buffer only: lots
  are judged before they reach it.
- **Discharge cutoff** (2% of max outflow), the **unload-until-empty rule** and
  the **batch-boundary interlock** are the lean rules, per reactor. A batch
  boundary in any reactor ends the outer step, so the policy re-decides for
  every reactor; the longest episode is then 124 control steps
  (`CONTEXT_SIZE = 160`).

### Constraints

| id | constraint | scope | magnitude |
|---|---|---|---|
| 1 | reactor overflow | per reactor, per step | `M/Mmax − 1` |
| 2 | reactor underflow | per reactor, per step | `(DRY_FLOOR − M)/DRY_FLOOR` |
| 3 | harvest titer floor, 65 g/L | per draining reactor, per step | `(spec − titer)/spec` |
| 4 | product purity spec, 75 g/L | per lot (and once if the plant shipped nothing) | `(spec − lot mean)/spec` |
| 5 | numerical instability | per reactor | integrator severity |
| 6 | mass-balance drift | plant, episode end | residual / tolerance |
| 7 | **storage overflow** | plant, per step | `(S − S_MAX)/S_MAX` |
| 8 | **storage underflow** | plant, per step | `(S_MIN − S)/S_MIN` |

The headline count is violation **events**: every (hour, constraint, instance)
triple, the instance being the reactor for 1–5 and the plant for 6–8. Two
vessels overflowing in the same hour are two events; `violation_counts` sums
to the headline exactly.

### Reward

The lean terms, per reactor where they are local (`metarl/env/plant.py`):

| term | value |
|---|---|
| production | `+10 × harvested ethanol / 1e7`, withdrawn in full if the lot is rejected |
| raw material | `−5.0 × fraction of each pool consumed` (the lean 1.5 rescaled to the larger pools, so a kilogram of feed costs what it cost there) |
| violation spike | `−0.3 × (1 + magnitude)` per event (ids 1–3, 7, 8) |
| Lagrangian | `−λ × violation`; duals per reactor for 1–4, and one each for 7 and 8 (off for `baseline`) |
| soft walls | quadratic up to `−0.02`: each reactor near overflow (90%), underflow (< 100 kg) and the harvest spec; storage above 90% of S_MAX and below 60 t (≥ 37 h of downstream draw above S_MIN) |
| purity shaping, lot release, instability, mass balance, smoothing, trickle, feedstock barrier | the lean terms, per reactor where local |

### Instrument model

The lean model on every reactor: online gauges (masses, lot masses, storage,
feed pools) at 0.5% noise; hourly at-line assays (species, lot titers) at 3%
RSD with no delay; each reactor's off-gas CO₂ rate at 3%; stage and credit
exact; pump gains, leaks and delivered feed strength unmeasured.
`MASK_INTERNAL_CONCENTRATIONS = True` zeroes every reactor's species
channels. `baseline` and `saferl` train on exact observation; `meta` trains
with the noise; every arm is evaluated with it.

---

## Uncertainty

### Task distribution (per episode)

| latent | prior | scope |
|---|---|---|
| `feed_solids`, `feed_gx` | U(0.8, 1.2) | plant (shared feed tanks) |
| `feed_gain_r{0,1,2}` | U(0.85, 1.15) | **per reactor**, independent |
| `leak_frac_r{0,1,2}` | U(0, 0.0015) per h | **per reactor**, independent |

So the encoder must also infer *which* vessel is off. OOD tasks widen every
half-width 1.8×. The task descriptor (aux target, privileged critic input) has
15 entries: feed, three gains, three leaks, three kinetic activities, three
contamination rates and the downstream rate.

### Mid-episode faults

| kind | target | effect | meta-training range | eval |
|---|---|---|---|---|
| `leak` | one reactor or all | extra mass loss per hour | 0.004–0.016 | 0.010 |
| `feed_gain` | one pump or all | gain × magnitude | 0.60–1.45 | 1.30 |
| `feed_conc` | the shared feed tanks | composition × magnitude | 0.30–1.40 | recurrent dilution |
| `disturb_kinetics_decay` | one reactor or all | hydrolysis and uptake × magnitude | 0.15–0.70 | 0.20 |
| `toxin_spike` | one reactor or all | its feed-line inhibitors, a broth pulse, uptake × 1/magnitude | 2.0–8.0 | 6.50 |
| `cell_viability` | one reactor or all | viable cells × magnitude | 0.30–0.70 | 0.50 |
| `contamination` | one reactor or all | its broth ethanol oxidised per hour | 0.0005–0.004 | — |
| **`downstream_rate`** | the storage tank | downstream draw × magnitude | 0.40–1.60 | 0.50 |

- Meta-training picks one reactor uniformly, or all three with probability
  `META_PLANTWIDE_FRAC = 0.2`. Patterns are the lean ones (single, drift,
  compound, lot change); the compound pairs add `downstream_rate + feed_gain`.
- In evaluation and in the held-out panel, a per-reactor fault **rotates**:
  episode k faults reactor `k mod 3`, identically for every arm, so the
  comparison stays paired and every vessel is faulted on a third of the tasks.
- `downstream_rate` is new: a distillation slowdown fills the tank (overflow,
  or batches held back); a surge empties it. The storage mass is online, so it
  is inferable, and it sits in `GATE_FAULT_KINDS` with the leak and pump
  faults.
- The magnitudes are the lean study's, which were calibrated against the lean
  `saferl`. Recalibrate them against the plant's `saferl` before trusting a
  row. Against the scripted scheduler every kind hurts on at least one axis
  except `cell_viability`, which costs nothing at any magnitude (the lean
  study's finding: the yeast regrows within the batch), and stays a row
  where a tie is expected.

---

## The action projection network (APN)

`metarl/apn/` (`network.py`, `data.py`, `train.py`): the lean shield, widened. With
concentrations masked it covers **each reactor's overflow and underflow and
the storage overflow** (seven margins, smooth minimum) under a one-interval
lookahead that reproduces the discharge cap, the shared feed pool drawn in
reactor order and the tank's inflow and draw. The projection never moves a
reactor's stage clock or a locked dimension; mask > projection > actor.

**Storage underflow is not shielded.** Within one interval the only way to
raise the tank is to drain a reactor already in unload. When none is, every
action is equally unsafe, and the shield would "repair" it by shutting every
feed, costing production for nothing. Keeping the tank supplied is decided
tens of hours earlier, so it is left to the policy and the reward, like
harvest titer and purity.

The network is twice the lean width (128) for its 87 inputs. Validation on 50k
fresh samples: sign accuracy 0.973, boundary accuracy 0.895, repair rate
0.942, no-op rate 0.996, safe preserved 1.000, mask drift 0 (all gates pass).
`metarl/apn/train.py` ranks validation checks on (passed every gate, score): the
lean trainer ranked on score alone and, after an early stop on a passing
check, restored an earlier check that failed the repair gate.

---

## Networks

Every arm uses the same architecture (`NET_*` in `metarl/training/config.py`).

- **Reactor tokens.** A reactor's token is its 24 process channels, its CO₂
  rate and the 3 plant-wide channels, embedded with a learned slot embedding
  (the three vessels' pumps differ). Two pre-LN self-attention blocks
  (`TokenMixer`, 4 heads) mix the tokens; this is where the reactors
  coordinate over the shared tank and feed pool. Each block's output
  projections start at zero, so the mixer starts as the identity.
- **Actor.** One FiLM-conditioned residual trunk shared by the three tokens
  emits each reactor's four actions from its own token, so every reactor's
  experience trains the same weights; z modulates every trunk layer.
- **Critics.** Each Q network mixes (observation, that reactor's action)
  tokens and flattens them into a wide FiLM residual trunk: the plant's value
  is not a sum over reactors, because the tank couples them. `meta`'s critics
  are conditioned on z, the true task descriptor and the plant's privileged
  state (79 channels), as in the lean study.
- **Encoder.** The lean GRU posterior, at width 256, over transitions read as
  (obs, action, reward, obs change). Latent dimension 8 (lean: 5).

| network | lean | plant |
|---|---|---|
| actor | 3 × 256 MLP, ~151k | tokens 256, 2 attention blocks, shared 3 × 256 trunk, ~1.27M |
| critics (pair) | 2 × 3 × 512, ~1.13M | tokens 256, 2 attention blocks, 3 × 1024 trunk each, ~7.9M (`saferl`), ~9.2M (`meta`) |
| encoder | GRU 128, ~108k | GRU 256, ~0.44M |

The observation normaliser, AdamW (FiLM exempt), LayerNorm and every
posterior-collapse countermeasure are the lean ones. Batch-1 CPU acting costs
~3 ms per control step.

---

## Training, selection and evaluation

The lean protocol (`metarl/training/`, every setting in its `config.py`),
with these plant-specific settings:

- `SELECT_MAX_VIOL = 3`, `SELECT_MAX_WORST_VIOL = 12` (lean 2 / 8, scaled for
  three reactors and a tank) — **first guesses: calibrate on a trained
  `saferl`**. If `saferl`'s best check is never feasible, training says so.
- `GATE_FAULT_KINDS = (leak, feed_gain, downstream_rate)`: the faults that
  move an online measurement.
- The held-out panel adds `downstream_rate@0.6`; per-reactor branches rotate.
- `EVAL_EPISODES = 100` (lean 200; a plant episode costs ~3×).
- Evaluation scenarios: in-distribution, OOD, the six `EVAL_DISTURBANCES`
  (five rotating per-reactor faults and `downstream_rate@0.5`), and the
  recurrent dilution.

Figures (`results/plot/[tag]/`) are the lean set (`compare_safety`,
`compare_episodes_violated`, `compare_production`, `compare_quality`,
`compare_adapt`, `violation_mix_*`, `violation_severity`, `distributions`,
`disturbance_response`, `training_curves`), plus **`compare_reactors.png`**
(on-spec ethanol and violation events per reactor). The disturbance-response
panels show the faulted reactor's actuator and add the storage trajectory.
`eval.json` records per-reactor on-spec ethanol, batches, actuator means and
violation events, the storage range and each episode's faulted reactor.

The lean `optional/headroom.py` and `optional/calibrate_faults.py` are not
ported yet; `compare_headroom.png` is not drawn.

---

## Files

Everything lives in the `metarl` package; the two large classes are split
into mixins, one module per concern.

| module | role |
|---|---|
| `metarl/env/core.py` | Numba kinetics and RK4 integrator, per-reactor parameter rows |
| `metarl/env/config.py` | reactors, feed pool, storage tank, quality specs, observation layout, instrument model, task distribution |
| `metarl/env/layout.py` | observation and action indices, the stage mask |
| `metarl/env/plant.py` | `PlantFermentationEnv`: 3 reactors + storage, the step, reward, constraints 1–8 |
| `metarl/env/faults.py` / `instruments.py` / `ledger.py` | its mixins: the eight fault kinds, the instrument model, lots, duals and terminal accounting |
| `metarl/agent/pearl.py` | `PEARLAgent`: the PEARL/SAC agent and its rollout copies |
| `metarl/agent/acting.py` / `learning.py` / `checkpoint.py` | its mixins: posterior and action selection (adaptation gate, APN hooks), the update, save / load / warm start |
| `metarl/agent/networks.py` / `replay.py` / `common.py` | token actor and critics, GRU encoder, gate; per-task replay; devices and utilities |
| `metarl/apn/network.py` / `data.py` / `train.py` | the shield: network and projection, margin label and sampler, training and validation |
| `metarl/training/config.py` | every setting of the study (`NET_*`, `SELECT_*`, fault ranges, scenarios, output paths) |
| `metarl/training/cli.py` | the modes behind `python -m metarl` |
| `metarl/training/arms.py` / `train.py` / `heldout.py` / `early_stopping.py` | arms and their agents, training, held-out checks and checkpoint selection, early stopping |
| `metarl/training/tasks.py` / `rollouts.py` | task pools and fault schedules; training and deterministic episodes |
| `metarl/training/evaluate.py` / `z_informativeness.py` | evaluation and the latent gate |
| `metarl/training/metrics.py` / `plots.py` / `logs.py` | statistics, figures, JSON logs |
| `tests/smoke_test.py` | fast CPU check of every code path and the design properties |
| `legacy/` | the previous implementation's modules, superseded |

## How to run

Run from this folder (the repository root). First the smoke test (~15 s; it
prints `ALL SMOKE CHECKS PASSED`):

```bash
python tests/smoke_test.py
```

Train the shared shield (also trained automatically when an arm needs it and
the checkpoint is missing or stale):

```bash
python -m metarl.apn.train
```

Train `saferl`, then `meta`, then evaluate and run the latent gate:

```bash
python -m metarl all
```

Or one stage at a time: `train_saferl`, `train_meta`, `evaluate`,
`z_informativeness`, `plot_eval`, `plot_curves`. Seeds and tags, in
PowerShell:

```powershell
$env:PLANT_SEED=1; $env:PLANT_RUN_TAG="seed1"; python -m metarl all
```

| environment variable | effect |
|---|---|
| `PLANT_SEED` | training and evaluation seed (default 0) |
| `PLANT_RUN_TAG` | subfolder for checkpoints, logs and figures |
| `PLANT_APN_PATH` | APN checkpoint path (default `policy/apn_plant.pth`) |
| `PLANT_HELDOUT_WORKERS` | held-out worker processes; `0` or `1` runs in-process |
| `PLANT_DEVICE` | force the training device, e.g. `cpu` |
| `PLANT_VERBOSE_PBAR` | `1` adds losses and latent diagnostics to the progress bar |

Outputs: `policy/[tag]/<arm>.pth` and `<arm>_best.pth`; `results/log/[tag]/`
(`<arm>_log.json`, `eval.json`, `eval_summary.json`, `z_informativeness.json`);
`results/plot/[tag]/`.

## Requirements

`numpy`, `numba`, `torch`, `matplotlib`, `tqdm` (optional). Gradient updates
use a CUDA/ROCm GPU when available (the critics are ~8–9M parameters);
rollouts always run on the CPU, held-out checks in worker processes.
