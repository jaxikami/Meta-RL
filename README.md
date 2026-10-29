# Meta-RL for Uncertainty — Three-Reactor Plant with Central Storage

The lean study's question (`../MetaRL_Lean`), asked of the full plant:

> Can a meta-RL agent that infers the operating regime online hold **production
> and product quality** while staying inside safety constraints, under flow,
> biochemical, storage and cooling faults a conventional safe-RL policy was
> never trained for?

**The environment is Fermentation-PPO's** (`../Fermentation-PPO/envs/`), so
this study's arms and Safe PPO are benchmarked on one plant. Its four modules
are ported into `metarl/env/` with only their import paths changed
(`kinetics`, `disturbances`, `plant`, `safe`, and its constants at
the top of `config.py`). `tests/smoke_test.py` checks that they still equal
Fermentation-PPO's sources, and steps both environments side by side under
every evaluation fault with bit-identical observations, rewards and violation
counts. It skips the check when `../Fermentation-PPO` is absent;
`FERMENTATION_PPO_ROOT` points it elsewhere. Both arms run on Safe PPO's
`FermentationEnv`. The benchmark is safe RL against meta-RL: there is no
unconstrained baseline arm, and Standard PPO's `FermentationEnvBench` is not
ported.

What stays this study's own: the flow-only task distribution, the two arms,
the warm-started and gated meta agent, the action projection network, and
the selection and evaluation protocol. `metarl/env/adapter.py` puts the
environment behind the agent's interface without changing its plant, reward
or constraints. It flattens the per-reactor observation and sums the
per-reactor rewards, installs a task's feeds, pump gains and leaks as the
plant's nominal values, and exposes the privileged views training reads (the
effective task descriptor, the fault flag, the privileged state) plus a
per-reactor ledger for the evaluation. The nominal task is the
Fermentation-PPO plant exactly.

Before the port, this study ran its own plant: the lean instrument model
(masked concentrations, noisy assays, an off-gas CO₂ channel, a POMDP), a
turnaround-conflict constraint as id 9, early-ending steps at batch
boundaries, and its own reward scale. That plant and the results trained on
it are in git history (commit `bc502ee` and before), as are the figures
and logs trained on it.

The previous multi-reactor implementation (product-of-Gaussians PEARL,
inventory-only constraints) is in git history (commit `88452a4` and before),
with its leftover modules.

---

## The two arms

| arm | environment | training tasks | latent z | Lagrangian duals | stage mask | action projection (APN) |
|---|---|---|---|---|---|---|
| `saferl` | Safe PPO's (`FermentationEnv`) | nominal only | no | yes | yes | **hard**: reactor overflow / underflow, storage overflow |
| `meta` | Safe PPO's | fresh prior task per episode + mid-episode faults, from `saferl`'s policy, which acts until z detects a fault | yes | yes | yes | **hard**: the same shield |

- `ADAPT = meta − saferl`, paired. `all` trains `saferl` then `meta` and
  reports `ADAPT`.
- `meta` is the lean study's arm unchanged in kind: warm start from `saferl`
  (FiLM weights at zero, so it acts as `saferl` for every z), a frozen copy of
  `saferl` as the adaptation gate's base, `a = base(s) + g(z)(actor(s, z) −
  base(s))`, the per-fault-kind anchor, L2-SP decay toward `saferl`, pink
  exploration noise, no-regression selection against `saferl` as it runs. See
  the lean README for the reasons behind each.
- `lagrangian=False` on a Safe PPO environment (the ablation's `lagrangian`
  component) is Fermentation-PPO's no-Lagrangian case study: every
  multiplier at zero and frozen, fixed per-violation penalties instead.
- The duals are reset every episode (`RESET_LAMBDAS_PER_EPISODE = True`, a
  setting of this study; the environment's default, and Safe PPO's training,
  carry them across episodes). An off-policy replay stores rewards priced by
  the duals of the episode they came from, so carrying them over would make
  the stored rewards stale.

---

## The plant

| unit | spec |
|---|---|
| reactors R0 / R1 / R2 | 220 t each; feed pumps 1× / **4×** / 0.5× (LF 2487 kg/h, C5 628 kg/h at 1×) |
| feedstock pool | 2,000 t LF and 600 t C5, shared, drawn in reactor order each hour |
| central storage | 400 t (`S_MAX`), floor 5 t (`S_MIN`), starts at 350 t, feeds downstream at 1.5 t/h |
| cooling loop | one, shared: the plant's CO₂ evolution rate (the heat-load proxy) ≤ 540 kg/h |

Each reactor runs the recipe charge, co-feed, ferment, unload, clean (262 h
at a unit clock); an episode is 600 h, acted on every 5 h: exactly 120
control steps (`CONTEXT_SIZE = 128`).

**The storage tank is the bottleneck, and it couples the reactors**; so does
the cooling loop. Downstream takes 900 t over an episode, the tank holds
400 t and starts at 350 t, so the plant can discharge at most ~950 t (~4.5
full batches) without overflowing it, against ~7 batches three free-running
vessels would make. It also runs dry by ~230 h without inflow. So unloads
have to be staggered, a finished batch sometimes held, and a batch started
early enough to keep downstream fed. The heat cap sits just below what the
three vessels release feeding at half rate in phase, so fast feeding in one
leaves less for the others. Scripted policies (`tests/smoke_test.py`, nominal
plant):

| scripted policy | violation events | lots released | on-spec ethanol | reward |
|---|---|---|---|---|
| idle | 370 storage underflow + 1 purity (nothing shipped) | 0 | 0 | −368.9 |
| clocks in lockstep, drain whenever unloading | 161 storage overflow + 2 purity | — | — | −26.7 |
| clocks staggered 1.0 / 0.8 / 0.6×, closed-loop feed toward 80% Mmax, drain below 85% storage | 1 purity (the lot open at episode end) | 5 / 6 | 6.61e7 | +1.86 |
| the same recipe at fixed feed rates | **none** (heat peak 432 kg/h) | 5 / 5 | 6.68e7 | +1.82 |

- **Action** `(12,)` in `[-1, 1]`, four per reactor: time multiplier
  (0.5–2×), LF feed, C5 feed, discharge.
- **Stage mask** per reactor (the environment's `get_action_mask`): clock
  free except cleaning; LF in charge and co-feed; C5 in co-feed; discharge in
  unload.
- **Observation** `(84,)`: the environment's `get_state_norm()`, one row of
  28 per reactor, flattened. It is exact: no instrument noise, every species
  observed.

  | channels of row i (at `28·i`) | content |
  |---|---|
  | `0..14` | reactor i species / `C_MAX_LIMITS` |
  | `15` | reactor i mass / Mmax |
  | `16..20` | reactor i stage one-hot |
  | `21` | reactor i stage credit remaining |
  | `22..24` | reactor i capacities: Mmax, LF pump, C5 pump (normalised) |
  | `25..27` | plant: storage mass / S_MAX, LF and C5 pools remaining (the same in every row) |

  Nothing in it shows a lot in progress (its mass and mean titer), a pump's
  gain, a leak, the storage or cooling capacity, or the heat load itself.
- **Lots** are per reactor batch, released or rejected when that reactor's
  cleanup ends (and at episode end). A rejected lot's production reward is
  withdrawn; the others' released lots still count.
- **Discharge cutoff** (2% of max outflow), the **unload-until-empty rule**
  and the **batch-boundary interlock** (the valve opens only in unload) are
  per reactor. An outer step always runs its five hours.

### Constraints

The environment's ids (`metarl/env/safe.py`):

| id | constraint | scope | depth past the limit |
|---|---|---|---|
| 1 | reactor overflow | per reactor, per step | `M/Mmax − 1` |
| 2 | reactor underflow | per reactor, per step | `(DRY_FLOOR − M)/DRY_FLOOR` |
| 3 | storage overflow | plant, per step | `(S − cap)/cap`, cap = S_MAX × storage capacity |
| 4 | storage underflow | plant, per step | `(S_MIN − S)/S_MIN` |
| 5 | numerical instability | plant (ends the episode) | — |
| 6 | mass-balance drift | plant, episode end | — |
| 7 | harvest titer floor, 65 g/L | per draining reactor, per step | `(spec − titer)/spec` |
| 8 | product purity spec, 75 g/L | per lot (and once if the plant shipped nothing) | `(spec − lot mean)/spec` |
| 9 | **heat-removal capacity** | plant, per step | `(heat load − cap)/cap`, cap = 540 kg CO₂/h × cooling capacity |

The headline count is violation **events** (`info["violation_events"]`):
every (hour, constraint, instance) triple, the instance being the reactor
for 1, 2 and 7, the lot for 8 and the plant otherwise; `violation_counts`
sums to it exactly. The adapter attributes ids 1, 2, 7 and 8 to their
reactors (`reactor_violation_events`).

Constraint 9 couples scheduling and control: how fast each reactor is fed,
and when the batches' fed-batch phases overlap, decide together whether the
plant stays under the cap, and it trades off against purity (a high-titer
batch must be fed slowly over a longer fed-batch).

### Reward

Safe PPO's (`metarl/env/safe.py`), per reactor; the agent gets their sum:

| term | value |
|---|---|
| production | `+0.5 × harvested ethanol / 1e7`, withdrawn in full if the lot is rejected |
| raw material | `−10 × fraction of each pool consumed` |
| violation price | `−λ × d / (1 + d)` at depth `d`; duals per reactor for 1, 2, 7, 8 and one each for 3, 4, 9, grown by `0.5 × d` per violated step, decayed by `0.5/600` otherwise, at most 2. The heat dual's cost is split over the reactors by their share of the load. No fixed spike unless `lagrangian=False` |
| soft walls | quadratic up to `−0.005`: each reactor above 90% Mmax, below 100 kg and draining within 15% of the harvest spec; storage above 90% of its volume and below 8 × S_MIN; heat load above 90% of the cap |
| barriers | storage log-barrier below 16 × S_MIN; feedstock below 2% |
| purity shaping | `−0.002` quadratic on each lot below 1.15 × spec |
| smoothing, trickle | `0.005 ×` mean squared change of the flow commands; `0.0005` per reactor in unload |

---

## Uncertainty

### Task distribution (per episode)

| latent | prior | scope |
|---|---|---|
| `feed_solids`, `feed_gx` | U(0.8, 1.2) | plant (shared feed tanks) |
| `feed_gain_r{0,1,2}` | U(0.85, 1.15) | **per reactor**, independent |
| `leak_frac_r{0,1,2}` | U(0, 0.0015) per h | **per reactor**, independent |

So the encoder must also infer *which* vessel is off. OOD tasks widen every
half-width 1.8×. A task's values are installed as the plant's nominal ones
before every reset: a `feed_conc` fault scales the task's feed, a
`feed_gain` fault multiplies the task's gain, a fault leak adds to the
task's. The task descriptor (aux target, privileged critic input) has 18
entries: feed, three gains, three leaks, three kinetic activities, three
contamination rates, the downstream rate, turnaround time, storage volume
and cooling capacity.

### Mid-episode faults

Fermentation-PPO's kinds and evaluation scenarios (`metarl/env/disturbances.py`):

| kind | target | effect | meta-training range | eval |
|---|---|---|---|---|
| `leak` | one reactor or all | extra mass loss per hour | 0.004–0.016 | 0.010 |
| `feed_gain` | one pump or all | gain × magnitude | 0.60–1.45 | 1.30 |
| `feed_conc` | the shared feed tanks | composition × magnitude | 0.30–1.40 | 0.55 (`dilution`) |
| `disturb_kinetics_decay` | one reactor or all | hydrolysis and uptake × magnitude | 0.15–0.70 | 0.20 |
| `toxin_spike` | one reactor or all | its feed-line inhibitors, a broth pulse, uptake × 1/magnitude | 2.0–8.0 | 6.50 |
| `cell_viability` | one reactor or all | viable cells × magnitude | 0.30–0.70 | 0.50 |
| `contamination` | one reactor or all | its broth ethanol oxidised per hour | 0.0005–0.004 | 0.002 |
| `downstream_rate` | the storage tank | downstream draw × magnitude | 0.40–1.60 | 0.50 |
| `turnaround_delay` | the shared turnaround equipment | every cleanup and re-charge that starts after onset × magnitude long | 1.5–5.0 | 4.0 |
| `storage_capacity` | the storage tank | usable volume × magnitude (the overflow wall moves, the level reading does not) | 0.75–0.97 | 0.90 |
| **`cooling_capacity`** | the cooling loop | heat-load cap × magnitude (nothing observed moves) | 0.65–0.95 | 0.80 |

- Every kind has four temporal **patterns** (`FAULT_PATTERNS`, built by
  `tasks.pattern_schedule`; the evaluation schedules equal
  Fermentation-PPO's `eval_schedule`, checked by the smoke test):

  | pattern | profile | evaluation timing (h of 600) |
  |---|---|---|
  | `step` | full magnitude from onset to the end | onset 300 |
  | `spike` | full magnitude, then cleared | 300–324 |
  | `gradual` | ramps up in 6 treads to full magnitude | 300 → 480 |
  | `recurrent` | on, off, on, off, on | 150–230, 310–390, 470–550 |

  A once-firing kind is cleared by an entry of magnitude 1.0 (a kill is not
  undone: the cells regrow), an additive kind (`leak`, `contamination`) by
  its end step. A ramp adds `magnitude / 6` per tread for an additive kind,
  multiplies by `magnitude^(1/6)` for `cell_viability`, and steps linearly
  from 1.0 otherwise. A toxin's acute broth pulse is sized by the RISE in
  feed toxin, so a ramp's pulses add up to one step's and each recurrence
  pulses anew.
- Meta-training picks one reactor uniformly, or all three with probability
  `META_PLANTWIDE_FRAC = 0.2`, and a pattern from `META_FAULT_PATTERNS`
  (step 0.35, spike 0.15, gradual 0.15, recurrent 0.20, plus training-only
  compound 0.15: two kinds stepping together), with spike lengths, ramp
  length and on / off windows drawn per episode (`META_SPIKE_H`,
  `META_GRADUAL_FRAC`, `META_RECURRENT_*`).
- In evaluation and in the held-out panel, a per-reactor fault **rotates**:
  episode k faults reactor `k mod 3`, identically for every arm, so the
  comparison stays paired and every vessel is faulted on a third of the tasks.
- `cooling_capacity` and `storage_capacity` move a limit the observation
  does not show: they reveal themselves only through the reward. The
  `cooling_capacity` range is new and uncalibrated, and the other ranges were
  calibrated against the previous plant's `saferl`. Recalibrate them against
  the retrained `saferl` before trusting a row.
- Against the scripted policies every kind hurts on at least one axis except
  `cell_viability`, which costs nothing at any magnitude (the lean study's
  finding: the yeast regrows within the batch), and stays a row where a tie
  is expected.

---

## The action projection network (APN)

`metarl/apn/` (`network.py`, `data.py`, `train.py`): the lean shield, widened. It
covers **each reactor's overflow and underflow and the storage overflow**
(seven margins, smooth minimum) under a one-interval lookahead that
reproduces the discharge cap, the shared feed pool drawn in reactor order and
the tank's inflow and draw. The projection never moves a reactor's stage
clock or a locked dimension; mask > projection > actor. The species channels
are zeroed at its input (`SHIELD_IGNORED_CHANNELS`): no shielded margin
depends on them, and reading them as noise cost the repair rate.

**Storage underflow is not shielded.** Within one interval the only way to
raise the tank is to drain a reactor already in unload. When none is, every
action is equally unsafe, and the shield would "repair" it by shutting every
feed, costing production for nothing. Keeping the tank supplied is decided
tens of hours earlier, so it is left to the policy and the reward. So are
harvest titer and purity (the observation carries no lot) and the heat load
(it needs the kinetics; Safe PPO learns that head, `safety/kinetic.py`).

The network is twice the lean width (128) for its 96 inputs. Validation of
`policy/apn_ppo_plant.pth` on 50k fresh samples: sign accuracy 0.992,
boundary accuracy 0.971, no-op rate 1.000, safe preserved 1.000, mask drift 0;
repair rate 0.897-0.906 across validation seeds, against a gate of 0.90.
The repair test counts a projection as repaired only when the *smooth*
minimum margin turns positive; on the hard constraints it repairs 97.4%, and
the misses end within 0.3% of a limit. The shield was trained through the
whole cosine schedule (`PATIENCE = 9000`): stopped mid-cycle, as the old
patience did, it reached repair 0.87. `policy/apn_plant.pth` is the previous
plant's shield and is not loaded. `metarl/apn/train.py` ranks validation
checks on (passed every gate, score): the lean trainer ranked on score alone
and, after an early stop on a passing check, restored an earlier check that
failed the repair gate.

---

## Networks

Every arm uses the same architecture (`NET_*` in `metarl/training/config.py`).

- **Reactor history** (`ReactorHistory`, the Fermentation-PPO policy's
  encoder; `NET_HISTORY_*`). A reactor's token is its observation row, the
  28 channels Safe PPO's policy reads per reactor. It is embedded (Linear →
  LayerNorm → ELU, plus a learned slot embedding for the vessel) and folded
  into a 2-layer GRU of width 256 whose weights the reactors share and whose
  state each keeps on its own. The reactors then attend to each other's
  layer-normalised GRU outputs (`CrossReactorAttention`, 4 heads in a 64-wide
  Q/K/V space, residual + LayerNorm, output projection starting at zero), so
  a reactor sees its siblings' history summaries rather than their
  single-step readings. A 128-wide skip embedding of the current token is
  appended: each reactor's feature is 384 wide.
- **Actor.** One FiLM-conditioned residual trunk shared by the three
  reactors' features emits each reactor's four actions, so every reactor's
  experience trains the same weights; z modulates every trunk layer. The
  policy is recurrent: acting carries the history state from step to step
  and resets it at an episode start; the update rebuilds each sampled
  transition's state from its episode's previous transitions (the same
  causal windows the encoder reads), exactly as online (`tests/smoke_test.py`).
- **Critics.** Each Q network reads the actor's history features with that
  reactor's action, mixes the three tokens (two pre-LN attention blocks,
  `NET_MIXER_*`: the joint action couples the reactors through the tank and
  the cooling loop) and flattens them into a wide FiLM residual trunk. The
  critic's loss trains the history module (the actor reads it detached, as
  in SAC with a shared encoder); `meta` keeps `saferl`'s history module
  frozen, so its actor, the gate's base policy and the anchor read the same
  features and it adapts through z. Every arm's critics are conditioned on z
  (zeros without context), the true task descriptor (18) and the plant's
  privileged state (91 channels: the observation, each reactor's lot mass and
  mean titer, the clock): training-only inputs, the same for both arms, so
  the critics are the same networks and a warm start copies them whole (the
  latent's FiLM columns start at zero).
- **Encoder.** The lean GRU posterior, at the lean width 128, over
  transitions read as (obs, action, reward, obs change), 181 inputs. Latent
  dimension 8 (lean: 5); the aux and fault-kind heads are 128 wide. The aux
  head regresses the task descriptor and each reactor's lot mean titer (the
  quality state the observation lacks).

Against Safe PPO (`../Fermentation-PPO/agents/networks.py`), parameters:

| network | Safe PPO | `saferl` | `meta` |
|---|---|---|---|
| per-reactor history (GRU 2 × 256, attention 64 × 4, skip 128) | 0.869M | 0.869M | 0.869M (frozen) |
| policy head | 384 → 128 → 4: 0.050M | 3 × 256 FiLM residual trunk: 0.248M | 0.248M |
| value | 384 → 128 → 1 per reactor: 0.050M | twin token-mixer Q, 3 × 1024 trunks: 9.548M | 9.548M |
| learned shield | KineticNet (purity, heat heads): 0.144M | APN (inventory): 0.063M | the same APN |
| inference | — | — | encoder 0.125M, gate 0.018M, aux and fault-kind heads 0.040M |
| acting (policy + inference) | 0.919M | 1.117M | 1.259M |
| trained, in all | 1.112M | 10.728M | 10.910M |

The policy side matches Safe PPO's: the same history module on the same
28-channel rows, with a wider head because z modulates it. The token width
did not change with the port (the old token was also 28 wide, with the
species zeroed and an off-gas rate), so no `NET_*` size changes; only the
inputs that scale with the observation grew (the encoder's, 169 → 181, and
the critics' conditioning, 104 → 117). The critics are SAC's twin Q
functions over the joint action; they are ~10× Safe PPO's whole network,
which a PPO critic does not need (its value function sees no action).

A capacity probe on this plant (300 episodes of perturbed scripted
policies on random tasks with meta-training faults, 600 updates per width)
trained `ReactorHistory` plus a linear head to track two quantities the
policy needs: each lot's mean titer (hidden; needs memory) and the next
step's plant heat load (constraint 9: nonlinear in the species, summed over
the reactors). Held-out normalised MSE:

| history width | parameters | lot titer | heat load |
|---|---|---|---|
| 64 | 0.07M | 0.043 | 0.062 |
| 128 | 0.24M | 0.042 | 0.064 |
| **256** (used) | 0.87M | 0.043 | 0.070 |
| 384 | 1.89M | 0.044 | 0.068 |

Flat from 64 to 384: the new observation and the heat constraint do not
make width a bottleneck, so a wider history would buy nothing. A narrower
one would fit these targets too. The width stays at 256 so the policy's
encoder is Safe PPO's, size for size, and the benchmark compares methods,
not network sizes. Whether a smaller history trains as well under RL is
untested.

`meta`'s frozen copy of `saferl`'s actor (the gate's base) adds no capacity.
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
- The privileged fault flag (the gate's target, the anchor's switch) is on
  while a fault holds the plant off its task's values: a step latches it, a
  spike or a recurrent fault's off-window clears it, so the gate learns to
  close again when the plant recovers.
- `EARLY_STOP_PATIENCE = 50` checks (500 iterations;
  `PLANT_EARLY_STOP_PATIENCE` overrides it). At 30, `saferl` stopped ~12%
  short of its previous held-out output.
- Training episodes run in the held-out worker pool while the GPU updates
  (`COLLECT_IN_WORKERS`): each iteration's episodes use the weights from
  before its updates and enter the replay after them. The task and fault
  draws are the in-process ones, in the same order. Profiled, collection was
  ~25% of an iteration and the updates ~65%.
- The selection thresholds were set on the previous plant (its constraint
  set and reward); check the retrained `saferl`'s held-out violations on this
  one before trusting the selection.
- The held-out panel adds `downstream_rate@0.6` and `cooling_capacity@0.82`;
  per-reactor branches rotate.
- `EVAL_EPISODES = 100` (lean 200; a plant episode costs ~3×;
  `PLANT_EVAL_EPISODES` overrides it for a quick run).
- Evaluation runs one job per (arm, task) in `EVAL_WORKERS` processes
  (default: every logical CPU). A job simulates the task's undisturbed
  episode once and continues every fault scenario from a snapshot taken just
  before its first onset, so each row is exactly the from-scratch episode
  (the smoke test checks it) at about half the simulation.
- Evaluation scenarios: in-distribution, OOD, then every
  `EVAL_DISTURBANCES` kind (Fermentation-PPO's) under every pattern, named
  `<kind>_<pattern>` (`leak`, `feed_gain`, `dilution`, `kinetics_decay`,
  `toxin`, `cell_viability`, `contamination`, `downstream_rate`,
  `turnaround_delay`, `storage_capacity`, `cooling_capacity` × `step`,
  `spike`, `gradual`, `recurrent`): 46 scenarios. The held-out panel and the
  latent gate use step faults.

### Why `meta` tied `saferl`, and the training fixes

*On the previous plant (before the port to Fermentation-PPO's environment).
The fixes stay; the numbers need re-measuring on this plant.*

The first run on the 38-scenario grid had `meta` beat `saferl` on 6
scenarios, lose on 2 and tie the rest; its held-out panel hardly moved over
1170 iterations. Probes on its checkpoint traced this to the latent, not the
actor's capacity:

- **z did not register most faults.** 200 h after onset the aux head decoded
  kinetic activity 0.95 (true 0.2), storage capacity 0.997 (true 0.90) and
  feed concentration 0.92 (true 0.56), so the gate stayed shut and `saferl`
  acted. Forcing the gate open at onset changed nothing but leak.
- **The encoder memorised the replay.** At 3 episodes per iteration and an
  encoder step on every update, each stored transition fed ~140 encoder
  steps. On its own batches the gate opened on 85–90% of faulted steps; on
  fresh episodes of the same distribution, 1–50% by kind.
- **The executed action told the encoder whether the gate was open**
  (exploration noise passes only an open gate), so detection fed on the gate
  itself. A gate opened on the privileged fault flag leaks that flag the
  same way.

The fixes (`config.py`; `meta` only, `saferl` is untouched):

| setting | value | effect |
|---|---|---|
| `META_TRAIN_GATE_OPEN` | True | training rollouts and the update act through the actor on every step; held-out checks and evaluation through the learned gate |
| `META_FAULT_CLS_WEIGHT` | 1.0 | class-balanced fault-kind head on z (privileged target, training only) |
| `META_DISTURB_PROB` | 0.45 → 0.7 | ~2% → ~4% of the replay per fault kind |
| `META_COLLECT_EPISODES_PER_ITER` | 9 | collected in parallel in the worker pool |
| `META_ENCODER_UPDATE_EVERY` | 3 | with the above, ~15 encoder steps per stored transition |

Retrained against the same `saferl` (run tags `v2`, selected check, and
`v2final`, final checkpoint at early stop), 100 paired tasks per scenario:
the final checkpoint beats `saferl` on 7 of 38 scenarios and loses none,
with no output cost anywhere. Violations per episode, `saferl` → `meta`:
`leak_step` 4.07 → 2.21, `leak_recurrent` 4.41 → 2.97 (+1.8% on-spec
ethanol), `dilution_step` 66.1 → 55.6, `dilution_recurrent` 34.4 → 32.0,
`toxin_step` 9.18 → 6.89, `downstream_rate_recurrent` 11.4 → 9.27,
`turnaround_delay_recurrent` 1.31 → 0.97 (+2.3%). The `storage_capacity`
rows stay tied (~100–117 overflow hours): the lowered wall shows only
through the overflow penalty, and the gate opens 0.05–0.2 there. Selection
picked iteration 590 on one high 10-task undisturbed ethanol reading, which
also stopped training early; the final checkpoint (≈1090) is better on the
faults (5 wins at 590).

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

### Ablation study (`metarl/ablation.py`)

The Fermentation-PPO ablation's design on the `meta` arm: each variant is
`ARMS["meta"]` with components switched off, trained, selected and evaluated
by the study's own code, over several seeds.

| component | ablated by | what else goes with it |
|---|---|---|
| `z` | `use_context=False` | the aux head, privileged critics and the adaptation gate |
| `multitask` | `multitask=False` (nominal task only) | held-out selection on the undisturbed checks only |
| `lagrangian` | `lagrangian=False` | none: the static shaping stays |
| `mask` | `action_mask=False` | none, but the plant has no interlock, so unmasked flows reach the shield and the plant |
| `apn` | `apn=False` | the mapping penalty |
| `warmstart` | `warm_start=None` (from scratch) | the anchor, L2-SP decay, actor warm-up, gate and no-regression selection |

The gate has no switch of its own: it is on exactly when `z` and `warmstart`
both are. `--design factorial --components z multitask` splits `ADAPT` into
the disturbed distribution and inference (`full`, `no_z`, `no_multitask`,
`no_z_multitask`, each against the `saferl` reference row).

Seed k runs under the run tag `ablation/seed<k>`. Its `saferl` trains there
first (copy `policy/saferl*.pth` into `policy/ablation/seed<k>/` to reuse one)
and is evaluated as the reference row. Every variant then trains with the
study's protocol, early stopping included; `--budget N` instead runs each for
exactly N iterations. Evaluation uses `evaluate`'s scenarios and tasks, so
deltas against `full` are paired per task within a seed. Outputs:
`results/log/ablation/summary.csv` and `ablation_curves.png`,
`ablation_final.png`, `ablation_severity.png` in `results/plot/ablation/`.
Leave-one-out over three seeds is 21 meta-sized runs plus three `saferl` runs.

---

## Files

Everything lives in the `metarl` package; the two large classes are split
into mixins, one module per concern.

| module | role |
|---|---|
| `metarl/env/kinetics.py` / `disturbances.py` / `plant.py` / `safe.py` | Fermentation-PPO's environment, ported with only the imports changed: kinetics and RK4, the eleven fault kinds and evaluation scenarios, the plant, Safe PPO's `FermentationEnv` |
| `metarl/env/config.py` | Fermentation-PPO's constants, then the observation offsets and the task distribution |
| `metarl/env/adapter.py` | `PlantFermentationEnv`: the environment behind the agent's interface (flat views, tasks, fault tracking, privileged state, per-reactor ledger) |
| `metarl/env/layout.py` / `faults.py` | observation and action indices and the stage mask; fault kind codes, aliases and presets |
| `metarl/agent/pearl.py` | `PEARLAgent`: the PEARL/SAC agent and its rollout copies |
| `metarl/agent/acting.py` / `learning.py` / `checkpoint.py` | its mixins: posterior and action selection (adaptation gate, APN hooks), the update, save / load / warm start |
| `metarl/agent/networks.py` / `replay.py` / `common.py` | token actor and critics, GRU encoder, gate; per-task replay; devices and utilities |
| `metarl/apn/network.py` / `data.py` / `train.py` | the shield: network and projection, margin label and sampler, training and validation |
| `metarl/training/config.py` | every setting of the study (`NET_*`, `SELECT_*`, fault ranges, scenarios, output paths) |
| `metarl/training/cli.py` | the modes behind `python -m metarl` |
| `metarl/training/arms.py` / `train.py` / `heldout.py` / `early_stopping.py` | arms and their agents, training, held-out checks and checkpoint selection, early stopping |
| `metarl/training/tasks.py` / `rollouts.py` / `workers.py` | task pools and fault schedules; training and deterministic (branched) episodes; the worker processes that run them |
| `metarl/training/evaluate.py` / `z_informativeness.py` | evaluation and the latent gate |
| `metarl/training/metrics.py` / `plots.py` / `logs.py` | statistics, figures, JSON logs |
| `metarl/ablation.py` | ablation study of the meta arm's components (`python -m metarl.ablation`) |
| `tests/smoke_test.py` | fast CPU check of every code path and the design properties |

## How to run

Run from this folder (the repository root). First the smoke test (~40 s; it
prints `ALL SMOKE CHECKS PASSED`, and checks parity with
`../Fermentation-PPO` when that checkout is there):

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
`z_informativeness`, `plot_eval`, `plot_curves`.

The ablation study (leave-one-out, seeds 0–2; `--help` for the options,
`--dry-run` to list the runs, `--summarize-only` to redraw):

```bash
python -m metarl.ablation
``` Seeds and tags, in
PowerShell:

```powershell
$env:PLANT_SEED=1; $env:PLANT_RUN_TAG="seed1"; python -m metarl all
```

| environment variable | effect |
|---|---|
| `PLANT_SEED` | training and evaluation seed (default 0) |
| `PLANT_RUN_TAG` | subfolder for checkpoints, logs and figures |
| `PLANT_APN_PATH` | APN checkpoint path (default `policy/apn_ppo_plant.pth`; `policy/apn_plant.pth` is the previous plant's) |
| `FERMENTATION_PPO_ROOT` | Fermentation-PPO checkout for the smoke test's parity check (default `../Fermentation-PPO`) |
| `PLANT_HELDOUT_WORKERS` | held-out worker processes; `0` or `1` runs in-process |
| `PLANT_COLLECT_WORKERS` | `0` collects training episodes in-process instead of in the worker pool |
| `PLANT_EVAL_EPISODES` | evaluation tasks per scenario (default 100) |
| `PLANT_EVAL_WORKERS` | evaluation worker processes (default: every logical CPU); `0` or `1` runs in-process |
| `PLANT_EARLY_STOP_PATIENCE` | early-stop patience in held-out checks (default 50) |
| `PLANT_DEVICE` | force the training device, e.g. `cpu` |
| `PLANT_VERBOSE_PBAR` | `1` adds losses and latent diagnostics to the progress bar |

Outputs: `policy/[tag]/<arm>.pth` and `<arm>_best.pth`; `results/log/[tag]/`
(`<arm>_log.json`, `eval.json`, `eval_summary.json`, `z_informativeness.json`);
`results/plot/[tag]/`.

## Requirements

`numpy`, `numba`, `torch`, `matplotlib`, `tqdm` (optional). Gradient updates
use a CUDA/ROCm GPU when available (the critics are ~9.5M parameters);
rollouts always run on the CPU, held-out checks, training episodes and
evaluation in worker processes.
