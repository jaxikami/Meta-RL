# Evaluation metrics: violation burden and net production

Two numbers summarise an evaluation. The **violation burden** $B$ combines
how often and how badly constraints are violated. The **net production**
$\Pi$ combines $B$ with output and actuator wear. Both are computed per
episode in `metarl/training/metrics.py` (`violation_burden`,
`net_production`) and reported in `eval.json` (`burden`, `score`, and their
`*_per_ep` arrays). The figures are `compare_burden`, `compare_score` and
`ablation_final`.

## 1. Violation burden

For constraint $c$, instance $i$ (a reactor, a lot, or the plant) and inner
step $t$ ($\Delta t = 1$ h), let $\delta_{c,i,t} \ge 0$ be the relative
exceedance of the limit:

| $c$ | constraint | $\delta$ |
|---|---|---|
| 1 | reactor overflow | $M_i/M_{\max} - 1$ |
| 2 | reactor underflow | $(M_{\text{dry}} - M_i)/M_{\text{dry}}$ |
| 3 | storage overflow | $(S - S_{\text{cap}})/S_{\text{cap}}$ |
| 4 | storage underflow | $(S_{\min} - S)/S_{\min}$ |
| 7 | harvest titer floor (while draining) | $(\tau_{\text{spec}} - \tau_i)/\tau_{\text{spec}}$ |
| 8 | product purity (per rejected lot) | lot shortfall below spec |
| 9 | heat-removal capacity | $(Q - Q_{\text{cap}})/Q_{\text{cap}}$ |
| 5, 6 | numerical instability, mass-balance drift | no depth (event only) |

The burden of an episode is

$$
B \;=\; \sum_{c}\sum_{i}\sum_{t}\Big(\mathbb{1}[\delta_{c,i,t}>0] \;+\; G\,\delta_{c,i,t}\Big)
\;=\; \underbrace{N}_{\text{violation-hours}} \;+\; G\,\underbrace{A}_{\text{exceedance area}},
\qquad G = 4 .
$$

In continuous time this is $\int \big(\mathbb{1}[\delta(t)>0] + G\,\delta(t)\big)\,dt$:
the area under the exceedance curve with a unit step at the limit.

**Why both terms.** The area $A$ alone gives a shallow violation almost no
weight. Fifty hours at 1 % over the limit would score 0.5, the same as one
hour at 50 % over. The count $N$ alone ignores depth. With both, each
violation-hour costs at least 1, and depth adds to that.

**Why $G = 4$.** The environment prices a violation event with the spike
$1 + G\,\delta/(1+\delta)$, with `DEPTH_GAIN` $= G = 4$. Near the limit this
is $1 + G\delta$, which is the summand of $B$. A shallow violation therefore
weighs in $B$ what it costs in the reward. Deeper, $\delta/(1+\delta) \le
\delta$, so $B$ is an upper bound on the reward's price. The reward saturates
in depth to keep training stable. A safety report should not saturate. No
weight is chosen for the report itself.

**Properties.**
- $B = 0$ if and only if the episode has no violation.
- $B$ grows with how often, how long and how deep the violations are.
- $B$ is additive over constraints and time, so it splits into
  per-constraint parts. The stacked panel of `compare_burden` shows this
  split.
- $B$ is dimensionless, in violation-hours at the limit.

## 2. Net production

$$
\Pi \;=\; E_{\text{on}} \;-\; \frac{\kappa\,B \;+\; R_{\text{sm}}}{\rho}
$$

- $E_{\text{on}}$: released, on-spec ethanol (kg). A rejected lot earns
  nothing, so product quality is already in this term.
- $\kappa$ = `VIOLATION_SPIKE` = 0.1: reward per unit of $B$ at the limit.
- $R_{\text{sm}} = \sum_k \sum_i c_s \,\overline{(u_{i,k}-u_{i,k-1})^2}$: the
  reward's flow-change penalty ($c_s = 0.005$). This is the action-smoothness
  term the project asks to report next to violations.
- $\rho$ = `PROD_REWARD_PER_ETH` = $5\times10^{-8}$ reward per kg. This converts
  the other terms to kg of ethanol.

$\Pi$ is the episode's value at the reward's own exchange rates. It differs
from the return in two ways. It drops the learned Lagrange multipliers,
which differ between arms. It also drops the shaping terms (soft walls,
barriers). That is why $\Pi$ can be compared across arms when the return
cannot. The reward's raw-material term is left out because it is about
$10^{-7}$ kg per episode, against about $2\times10^6$ kg of wear.

**Scale.** $\kappa/\rho = 2\times10^6$ kg per unit of $B$. A nominal
episode makes about $7.5\times10^7$ kg on spec, so about 37 units of $B$
cancel its whole output, and $\Pi < 0$ whenever
$B > E_{\text{on}}/(2\times10^6)$. A negative $\Pi$ means the episode's
violations cost more than its product at the reward's rate. This is the
same rate the held-out selection score uses (`heldout.violation_price`).

**Robustness to the price.** The ranking of two arms can depend on
$\kappa$. Report the break-even price

$$
\kappa^{*} \;=\; \rho\,\frac{\Delta E_{\text{on}} - \Delta R_{\text{sm}}/\rho}{\Delta B},
\qquad \Delta x = x_{\text{meta}} - x_{\text{saferl}},
$$

the violation price at which the two arms tie. If $\Delta E_{\text{on}}$ and
$-\Delta B$ have the same sign, one arm dominates at every price, and
$\kappa$ does not matter.

## 3. Reporting

- **Per scenario:** mean ± std over the $n$ paired episodes. Arm
  differences are paired deltas with a 95 % $t$ interval (`_paired_delta`).
  A difference counts as significant when the interval excludes 0 and is
  not negligible.
- **Headline:** $\bar\Pi$ and $\bar B$, the means over the $S$ scenarios with
  each scenario weighted equally, plus the number of scenarios on which each
  arm's $\Pi$ is significantly higher. `evaluate` prints this and stores it
  in `eval_summary.json` under `headline`.
- **Dimensionless form (optional):** $J = \Pi / E_{\text{on}}^{\text{ref}}$
  with the reference arm's in-distribution on-spec ethanol. $J = 1$ is the
  reference's nominal output with no violations and no wear.
- **Ablation:** `ablation_final` shows $B$ and $\Pi$ per configuration and
  scenario, paired against `full`. $\Delta\Pi$ is given as a percentage of
  `full`'s on-spec ethanol, because $\Pi$ itself can be close to zero.

A sentence for the results section might read: "Across $S$ scenarios, meta
reached a mean net production of $\bar\Pi$ against saferl's (higher on $k$,
lower on $l$, tied on the rest; paired over $n$ tasks), with a violation
burden of $\bar B$ against …"

## 4. What these metrics leave out

- **Near misses.** Entering a soft-wall buffer without crossing the limit is
  not a violation, so it adds nothing to $B$. `near_miss_counts` in the
  environment's `info` records them.
- **Recovery time after a fault.** This is in `recovery` in `eval.json`.
- **Checkpoint selection.** Selection still scores violation events, not
  $B$. These metrics change only how results are reported, so the
  checkpoints in `policy/` stay valid.
