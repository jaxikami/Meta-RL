# Meta-RL

- Python: `C:/Users/jaxi0/miniconda3/envs/work/python.exe` (the base env's torch is broken). Ruff: `<that python> -m ruff`.
- Check every change with `python tests/smoke_test.py` (~1 min, prints `ALL SMOKE CHECKS PASSED`).

## Do not refactor

- `metarl/env/{kinetics,disturbances,plant,safe}.py` and the Fermentation-PPO half of `metarl/env/config.py` (everything above the "META-RL" banner) are Fermentation-PPO's environment, ported verbatim so the two projects benchmark on one plant. The smoke test fails if they drift. Change them only by re-copying from `../Fermentation-PPO/envs`. Meta-RL behaviour belongs in `metarl/env/adapter.py`. `ruff.toml` excludes these files.
- `metarl/apn/{constants,kinetic,apn,adaptation,plant_sim,data,train,refiner,validate}.py` are Fermentation-PPO's `safety/` package (Safe PPO's multi-head APN), and `metarl/apn/validation_plots.py` is its `reporting/validation_plots.py`. They are ported and checked the same way. Re-copy them with `envs.`→`metarl.env.`, `safety.`→`metarl.apn.` and `reporting.`→`metarl.apn.`. How the agent uses the APN lives in `metarl/agent/` (acting, learning, pearl).
- After a re-copy, check that `adapter.py` still provides what Meta-RL relies on. A Fermentation-PPO refactor removed `reset_lambdas_per_episode` (unused there), so the adapter now zeroes the duals itself, and the smoke test asserts it. Numba recompiles a re-copied `kinetics.py` (`fastmath`), and the first run's bit-parity check can then be off by one ULP: run it again before investigating.
- Changing a network shape, the observation layout or a reward-relevant constant invalidates every checkpoint in `policy/` (and the APN's KineticNet, `policy/apn_kinetic.pth`). Say so before doing it.

## Reference repositories

The global `process-control-rl` skill (`~/.claude/skills/process-control-rl`) distils REINFORCE-PSE, emedd33's tank-control thesis, InstaDeep's outer-value-function paper and awesome-metarl. How they bear on this project:

- **Baselines:** the benchmark is `saferl` vs `meta` only. There is no unconstrained-RL arm, and Standard PPO's `bench.py` is not ported. Neither arm has a classical baseline tuned on the reward. A scripted recipe exists only in the smoke test, and freeze-and-fine-tune is a cheaper adaptation baseline than the `meta` warm start.
- **Smoothness:** the reward already penalises flow-command changes. Report action smoothness next to violations in evaluations.
- **Outer value function:** not applicable as is. No hyperparameter is meta-learned (`gamma` is fixed; task inference is through z). It would matter only if a meta-learned discount or entropy target were added.
- **Literature:** the design follows PEARL and VariBAD. awesome-metarl has no safe or constrained meta-RL to compare against.

## Skills

`.claude/skills/{refactor,refactor-plan,ruff-recursive-fix,lean-comments}` are copied unchanged from github/awesome-copilot (MIT). The comments in this codebase record why settings have their values (often with measured numbers). Under `lean-comments`, those are rationale to keep, not narration.
