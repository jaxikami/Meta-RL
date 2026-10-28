"""The three-reactor plant with central storage: Fermentation-PPO's environment.

Ported from Fermentation-PPO's ``envs/`` with only the import paths changed,
so Safe PPO and the arms here are benchmarked on one plant
(``tests/smoke_test.py`` steps both side by side when ../Fermentation-PPO is
present):

    kinetics      Numba kinetics and RK4 integrator          (envs/kinetics.py)
    disturbances  mid-episode faults and evaluation scenarios (envs/disturbances.py)
    plant         ``FermentationPlant``: what both envs share (envs/plant.py)
    safe          ``FermentationEnv``: Lagrangian shaping     (envs/safe.py)
    bench         ``FermentationEnvBench``: static penalties  (envs/bench.py)
    config        their constants (envs/config.py), then this project's
                  observation offsets and task distribution

This project's own modules:

    layout        the flat observation / action layout and the stage mask
    faults        fault kind codes, aliases and selection presets
    adapter       ``PlantFermentationEnv`` / ``BenchPlantFermentationEnv``:
                  the two environments behind the meta-RL interface (tasks,
                  fault tracking, flat views, per-reactor ledger)
"""
