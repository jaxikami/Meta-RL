"""Meta-RL under uncertainty on a three-reactor plant with central storage.

Subpackages:

    env       the plant: kinetics, configuration, faults, instruments, ledger
    agent     the PEARL / SAC agent every arm runs
    apn       the action projection network (the shared safety shield)
    training  arms, training, checkpoint selection, evaluation and figures

Run the study with ``python -m metarl <mode>`` (modes in
``metarl.training.cli``).
"""
