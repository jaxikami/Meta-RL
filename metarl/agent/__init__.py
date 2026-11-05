"""The PEARL meta-RL agent (SAC backbone) every arm runs.

    pearl       ``PEARLAgent``: construction and rollout copies
    acting      online posterior and action selection (``ActingMixin``)
    shield      the APN: projection, L3, KineticNet refresh (``ShieldMixin``)
    learning    the meta-gradient update (``LearningMixin``)
    checkpoint  save / load / warm start (``CheckpointMixin``)
    networks    token actor and critics, GRU encoder, aux head, gate
    replay      per-task replay with causal context windows
    common      devices, constants and small utilities
"""
