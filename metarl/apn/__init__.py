"""Safe PPO's safety layer: the multi-head Action Projection Network (APN).

Fermentation-PPO's ``safety/`` package, ported with only its import paths
changed (``tests/smoke_test.py`` checks the sources still match):

    constants    observation layout, plant constants and head buffers
    kinetic      KineticNet, the learned heads' per-reactor network
    apn          the head registry and MultiHeadAPN (margins, projection, safety loss)
    adaptation   L3: adaptive backoff and the online mass-balance residual
    plant_sim    vectorized single-reactor plant for labelling
    data / train / refiner / validate   the learned heads' data, training,
                 in-training refresh and validation against the real plant
    validation_plots   validate's figure (Fermentation-PPO's reporting/)

Safe PPO's state estimator (``safety/estimator.py``) is not ported: it filters
sensor noise, and this study's observation has none, so it would return the
reading unchanged. The agent's use of the APN is in ``metarl.agent``.
"""
