"""The three-arm study: training, checkpoint selection, evaluation, figures.

    cli                entry point (``python -m metarl <mode>``)
    config             every setting of the study
    arms               the arms, their checkpoints and agents
    train              training one arm
    heldout            held-out checks, no-regression, worker pool
    early_stopping     plateau / divergence stopping
    tasks              environments, task pools, fault schedules
    rollouts           training and deterministic episodes
    evaluate           every arm on every scenario
    z_informativeness  the latent gate
    metrics            separability, gate openings, paired statistics
    plots              figures
    logs               JSON logs
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
