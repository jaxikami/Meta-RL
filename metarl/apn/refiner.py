"""DAgger-style refresh of the APN's learned heads during PPO training."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.apn.data import DATA_PATH
from metarl.apn.train import fit, to_tensors

class APNRefiner:
    """DAgger-style refresh of KineticNet on the states the policy visits.

    KineticNet is pretrained on random behaviour.  As the policy improves it
    spends its time in states that behaviour rarely reached, so every ``every``
    PPO updates the refiner relabels a sample of the policy's own (state,
    action) pairs in the vectorized plant, with the actions perturbed so the
    heads keep an accurate gradient around them, and fine-tunes KineticNet on
    them mixed with replayed pretraining data so it does not forget the rest.

    Attributes:
        refreshes: Number of fine-tunes done so far.
    """

    def __init__(self, kinetic, device, every=5, n_label=2048, steps=150, lr=1e-4,
                 buffer_size=20000, replay_size=50000, noise=0.3, seed=0, data_path=DATA_PATH):
        """Set up the buffer and load the replay data.

        Args:
            kinetic: The APN's KineticNet (fine-tuned in place, frozen between refreshes).
            device: Torch device of the KineticNet.
            every: PPO updates between refreshes.
            n_label: Policy pairs relabelled per refresh.
            steps: Fine-tuning minibatch steps per refresh.
            lr: Fine-tuning learning rate.
            buffer_size: Reservoir of the policy's per-reactor (state, action) pairs.
            replay_size: Pretraining pairs kept for replay.
            noise: Std of the Gaussian perturbation of the policy's actions.
            seed: Seed.
            data_path: Pretraining dataset to replay (skipped if missing).
        """
        self.kinetic = kinetic
        self.device = device
        self.every = every
        self.n_label = n_label
        self.steps = steps
        self.lr = lr
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.obs = np.empty((buffer_size, 25), dtype=np.float32)
        self.act = np.empty((buffer_size, 4), dtype=np.float32)
        self.seen = 0
        self.updates = 0
        self.refreshes = 0
        self.replay = None
        if os.path.exists(data_path):
            data = dict(np.load(data_path))
            idx = self.rng.choice(len(data["obs"]), size=min(replay_size, len(data["obs"])),
                                  replace=False)
            self.replay = to_tensors({k: v[idx] for k, v in data.items()}, device)

    def add(self, states, actions):
        """Reservoir-sample the per-reactor pairs of one rollout.

        Args:
            states: Observations [T, n, 28].
            actions: Actions sent to the plant [T, n, 4].
        """
        obs = np.asarray(states, dtype=np.float32)[..., :25].reshape(-1, 25)
        act = np.asarray(actions, dtype=np.float32).reshape(-1, 4)
        size = len(self.obs)
        for o, a in zip(obs, act):
            if self.seen < size:
                slot = self.seen
            else:
                slot = int(self.rng.integers(0, self.seen + 1))
                if slot >= size:
                    self.seen += 1
                    continue
            self.obs[slot] = o
            self.act[slot] = a
            self.seen += 1

    def step(self):
        """Count one PPO update; refresh KineticNet when one is due.

        Returns:
            True if a refresh ran.
        """
        from metarl.apn.plant_sim import label, masked_action

        self.updates += 1
        filled = min(self.seen, len(self.obs))
        if self.updates % self.every or filled < self.n_label:
            return False
        idx = self.rng.choice(filled, size=self.n_label, replace=False)
        obs = self.obs[idx]
        stage = obs[:, 16:21].argmax(1)
        action = masked_action(stage, self.act[idx] + self.rng.normal(0.0, self.noise, (len(idx), 4)))
        heat, titer, _ = label(obs, None, action, self.kinetic.heat_steps, workers=1)
        fresh = to_tensors({"obs": obs, "action": action, "heat": heat.astype(np.float32),
                            "titer": titer.astype(np.float32)}, self.device)
        if self.replay is not None:
            pick = torch.randint(0, len(self.replay[0]), (2 * self.n_label,), device=self.device)
            fresh = tuple(torch.cat([f, r[pick]]) for f, r in zip(fresh, self.replay))
        fit(self.kinetic, fresh, self.steps, self.lr, batch=min(1024, len(fresh[0])))
        self.refreshes += 1
        return True
