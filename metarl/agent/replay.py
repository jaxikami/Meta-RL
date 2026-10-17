"""Per-task replay with causal context windows for the recurrent encoder."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import TASK_CONTEXT_DIM
from metarl.agent.common import device

# Transitions per contiguous chunk in ``sample_windows``.
WINDOW_CHUNK_LEN = 8


def _to_device(arrays):
    """Upload NumPy arrays to the training device, one copy each."""
    return tuple(torch.as_tensor(a).to(device) for a in arrays)


def _as_long(x):
    """An int64 index tensor on the training device."""
    return torch.as_tensor(x, dtype=torch.long, device=device)


class _TaskRing:
    """Fixed-capacity FIFO of transitions stored as parallel float32 arrays.

    Logical index 0 is the oldest transition. Each transition records
    ``ep_step`` (its index within its episode), the privileged fault code
    (0 until a fault has acted, then the first acting kind's code) and,
    when built with ``priv_dim``, the plant's privileged state before and
    after it.
    """

    def __init__(self, obs_dim, action_dim, capacity, initial=4096,
                 priv_dim=0):
        self.capacity = int(capacity)
        self.priv_dim = int(priv_dim)
        self.n = 0
        self.head = 0
        self._next_ep_step = 0
        self._alloc(min(self.capacity, initial), obs_dim, action_dim)

    def _alloc(self, rows, obs_dim, action_dim):
        self.obs = np.empty((rows, obs_dim), dtype=np.float32)
        self.act = np.empty((rows, action_dim), dtype=np.float32)
        self.rew = np.empty(rows, dtype=np.float32)
        self.next_obs = np.empty((rows, obs_dim), dtype=np.float32)
        self.done = np.empty(rows, dtype=np.float32)
        self.ctx = np.empty((rows, TASK_CONTEXT_DIM), dtype=np.float32)
        self.fault = np.empty(rows, dtype=np.float32)
        self.priv = np.empty((rows, self.priv_dim), dtype=np.float32)
        self.next_priv = np.empty((rows, self.priv_dim), dtype=np.float32)
        self.ep_step = np.empty(rows, dtype=np.int64)

    def _fields(self):
        return (self.obs, self.act, self.rew, self.next_obs, self.done,
                self.ctx, self.fault, self.priv, self.next_priv)

    def append(self, obs, act, rew, next_obs, done, ctx, fault=0.0,
               priv=None, next_priv=None):
        """Write one transition, overwriting the oldest when full."""
        if self.priv_dim and (priv is None or next_priv is None):
            raise ValueError(f"this replay stores a {self.priv_dim}-wide "
                             f"privileged state per transition; pass priv "
                             f"and next_priv")
        rows = len(self.rew)
        if self.n == rows and rows < self.capacity:
            old = self._fields() + (self.ep_step,)
            self._alloc(min(self.capacity, 2 * rows), self.obs.shape[1],
                        self.act.shape[1])
            for new, prev in zip(self._fields() + (self.ep_step,), old):
                new[:rows] = prev
        i = self.head
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ctx[i] = ctx
        self.fault[i] = fault
        if self.priv_dim:
            self.priv[i] = priv
            self.next_priv[i] = next_priv
        self.ep_step[i] = self._next_ep_step
        self._next_ep_step = 0 if done else self._next_ep_step + 1
        self.head = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def _physical(self, logical):
        oldest = (self.head - self.n) % self.capacity
        return (oldest + np.asarray(logical)) % self.capacity

    def gather_np(self, logical, context_only=False):
        p = self._physical(logical)
        fields = self._fields()[:4] if context_only else self._fields()
        return tuple(f[p] for f in fields)

    def gather(self, logical):
        return _to_device(self.gather_np(logical))

    def episode_steps(self, logical):
        return self.ep_step[self._physical(logical)]


class MultiTaskReplayBuffer:
    """Per-task replay buffer supporting RL batches and causal context windows."""

    def __init__(self, obs_dim, action_dim, capacity=100000, priv_dim=0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.priv_dim = int(priv_dim)
        self._tasks = {}

    def add(self, task_id, obs, act_exec, reward, next_obs, done, context=None,
            fault=False, priv=None, next_priv=None):
        """Append one transition (the EXECUTED action) to a task's bucket."""
        buf = self._tasks.get(task_id)
        if buf is None:
            buf = self._tasks[task_id] = _TaskRing(
                self.obs_dim, self.action_dim, self.capacity,
                priv_dim=self.priv_dim)
        buf.append(obs, act_exec, reward, next_obs, 1.0 if done else 0.0,
                   0.0 if context is None else context,
                   max(0.0, float(fault)), priv=priv, next_priv=next_priv)

    def task_ids(self):
        return [t for t, b in self._tasks.items() if b.n > 0]

    def size(self, task_id):
        buf = self._tasks.get(task_id)
        return 0 if buf is None else buf.n

    def sample_batch_np(self, task_id, batch_size):
        buf = self._tasks[task_id]
        idx = np.random.randint(0, buf.n, size=batch_size)
        return buf.gather_np(idx)

    def sample_batch(self, task_id, batch_size):
        return _to_device(self.sample_batch_np(task_id, batch_size))

    def sample_windows_np(self, task_id, batch_size, window,
                          chunk_len=WINDOW_CHUNK_LEN, starts=None):
        """``sample_windows`` as NumPy arrays, before any device upload."""
        buf = self._tasks[task_id]
        n = buf.n
        L = max(1, min(int(chunk_len), n))
        W = int(window)
        if starts is None:
            K = -(-int(batch_size) // L)
            starts = np.random.randint(0, n - L + 1, size=K)
        starts = np.asarray(starts, dtype=np.int64)
        K = len(starts)
        p = np.arange(L)
        idx = starts[:, None] + p[None, :]
        k_ep = buf.episode_steps(idx.reshape(-1)).reshape(K, L)
        n_ctx = np.minimum(np.minimum(k_ep, W), idx)
        hi = W + p[None, :].repeat(K, axis=0)
        lo = hi - n_ctx
        batch = buf.gather_np(idx.reshape(-1))
        R = W + L - 1
        row_idx = starts[:, None] - W + np.arange(R)[None, :]
        o, a, r, o2 = buf.gather_np(np.clip(row_idx, 0, n - 1).reshape(-1),
                                    context_only=True)
        rows = (o.reshape(K, R, -1), a.reshape(K, R, -1), r.reshape(K, R),
                o2.reshape(K, R, -1))
        return batch, rows, lo, hi

    def sample_windows(self, task_id, batch_size, window,
                       chunk_len=WINDOW_CHUNK_LEN, starts=None):
        """Sample an RL batch together with each transition's causal context.

        Each transition is paired with the episode's previous
        ``min(k, window)`` transitions, exactly the context its action was
        taken under online. Returns (batch, rows, lo, hi); see the lean agent.
        """
        batch, rows, lo, hi = self.sample_windows_np(
            task_id, batch_size, window, chunk_len=chunk_len, starts=starts)
        return _to_device(batch), _to_device(rows), _as_long(lo), _as_long(hi)
