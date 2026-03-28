"""JAX-friendly replay buffer with task_id support."""
import numpy as np


class ReplayBuffer:
    """Fixed-size numpy replay buffer storing (s, a, r, s', done, task_id)."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.task_id = np.zeros((capacity,), dtype=np.int32)

    def push(self, obs, act, rew, next_obs, done, task_id=0):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.task_id[self.ptr] = task_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, obs, act, rew, next_obs, done, task_id=None):
        """Push a batch of transitions. Arrays should be [batch, ...]."""
        n = obs.shape[0]
        # Convert from JAX arrays to numpy if needed
        obs = np.asarray(obs)
        act = np.asarray(act)
        rew = np.asarray(rew).reshape(-1, 1) if rew.ndim == 1 else np.asarray(rew)
        next_obs = np.asarray(next_obs)
        done = np.asarray(done).reshape(-1, 1) if done.ndim == 1 else np.asarray(done)
        if task_id is not None:
            task_id = np.asarray(task_id).astype(np.int32)
        else:
            task_id = np.zeros(n, dtype=np.int32)

        if self.ptr + n <= self.capacity:
            self.obs[self.ptr:self.ptr + n] = obs
            self.act[self.ptr:self.ptr + n] = act
            self.rew[self.ptr:self.ptr + n] = rew
            self.next_obs[self.ptr:self.ptr + n] = next_obs
            self.done[self.ptr:self.ptr + n] = done
            self.task_id[self.ptr:self.ptr + n] = task_id
        else:
            # Wrap around
            first = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.act[self.ptr:] = act[:first]
            self.rew[self.ptr:] = rew[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.done[self.ptr:] = done[:first]
            self.task_id[self.ptr:] = task_id[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.act[:rest] = act[first:]
            self.rew[:rest] = rew[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.done[:rest] = done[first:]
            self.task_id[:rest] = task_id[first:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator = None):
        if rng is None:
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "task_id": self.task_id[idx],
        }

    def sample_stacked(self, n_batches: int, batch_size: int):
        """Pre-sample n_batches batches, stacked into [n_batches, batch_size, ...].

        Used with jax.lax.scan to fuse all gradient steps into one JIT call.
        """
        all_idx = np.random.randint(0, self.size, size=(n_batches, batch_size))
        return {
            "obs": self.obs[all_idx],         # [N, B, obs_dim]
            "act": self.act[all_idx],         # [N, B, act_dim]
            "rew": self.rew[all_idx],         # [N, B, 1]
            "next_obs": self.next_obs[all_idx],  # [N, B, obs_dim]
            "done": self.done[all_idx],       # [N, B, 1]
            "task_id": self.task_id[all_idx],  # [N, B]
        }

    def __len__(self):
        return self.size
