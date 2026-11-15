"""GPU-native JAX replay buffer — all data lives on device.

Key optimization: eliminates the GPU->CPU->GPU round-trip that the old numpy
buffer required every iteration:
  OLD: rollout(GPU) -> np.array (CPU) -> buffer(CPU) -> np.random idx (CPU) -> jnp.array (GPU)
  NEW: rollout(GPU) -> buffer(GPU) -> jax.random idx (GPU) -> train(GPU)

The buffer uses pre-allocated JAX arrays and in-place updates via .at[].set().
Sampling uses jax.random.randint + jnp.take for fully on-device indexing.

Falls back to CPU-side numpy for:
  - Checkpoint save/load (unavoidable disk I/O)
  - Random exploration phase (sequential env.step returns numpy)
"""
import jax
import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    """Fixed-size replay buffer with JAX device arrays.

    All storage arrays live on the default JAX device (GPU if available).
    push_batch_jax() and sample_stacked() are fully on-device.
    push() for single transitions (random exploration) still works via CPU->GPU.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000,
                 belief_dim: int = 0):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.belief_dim = belief_dim
        self.ptr = 0
        self.size = 0

        # Pre-allocate on device
        self.obs = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.act = jnp.zeros((capacity, act_dim), dtype=jnp.float32)
        self.rew = jnp.zeros((capacity, 1), dtype=jnp.float32)
        self.next_obs = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.done = jnp.zeros((capacity, 1), dtype=jnp.float32)
        self.task_id = jnp.zeros((capacity,), dtype=jnp.int32)
        # True when this observation starts a fresh simulator trajectory.
        # Recurrent ESCP uses it to prevent replay histories from crossing
        # episode or rollout-reset boundaries.
        self.episode_start = jnp.zeros((capacity,), dtype=jnp.bool_)
        # GPT-5.5 advice #2: per-transition belief storage. When belief_dim==0
        # the array is empty and sampling returns zero-width belief tensors,
        # preserving backward compat with non-belief-conditioned algos.
        self.belief = jnp.zeros((capacity, belief_dim), dtype=jnp.float32)
        self.next_belief = jnp.zeros(
            (capacity, belief_dim), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Single-transition push (random exploration phase — infrequent)
    # ------------------------------------------------------------------
    def push(self, obs, act, rew, next_obs, done, task_id=0,
             belief=None, next_belief=None, episode_start=False):
        """Push one transition. Accepts numpy arrays (auto-converts)."""
        i = self.ptr
        self.obs = self.obs.at[i].set(jnp.asarray(obs, dtype=jnp.float32))
        self.act = self.act.at[i].set(jnp.asarray(act, dtype=jnp.float32))
        self.rew = self.rew.at[i].set(jnp.float32(rew))
        self.next_obs = self.next_obs.at[i].set(jnp.asarray(next_obs, dtype=jnp.float32))
        self.done = self.done.at[i].set(jnp.float32(done))
        self.task_id = self.task_id.at[i].set(jnp.int32(task_id))
        self.episode_start = self.episode_start.at[i].set(
            jnp.bool_(episode_start))
        if self.belief_dim > 0:
            if belief is not None:
                self.belief = self.belief.at[i].set(
                    jnp.asarray(belief, dtype=jnp.float32))
            if next_belief is not None:
                self.next_belief = self.next_belief.at[i].set(
                    jnp.asarray(next_belief, dtype=jnp.float32))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Batch push — accepts JAX arrays directly (zero-copy from rollout)
    # ------------------------------------------------------------------
    def push_batch_jax(self, obs, act, rew, next_obs, done, task_id=None,
                       belief=None, next_belief=None, episode_start=None):
        """Push a batch of transitions from JAX arrays (no CPU transfer).

        Args:
            obs, next_obs: [N, obs_dim] jax arrays
            act: [N, act_dim] jax array
            rew: [N, 1] or [N] jax array
            done: [N, 1] or [N] jax array
            task_id: [N] jax/numpy int array, or None
            belief: [belief_dim] (broadcast to all N) or [N, belief_dim],
                    or None (zeros). GPT-5.5 advice #2: store the belief
                    that was active at rollout time so off-policy critic
                    updates can use the matching belief, not the current
                    iter's belief.
            next_belief: context after observing each transition. BAPR-v2
                    uses this for causal Bellman targets.
        """
        n = obs.shape[0]
        rew = rew.reshape(-1, 1) if rew.ndim == 1 else rew
        done = done.reshape(-1, 1) if done.ndim == 1 else done
        if task_id is None:
            task_id = jnp.zeros(n, dtype=jnp.int32)
        else:
            task_id = jnp.asarray(task_id, dtype=jnp.int32)
        if episode_start is None:
            episode_start = jnp.concatenate([
                jnp.ones((1,), dtype=jnp.bool_),
                done[:-1, 0] > 0.5,
            ])
        else:
            episode_start = jnp.asarray(
                episode_start, dtype=jnp.bool_).reshape((n,))

        if self.belief_dim > 0:
            def prepare(value):
                if value is None:
                    return jnp.zeros(
                        (n, self.belief_dim), dtype=jnp.float32)
                b = jnp.asarray(value, dtype=jnp.float32)
                if b.ndim == 1:
                    return jnp.broadcast_to(
                        b[None, :], (n, self.belief_dim))
                return b

            belief_batch = prepare(belief)
            next_belief_batch = prepare(next_belief)

        # Compute insertion indices (handles wrap-around)
        idx = (jnp.arange(n) + self.ptr) % self.capacity

        self.obs = self.obs.at[idx].set(obs)
        self.act = self.act.at[idx].set(act)
        self.rew = self.rew.at[idx].set(rew)
        self.next_obs = self.next_obs.at[idx].set(next_obs)
        self.done = self.done.at[idx].set(done)
        self.task_id = self.task_id.at[idx].set(task_id)
        self.episode_start = self.episode_start.at[idx].set(episode_start)
        if self.belief_dim > 0:
            self.belief = self.belief.at[idx].set(belief_batch)
            self.next_belief = self.next_belief.at[idx].set(
                next_belief_batch)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def clear(self):
        """Logically clear replay without reallocating device storage."""
        self.ptr = 0
        self.size = 0

    # ------------------------------------------------------------------
    # Legacy batch push (numpy) — used by checkpoint restore & compat
    # ------------------------------------------------------------------
    def push_batch(self, obs, act, rew, next_obs, done, task_id=None):
        """Push a batch of numpy transitions (converts to JAX)."""
        self.push_batch_jax(
            jnp.asarray(obs), jnp.asarray(act),
            jnp.asarray(rew), jnp.asarray(next_obs),
            jnp.asarray(done),
            jnp.asarray(task_id) if task_id is not None else None)

    # ------------------------------------------------------------------
    # Sampling — fully on device
    # ------------------------------------------------------------------
    def sample_stacked(self, n_batches: int, batch_size: int,
                       rng_key=None):
        """Sample n_batches x batch_size transitions, fully on GPU.

        Returns dict of JAX arrays with shape [n_batches, batch_size, ...].
        Used with jax.lax.scan — output goes directly to _scan_update.
        """
        if rng_key is None:
            # Fallback: generate a JAX key from numpy random state
            rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

        # [n_batches, batch_size] random indices — all on device
        idx = jax.random.randint(
            rng_key, (n_batches, batch_size), 0, self.size)

        out = {
            "obs": self.obs[idx],            # [N, B, obs_dim]
            "act": self.act[idx],            # [N, B, act_dim]
            "rew": self.rew[idx],            # [N, B, 1]
            "next_obs": self.next_obs[idx],  # [N, B, obs_dim]
            "done": self.done[idx],          # [N, B, 1]
            "task_id": self.task_id[idx],    # [N, B]
        }
        if self.belief_dim > 0:
            out["belief"] = self.belief[idx]  # [N, B, belief_dim]
            out["next_belief"] = self.next_belief[idx]
        return out

    def sample_stacked_sequences(self, n_batches: int, batch_size: int,
                                 history_length: int, rng_key=None):
        """Sample reset-aware causal histories for recurrent ESCP.

        Each sampled update ends at one uniformly selected replay transition.
        Prefix rows before the oldest available transition are zero padded.
        ``episode_start`` clears the recurrent state, so a window may be
        gathered efficiently across storage boundaries without leaking history
        across independent simulator trajectories.
        """
        history_length = int(history_length)
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.size <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        if rng_key is None:
            rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

        end_logical = jax.random.randint(
            rng_key, (n_batches, batch_size), 0, self.size)
        offsets = jnp.arange(
            1 - history_length, 1, dtype=jnp.int32)
        logical = end_logical[..., None] + offsets
        valid = logical >= 0
        oldest = self.ptr if self.size == self.capacity else 0
        physical = (jnp.maximum(logical, 0) + oldest) % self.capacity

        observations = self.obs[physical]
        observations = jnp.where(
            valid[..., None], observations, jnp.zeros_like(observations))
        starts = jnp.logical_or(
            ~valid, self.episode_start[physical])

        previous_logical = logical - 1
        previous_valid = previous_logical >= 0
        previous_physical = (
            jnp.maximum(previous_logical, 0) + oldest) % self.capacity
        previous_actions = self.act[previous_physical]
        previous_actions = jnp.where(
            jnp.logical_and(previous_valid, ~starts)[..., None],
            previous_actions,
            jnp.zeros_like(previous_actions),
        )

        final_physical = (end_logical + oldest) % self.capacity
        actions = self.act[final_physical]
        rewards = self.rew[final_physical]
        next_observation = self.next_obs[final_physical]
        dones = self.done[final_physical]
        task_ids = self.task_id[final_physical]

        next_observations = jnp.concatenate([
            observations[..., 1:, :],
            next_observation[..., None, :],
        ], axis=-2)
        next_previous_actions = jnp.concatenate([
            previous_actions[..., 1:, :],
            actions[..., None, :],
        ], axis=-2)
        next_starts = jnp.concatenate([
            starts[..., 1:],
            jnp.zeros(dones.shape[:-1] + (1,), dtype=jnp.bool_),
        ], axis=-1)

        return {
            "obs": observations,
            "prev_act": previous_actions,
            "reset_before": starts,
            "act": actions,
            "rew": rewards,
            "next_obs": next_observations,
            "next_prev_act": next_previous_actions,
            "next_reset_before": next_starts,
            "done": dones,
            "task_id": task_ids,
        }

    def sample_stacked_mixed(self, n_batches: int, batch_size: int,
                             rng_key=None, recent_frac: float = 0.0,
                             recent_window: int = 50_000):
        """Belief-aware sampling: uniform + recent-transitions mixture.

        Change 3 (GPT-5.5 v2). When BOCD detects regime change (high λ_w),
        BAPR's training loop sets recent_frac proportional to λ_w, biasing
        replay toward recent transitions. This addresses the "regime
        staleness" problem the paper claims to solve — without it, replay
        is uniform and BOCD-driven adaptive β is the only intervention.

        Args:
            n_batches, batch_size: standard
            rng_key: JAX PRNGKey
            recent_frac: fraction of batch from recent_window suffix [0, 0.9]
            recent_window: how many past samples count as "recent"
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

        recent_frac = float(np.clip(recent_frac, 0.0, 0.9))
        n_recent = int(batch_size * recent_frac)
        n_uniform = batch_size - n_recent

        k1, k2 = jax.random.split(rng_key)
        idx_uniform = jax.random.randint(
            k1, (n_batches, n_uniform), 0, self.size)

        if n_recent > 0:
            lo = max(0, self.size - recent_window)
            idx_recent = jax.random.randint(
                k2, (n_batches, n_recent), lo, self.size)
            idx = jnp.concatenate([idx_uniform, idx_recent], axis=1)
        else:
            idx = idx_uniform

        out = {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "task_id": self.task_id[idx],
        }
        if self.belief_dim > 0:
            out["belief"] = self.belief[idx]
            out["next_belief"] = self.next_belief[idx]
        return out

    def sample(self, batch_size: int, rng: np.random.Generator = None):
        """Sample one batch. Returns dict of JAX arrays [B, ...]."""
        key = jax.random.PRNGKey(
            np.random.randint(0, 2**31) if rng is None
            else int(rng.integers(0, 2**31)))
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        out = {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "task_id": self.task_id[idx],
        }
        if self.belief_dim > 0:
            out["belief"] = self.belief[idx]
            out["next_belief"] = self.next_belief[idx]
        return out

    # ------------------------------------------------------------------
    # Numpy conversion for checkpoint save/load
    # ------------------------------------------------------------------
    def to_numpy(self):
        """Export buffer contents as numpy dict (for checkpoint save)."""
        s = self.size
        out = {
            'obs': np.array(self.obs[:s]),
            'act': np.array(self.act[:s]),
            'rew': np.array(self.rew[:s]),
            'next_obs': np.array(self.next_obs[:s]),
            'done': np.array(self.done[:s]),
            'task_id': np.array(self.task_id[:s]),
            'episode_start': np.array(self.episode_start[:s]),
            'ptr': self.ptr,
            'size': self.size,
        }
        if self.belief_dim > 0:
            out['belief'] = np.array(self.belief[:s])
            out['next_belief'] = np.array(self.next_belief[:s])
        return out

    def from_numpy(self, buf_dict):
        """Restore buffer contents from numpy dict (for checkpoint load)."""
        s = int(buf_dict['size'])
        # Write into pre-allocated device arrays
        idx = jnp.arange(s)
        self.obs = self.obs.at[idx].set(jnp.array(buf_dict['obs']))
        self.act = self.act.at[idx].set(jnp.array(buf_dict['act']))
        self.rew = self.rew.at[idx].set(jnp.array(buf_dict['rew']))
        self.next_obs = self.next_obs.at[idx].set(jnp.array(buf_dict['next_obs']))
        self.done = self.done.at[idx].set(jnp.array(buf_dict['done']))
        self.task_id = self.task_id.at[idx].set(jnp.array(buf_dict['task_id']))
        if 'episode_start' in buf_dict:
            starts = jnp.asarray(buf_dict['episode_start'], dtype=jnp.bool_)
        else:
            dones = jnp.asarray(buf_dict['done'])[:s, 0] > 0.5
            starts = jnp.concatenate([
                jnp.ones((1,), dtype=jnp.bool_), dones[:-1]
            ]) if s > 0 else jnp.zeros((0,), dtype=jnp.bool_)
        self.episode_start = self.episode_start.at[idx].set(starts)
        # Belief: load if present (forward compat with non-belief checkpoints)
        if self.belief_dim > 0 and 'belief' in buf_dict:
            self.belief = self.belief.at[idx].set(jnp.array(buf_dict['belief']))
        if self.belief_dim > 0 and 'next_belief' in buf_dict:
            self.next_belief = self.next_belief.at[idx].set(
                jnp.array(buf_dict['next_belief']))
        self.ptr = int(buf_dict['ptr'])
        self.size = s

    def __len__(self):
        return self.size
