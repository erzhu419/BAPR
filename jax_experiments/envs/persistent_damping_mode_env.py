"""Untouched persistent joint-damping regimes for oracle-headroom tests."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


FAMILY = "joint_damping_fault"
MODE_PATTERNS = ("low_half", "high_half", "even", "odd")
BASE_DAMPING_MULTIPLIER = 4.0
ACTION_NOISE_STD = 0.02


def _affected_mask(pattern: str, count: int) -> np.ndarray:
    indices = np.arange(int(count))
    split = (int(count) + 1) // 2
    if pattern == "low_half":
        return indices < split
    if pattern == "high_half":
        return indices >= split
    if pattern == "even":
        return indices % 2 == 0
    if pattern == "odd":
        return indices % 2 == 1
    raise ValueError(f"unknown damping pattern {pattern!r}")


class PersistentDampingModeEnv(StochasticModeEnv):
    """Persistent equal-log-norm damping faults with small action noise."""

    def __init__(self, env_name: str, family: str = FAMILY,
                 dwell_steps: int = 500, dwell_distribution: str = "fixed",
                 seed: int = 0, backend: str = "spring",
                 fixed_mode_id: int | None = None):
        if str(family) != FAMILY:
            raise ValueError(
                f"PersistentDampingModeEnv only supports {FAMILY!r}, "
                f"got {family!r}")
        # Reuse the mature persistent-mode clock and stochastic rollout code.
        # The temporary parent family is replaced before any user transition.
        super().__init__(
            env_name=env_name,
            family="deterministic_mean",
            dwell_steps=dwell_steps,
            dwell_distribution=dwell_distribution,
            seed=seed,
            backend=backend,
            fixed_mode_id=fixed_mode_id,
        )
        self.family = FAMILY
        self._tasks = self._build_tasks()
        self._build_mode_systems(self._tasks)
        initial_mode = 0 if self.fixed_mode_id is None else self.fixed_mode_id
        self._step_counter = 0
        self._switch_history = [(0, int(initial_mode))]
        self._activate_mode(int(initial_mode))
        self._next_switch_step = self._sample_dwell()

    def _positive_damping_indices(self) -> np.ndarray:
        damping = np.asarray(
            self.base_sys.dof.damping, dtype=np.float32).reshape(-1)
        indices = np.flatnonzero(damping > 1e-8)
        if len(indices) != int(self.act_dim):
            raise ValueError(
                "joint-damping benchmark requires one positive damping "
                f"coordinate per action: positive={len(indices)} "
                f"actions={self.act_dim}")
        return indices

    def _build_tasks(self):
        if getattr(self, "family", None) != FAMILY:
            return super()._build_tasks()
        base_gravity = np.asarray(self.base_sys.gravity, dtype=np.float32)
        base_damping = np.asarray(
            self.base_sys.dof.damping, dtype=np.float32)
        positive = self._positive_damping_indices()
        tasks = []
        for mode_id, pattern in enumerate(MODE_PATTERNS):
            affected = _affected_mask(pattern, len(positive))
            affected_count = int(np.sum(affected))
            if affected_count <= 0:
                raise ValueError(
                    f"damping mode {pattern} affects no action coordinate")
            # Equalize the L2 norm of the log-parameter displacement despite
            # odd action dimensions having unequal subset cardinalities.
            log_multiplier = (
                np.log(BASE_DAMPING_MULTIPLIER)
                * np.sqrt(float(len(positive)) / float(affected_count))
            )
            multiplier = float(np.exp(log_multiplier))
            damping = np.array(base_damping, copy=True)
            damping[positive[affected]] *= multiplier
            log_ratio = np.log(np.maximum(
                damping[positive] / base_damping[positive], 1e-12))
            tasks.append({
                "mode_id": int(mode_id),
                "gravity": np.array(base_gravity, copy=True),
                "gravity_scale": 1.0,
                "dof_damping": damping.astype(np.float32),
                "damping_multiplier": np.where(
                    affected, multiplier, 1.0).astype(np.float32),
                "damping_pattern_id": int(mode_id),
                "damping_log_l2": float(np.linalg.norm(log_ratio)),
                "action_gain": 1.0,
                "action_noise_std": ACTION_NOISE_STD,
                "packet_loss_prob": 0.0,
                "burst_prob": 0.0,
                "burst_std": 0.0,
            })
        return tasks

    def _build_mode_systems(self, tasks) -> None:
        systems = []
        for task in tasks:
            replacements = {
                "gravity": jnp.asarray(task["gravity"], dtype=jnp.float32),
            }
            if "dof_damping" in task:
                replacements["dof.damping"] = jnp.asarray(
                    task["dof_damping"], dtype=jnp.float32)
            systems.append(self.base_sys.tree_replace(replacements))
        self._mode_sys = systems

    def configure_eval_switching(self, tasks, period_steps: int):
        vectors = np.asarray([
            np.concatenate([
                np.asarray(task["dof_damping"], dtype=np.float64).reshape(-1),
                np.asarray(task["damping_multiplier"],
                           dtype=np.float64).reshape(-1),
            ])
            for task in tasks
        ])
        scale = np.std(vectors, axis=0)
        normalized = (vectors - np.mean(vectors, axis=0)) / np.where(
            scale > 1e-8, scale, 1.0)
        distance = np.sum(
            np.square(normalized[:, None] - normalized[None, :]), axis=-1)
        np.fill_diagonal(distance, -np.inf)
        first, second = np.unravel_index(np.argmax(distance), distance.shape)
        pair = [int(first), int(second)]
        if self._eval_sequence_calls % 2:
            pair.reverse()
        self._eval_sequence_calls += 1
        self.configure_eval_mode_sequence(tasks, pair, period_steps)

