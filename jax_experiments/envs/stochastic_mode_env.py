"""Persistent hidden-mode Brax environments with transition noise.

Each mode owns fixed mean physics and actuator parameters for its full dwell.
Only exogenous actuator events are sampled per step.  This keeps morphology
temporally coherent while making a mode a transition distribution rather than
a single deterministic dynamics function.  Packet-loss, burst-torque,
structured-channel, and actuator-polarity families are oracle-headroom
protocols: they change the persistent transition distribution without changing
the robot body at every step.
"""
from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.envs.brax_env import (
    BraxNonstationaryEnv,
    apply_action_disturbance,
)


MODE_FAMILIES = {
    "deterministic_mean": (
        {"gravity_scale": 1.00, "action_gain": 1.00, "action_noise_std": 0.00},
        {"gravity_scale": 0.75, "action_gain": 1.00, "action_noise_std": 0.00},
        {"gravity_scale": 1.25, "action_gain": 1.00, "action_noise_std": 0.00},
        {"gravity_scale": 1.00, "action_gain": 0.70, "action_noise_std": 0.00},
    ),
    "variance_only": (
        {"gravity_scale": 1.00, "action_gain": 1.00, "action_noise_std": 0.00},
        {"gravity_scale": 1.00, "action_gain": 1.00, "action_noise_std": 0.05},
        {"gravity_scale": 1.00, "action_gain": 1.00, "action_noise_std": 0.12},
        {"gravity_scale": 1.00, "action_gain": 1.00, "action_noise_std": 0.22},
    ),
    "mean_variance": (
        {"gravity_scale": 1.00, "action_gain": 1.00, "action_noise_std": 0.02},
        {"gravity_scale": 0.75, "action_gain": 0.85, "action_noise_std": 0.08},
        {"gravity_scale": 1.25, "action_gain": 1.10, "action_noise_std": 0.12},
        {"gravity_scale": 1.00, "action_gain": 0.70, "action_noise_std": 0.20},
    ),
    "packet_loss": (
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "packet_loss_prob": 0.00},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "packet_loss_prob": 0.08},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "packet_loss_prob": 0.18},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "packet_loss_prob": 0.32},
    ),
    "burst_torque": (
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "burst_prob": 0.00,
         "burst_std": 0.00},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "burst_prob": 0.03,
         "burst_std": 0.25},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "burst_prob": 0.08,
         "burst_std": 0.45},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_noise_std": 0.00, "burst_prob": 0.15,
         "burst_std": 0.70},
    ),
    # Equal-severity, qualitatively different persistent actuator faults.  All
    # modes retain the same small aleatoric execution noise; only the affected
    # channel subset changes.  This avoids the scalar severity ladder in which
    # one conservative fixed policy can dominate every mode.
    "structured_channel": (
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "low_half", "impaired_gain": 0.45,
         "action_noise_std": 0.04},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "high_half", "impaired_gain": 0.45,
         "action_noise_std": 0.04},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "even", "impaired_gain": 0.45,
         "action_noise_std": 0.04},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "odd", "impaired_gain": 0.45,
         "action_noise_std": 0.04},
    ),
    # Persistent actuator-calibration faults with genuinely conflicting
    # control mappings. Every mode reverses a different approximately half
    # of the motor channels while retaining identical execution noise and
    # robot physics.
    "actuator_polarity": (
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "low_half", "impaired_gain": -1.00,
         "action_noise_std": 0.02},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "high_half", "impaired_gain": -1.00,
         "action_noise_std": 0.02},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "even", "impaired_gain": -1.00,
         "action_noise_std": 0.02},
        {"gravity_scale": 1.00, "action_gain": 1.00,
         "action_gain_pattern": "odd", "impaired_gain": -1.00,
         "action_noise_std": 0.02},
    ),
}


class StochasticModeEnv(BraxNonstationaryEnv):
    """Four persistent modes with mode-conditioned actuator distributions."""

    def __init__(self, env_name: str, family: str = "mean_variance",
                 dwell_steps: int = 500, dwell_distribution: str = "fixed",
                 seed: int = 0, backend: str = "spring",
                 fixed_mode_id: int | None = None):
        if family not in MODE_FAMILIES:
            raise ValueError(
                f"unknown stochastic mode family {family!r}; "
                f"expected one of {sorted(MODE_FAMILIES)}")
        if dwell_distribution not in ("fixed", "exponential"):
            raise ValueError(
                "stochastic dwell distribution must be fixed or exponential")
        if fixed_mode_id is not None and not 0 <= int(fixed_mode_id) < 4:
            raise ValueError(
                f"stochastic fixed_mode_id must be in [0,3], got "
                f"{fixed_mode_id}")
        super().__init__(
            env_name, rand_params=["gravity"], log_scale_limit=0.0,
            seed=seed, backend=backend)
        self.family = str(family)
        self.dwell_steps = max(1, int(dwell_steps))
        self.dwell_distribution = str(dwell_distribution)
        self.fixed_mode_id = (
            None if fixed_mode_id is None else int(fixed_mode_id))
        self._switching_enabled = self.fixed_mode_id is None
        self.rollout_chunk_steps = self.dwell_steps
        self._mode_rng = np.random.RandomState(seed + 73_001)
        self._eval_mode_sequence = None
        self._eval_sequence_index = 0
        self._eval_sequence_calls = 0
        self._switch_history = [(0, 0)]
        self._tasks = self._build_tasks()
        self._build_mode_systems(self._tasks)
        self._activate_mode(
            0 if self.fixed_mode_id is None else self.fixed_mode_id)
        self._next_switch_step = self._sample_dwell()

    @property
    def num_modes(self) -> int:
        return len(self._tasks)

    def _build_tasks(self) -> List[Dict]:
        base_gravity = np.asarray(
            self._base_values["gravity"], dtype=np.float32)
        tasks = []
        for mode_id, profile in enumerate(MODE_FAMILIES[self.family]):
            action_gain = self._action_gain_for_profile(profile)
            tasks.append({
                "mode_id": int(mode_id),
                "gravity": base_gravity * float(profile["gravity_scale"]),
                "gravity_scale": float(profile["gravity_scale"]),
                "action_gain": action_gain,
                "action_noise_std": float(profile["action_noise_std"]),
                "packet_loss_prob": float(
                    profile.get("packet_loss_prob", 0.0)),
                "burst_prob": float(profile.get("burst_prob", 0.0)),
                "burst_std": float(profile.get("burst_std", 0.0)),
            })
        return tasks

    def _action_gain_for_profile(self, profile: Dict):
        pattern = profile.get("action_gain_pattern")
        nominal_gain = float(profile["action_gain"])
        if pattern is None:
            return nominal_gain
        impaired_gain = float(profile["impaired_gain"])
        gain = np.full((self.act_dim,), nominal_gain, dtype=np.float32)
        indices = np.arange(self.act_dim)
        if pattern == "low_half":
            affected = indices < self.act_dim // 2
        elif pattern == "high_half":
            affected = indices >= self.act_dim // 2
        elif pattern == "even":
            affected = indices % 2 == 0
        elif pattern == "odd":
            affected = indices % 2 == 1
        else:
            raise ValueError(f"unknown action_gain_pattern={pattern!r}")
        gain[affected] = impaired_gain
        return gain

    def _build_mode_systems(self, tasks) -> None:
        self._mode_sys = [
            self.base_sys.tree_replace({
                "gravity": jnp.asarray(task["gravity"], dtype=jnp.float32),
            })
            for task in tasks
        ]

    def _sample_dwell(self) -> int:
        if self.dwell_distribution == "fixed":
            return self.dwell_steps
        return max(1, int(self._mode_rng.exponential(self.dwell_steps)))

    def _sample_next_mode(self) -> int:
        if self._eval_mode_sequence:
            self._eval_sequence_index = (
                self._eval_sequence_index + 1
            ) % len(self._eval_mode_sequence)
            return int(self._eval_mode_sequence[self._eval_sequence_index])
        choices = [
            index for index in range(self.num_modes)
            if index != self.current_task_id
        ]
        return int(self._mode_rng.choice(choices))

    def _activate_mode(self, mode_id: int) -> None:
        mode_id = int(mode_id) % self.num_modes
        task = self._tasks[mode_id]
        self.current_task_id = mode_id
        self._set_sys(self._mode_sys[mode_id])
        self._set_action_disturbance(
            task["action_gain"], task["action_noise_std"],
            task.get("packet_loss_prob", 0.0),
            task.get("burst_prob", 0.0),
            task.get("burst_std", 0.0))

    def _check_switch(self):
        if not self._switching_enabled:
            return
        if self._step_counter < self._next_switch_step:
            return
        self._activate_mode(self._sample_next_mode())
        self._next_switch_step = self._step_counter + self._sample_dwell()
        self._switch_history.append(
            (int(self._step_counter), int(self.current_task_id)))

    def task_id_for_next_step(self) -> int:
        return int(self.current_task_id)

    def sample_tasks(self, n_tasks: int) -> List[Dict]:
        if int(n_tasks) == 0:
            return []
        if int(n_tasks) != self.num_modes:
            raise ValueError(
                f"stochastic_mode requires task_num={self.num_modes}, "
                f"got {n_tasks}")
        return [
            {
                key: (np.array(value, copy=True)
                      if isinstance(value, np.ndarray) else value)
                for key, value in task.items()
            }
            for task in self._tasks
        ]

    def set_task(self, task: Dict):
        mode_id = int(task.get("mode_id", 0))
        if not 0 <= mode_id < self.num_modes:
            raise ValueError(f"invalid stochastic mode_id={mode_id}")
        self._activate_mode(mode_id)

    def set_nonstationary_para(self, tasks, changing_period=None,
                                changing_interval=None):
        if len(tasks) != self.num_modes:
            raise ValueError(
                f"expected {self.num_modes} stochastic modes, got {len(tasks)}")
        self._tasks = list(tasks)
        self._build_mode_systems(self._tasks)
        self._eval_mode_sequence = None
        self._eval_sequence_index = 0
        self._switching_enabled = self.fixed_mode_id is None
        self._step_counter = 0
        initial_mode = (
            0 if self.fixed_mode_id is None else self.fixed_mode_id)
        self._switch_history = [(0, initial_mode)]
        self._activate_mode(initial_mode)
        self._next_switch_step = self._sample_dwell()

    def configure_eval_switching(self, tasks, period_steps: int):
        vectors = np.asarray([
            np.concatenate([
                np.asarray([task["gravity_scale"]], dtype=np.float64),
                np.asarray(task["action_gain"], dtype=np.float64).reshape(-1),
                np.asarray([
                    task["action_noise_std"],
                    task.get("packet_loss_prob", 0.0),
                    task.get("burst_prob", 0.0),
                    task.get("burst_std", 0.0),
                ], dtype=np.float64),
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

    def configure_eval_mode_sequence(
            self, tasks, mode_sequence, period_steps: int):
        """Install an explicit deterministic mode sequence for evaluation."""
        if len(tasks) != self.num_modes:
            raise ValueError(
                f"expected {self.num_modes} stochastic modes, got "
                f"{len(tasks)}")
        sequence = [int(mode) for mode in mode_sequence]
        if (len(sequence) < 2
                or any(not 0 <= mode < self.num_modes for mode in sequence)):
            raise ValueError(
                "eval mode sequence must contain at least two valid modes")
        self._tasks = list(tasks)
        self._build_mode_systems(self._tasks)
        self._eval_mode_sequence = sequence
        self._eval_sequence_index = 0
        self._switching_enabled = True
        self.dwell_steps = max(1, int(period_steps))
        self.rollout_chunk_steps = self.dwell_steps
        self.dwell_distribution = "fixed"
        self._step_counter = 0
        self._switch_history = [(0, sequence[0])]
        self._activate_mode(sequence[0])
        self._next_switch_step = self.dwell_steps

    def step(self, action):
        """Execute one stochastic transition, then advance the mode clock."""
        mode_used = int(self.current_task_id)
        gain_used = self._action_gain
        noise_std_used = self._action_noise_std
        packet_loss_prob_used = self._action_packet_loss_prob
        burst_prob_used = self._action_burst_prob
        burst_std_used = self._action_burst_std
        self.rng, noise_key = jax.random.split(self.rng)
        commanded_action = jnp.asarray(action, dtype=jnp.float32)
        executed_action = apply_action_disturbance(
            commanded_action, noise_key, gain_used, noise_std_used,
            packet_loss_prob_used, burst_prob_used, burst_std_used)
        self._state = self._step_fn(
            self._current_sys, self._state, executed_action)
        self._step_counter += 1
        self._check_switch()
        return (
            np.asarray(self._state.obs),
            float(self._state.reward),
            bool(self._state.done),
            {
                "mode_used": mode_used,
                "executed_action": np.asarray(executed_action),
                "action_gain": np.asarray(gain_used),
                "action_noise_std": np.asarray(noise_std_used),
                "packet_loss_prob": float(packet_loss_prob_used),
                "burst_prob": float(burst_prob_used),
                "burst_std": float(burst_std_used),
            },
        )

    def get_switch_history(self):
        return list(self._switch_history)
