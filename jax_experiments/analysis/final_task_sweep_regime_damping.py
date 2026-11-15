"""Strict sweep entry that installs the persistent-damping environment."""
from __future__ import annotations

import jax_experiments.train as train
from jax_experiments.envs.persistent_damping_mode_env import (
    PersistentDampingModeEnv,
)


train.StochasticModeEnv = PersistentDampingModeEnv

from jax_experiments.analysis import final_task_sweep


if __name__ == "__main__":
    final_task_sweep.main()

