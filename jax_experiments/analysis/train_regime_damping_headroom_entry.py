"""Training entry that installs the isolated persistent-damping environment."""
from __future__ import annotations

import jax_experiments.train as train
from jax_experiments.envs.persistent_damping_mode_env import (
    PersistentDampingModeEnv,
)


train.StochasticModeEnv = PersistentDampingModeEnv


if __name__ == "__main__":
    train.main()

