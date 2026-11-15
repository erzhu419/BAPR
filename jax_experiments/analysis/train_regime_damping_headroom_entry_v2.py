"""Training entry accepting the registered persistent-damping family."""
from __future__ import annotations

import argparse

import jax_experiments.train as train
from jax_experiments.analysis import regime_damping_headroom_v2 as protocol
from jax_experiments.envs.persistent_damping_mode_env import (
    PersistentDampingModeEnv,
)


_ORIGINAL_ADD_ARGUMENT = argparse._ActionsContainer.add_argument


def _add_argument(self, *name_or_flags, **kwargs):
    if "--stochastic_mode_family" in name_or_flags:
        choices = list(kwargs.get("choices") or ())
        if protocol.FAMILY not in choices:
            choices.append(protocol.FAMILY)
        kwargs["choices"] = choices
    return _ORIGINAL_ADD_ARGUMENT(self, *name_or_flags, **kwargs)


argparse._ActionsContainer.add_argument = _add_argument
train.StochasticModeEnv = PersistentDampingModeEnv


if __name__ == "__main__":
    train.main()

