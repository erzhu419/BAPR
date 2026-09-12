"""Run the unchanged trainer with the v19 critic-calibration SAC class."""
from __future__ import annotations

import os

from jax_experiments import train as train_module
from jax_experiments.algos.sac_actor_delay import SACActorDelay


def _make_algo(algo_name, obs_dim, act_dim, config):
    if algo_name != "sac":
        return _ORIGINAL_MAKE_ALGO(algo_name, obs_dim, act_dim, config)
    threshold = int(os.environ["BAPR_SAC_ACTOR_UPDATE_AFTER"])
    config.sac_actor_update_after = threshold
    return SACActorDelay(obs_dim, act_dim, config, seed=config.seed)


_ORIGINAL_MAKE_ALGO = train_module.make_algo


def main() -> None:
    if "BAPR_SAC_ACTOR_UPDATE_AFTER" not in os.environ:
        raise SystemExit("BAPR_SAC_ACTOR_UPDATE_AFTER is required")
    train_module.make_algo = _make_algo
    train_module.main()


if __name__ == "__main__":
    main()
