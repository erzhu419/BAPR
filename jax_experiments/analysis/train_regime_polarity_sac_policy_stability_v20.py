"""Run the common trainer with the v20 policy-stability SAC class."""
from __future__ import annotations

import os

from jax_experiments import train as train_module
from jax_experiments.algos.sac_policy_stability import SACPolicyStability


def _make_algo(algo_name, obs_dim, act_dim, config):
    if algo_name != "sac":
        return _ORIGINAL_MAKE_ALGO(algo_name, obs_dim, act_dim, config)
    config.sac_actor_update_period = int(
        os.environ["BAPR_SAC_ACTOR_UPDATE_PERIOD"])
    config.sac_select_best_eval = (
        os.environ["BAPR_SAC_SELECT_BEST_EVAL"] == "1")
    return SACPolicyStability(obs_dim, act_dim, config, seed=config.seed)


_ORIGINAL_MAKE_ALGO = train_module.make_algo


def main() -> None:
    missing = [
        name for name in (
            "BAPR_SAC_ACTOR_UPDATE_PERIOD",
            "BAPR_SAC_SELECT_BEST_EVAL",
        )
        if name not in os.environ
    ]
    if missing:
        raise SystemExit("missing v20 environment: " + ",".join(missing))
    train_module.make_algo = _make_algo
    train_module.main()


if __name__ == "__main__":
    main()
