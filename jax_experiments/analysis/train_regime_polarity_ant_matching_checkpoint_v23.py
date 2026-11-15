"""Run policy-stability SAC with matching-fixed-mode model selection."""
from __future__ import annotations

import os

from jax_experiments import train as train_module
from jax_experiments.algos.sac_policy_stability import SACPolicyStability


_ORIGINAL_MAKE_ALGO = train_module.make_algo
_ORIGINAL_EVALUATE_STATIONARY = train_module.evaluate_stationary


def select_matching_task(tasks, fixed_mode: int):
    matches = [
        task for task in list(tasks or [])
        if int(task.get("mode_id", -1)) == int(fixed_mode)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one validation task for mode {fixed_mode}, "
            f"found {len(matches)}")
    return matches


def _make_algo(algo_name, obs_dim, act_dim, config):
    if algo_name != "sac":
        return _ORIGINAL_MAKE_ALGO(algo_name, obs_dim, act_dim, config)
    config.sac_actor_update_period = int(
        os.environ["BAPR_SAC_ACTOR_UPDATE_PERIOD"])
    config.sac_select_best_eval = True
    return SACPolicyStability(obs_dim, act_dim, config, seed=config.seed)


def _evaluate_matching(
    agent,
    env,
    config,
    tasks=None,
    n_episodes=10,
    context_source=None,
    advantage_enabled=None,
    record_diagnostics=True,
):
    selected = select_matching_task(
        tasks, int(config.stochastic_mode_fixed_id))
    return _ORIGINAL_EVALUATE_STATIONARY(
        agent,
        env,
        config,
        selected,
        n_episodes=n_episodes,
        context_source=context_source,
        advantage_enabled=advantage_enabled,
        record_diagnostics=record_diagnostics,
    )


def main() -> None:
    if "BAPR_SAC_ACTOR_UPDATE_PERIOD" not in os.environ:
        raise SystemExit("missing BAPR_SAC_ACTOR_UPDATE_PERIOD")
    if os.environ.get("BAPR_SAC_SELECT_BEST_EVAL") != "1":
        raise SystemExit("Ant v23 requires validation checkpoint selection")
    train_module.make_algo = _make_algo
    train_module.evaluate_stationary = _evaluate_matching
    train_module.main()


if __name__ == "__main__":
    main()
