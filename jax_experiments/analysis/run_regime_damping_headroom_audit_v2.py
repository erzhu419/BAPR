"""Run paired strict audits for one v2 persistent-damping checkpoint."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import regime_damping_headroom_v2 as protocol
from jax_experiments.analysis import (
    run_regime_control_headroom_audit as common,
)
from jax_experiments.analysis import (
    run_regime_damping_headroom_controller_v2 as controller,
)


controller.common.protocol = protocol
common.protocol = protocol
common.validate_bundle = controller.validate_bundle
_COMMON_EVALUATION_COMMAND = common._evaluation_command


def evaluation_command(
        env: str, role: str, seed: int, event_seed: int,
        output) -> list[str]:
    command = _COMMON_EVALUATION_COMMAND(
        env, role, seed, event_seed, output)
    module_index = command.index(
        "jax_experiments.analysis.final_task_sweep")
    command[module_index] = (
        "jax_experiments.analysis.final_task_sweep_regime_damping")
    return command


common._evaluation_command = evaluation_command
validate_audit = common.validate_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--role", choices=protocol.ROLES, required=True)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS,
                        type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    protocol.validate_registration()
    common.run(args.env, args.role, args.seed)


if __name__ == "__main__":
    main()

