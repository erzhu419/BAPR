"""Train one v2 persistent-damping robust/oracle controller."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import regime_damping_headroom_v2 as protocol
from jax_experiments.analysis import (
    run_regime_control_headroom_controller as common,
)


common.protocol = protocol
_COMMON_TRAINING_COMMAND = common.training_command


def training_command(env: str, role: str, seed: int) -> list[str]:
    command = _COMMON_TRAINING_COMMAND(env, role, seed)
    module_index = command.index("jax_experiments.train")
    command[module_index] = (
        "jax_experiments.analysis.train_regime_damping_headroom_entry_v2")
    return command


common.training_command = training_command
validate_bundle = common.validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--role", choices=protocol.ROLES, required=True)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS,
                        type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    protocol.validate_registration()
    common.run(args.env, args.role, args.seed)


if __name__ == "__main__":
    main()

