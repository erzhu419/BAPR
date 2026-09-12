"""Train one V27 robust or privileged-oracle controller."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_channel_survival_headroom_v27 as protocol,
)
from jax_experiments.analysis import (
    run_regime_control_headroom_controller as common,
)


def _bind() -> None:
    common.protocol = protocol


def training_command(env: str, role: str, seed: int):
    _bind()
    return common.training_command(env, role, seed)


def run(env: str, role: str, seed: int) -> None:
    protocol.validate_registration()
    _bind()
    common.run(env, role, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--role", choices=protocol.ROLES, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.env, args.role, args.seed)


if __name__ == "__main__":
    main()
