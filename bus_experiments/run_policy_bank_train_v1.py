"""Train one registered bus robust source or fixed-mode specialist."""
from __future__ import annotations

import argparse
from pathlib import Path

from bus_experiments import policy_bank_headroom_v1 as protocol
from bus_experiments.frozen_policy_bank_core import TrainConfig, train


def build_config(role: str, seed: int, mode: str | None) -> TrainConfig:
    if role == "robust_source":
        return TrainConfig(
            seed=seed,
            max_episodes=protocol.SOURCE_EPISODES,
            role=role,
            checkpoint_interval=protocol.CHECKPOINT_INTERVAL,
        )
    return TrainConfig(
        seed=seed,
        max_episodes=protocol.SPECIALIST_EPISODES,
        role=role,
        fixed_mode=protocol.require_mode(str(mode)),
        checkpoint_interval=protocol.CHECKPOINT_INTERVAL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("robust_source", "specialist"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=protocol.MODES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol.validate_registration()
    seed = protocol.require_seed(args.seed)
    if args.role == "robust_source":
        if args.mode is not None:
            parser.error("robust_source does not accept --mode")
        run_dir = protocol.source_run_dir(seed)
        bundle_dir = protocol.source_bundle_dir(seed)
        warmstart = None
    else:
        if args.mode is None:
            parser.error("specialist requires --mode")
        mode = protocol.require_mode(args.mode)
        run_dir = protocol.specialist_run_dir(seed, mode)
        bundle_dir = protocol.specialist_bundle_dir(seed, mode)
        warmstart = protocol.source_controller(seed)
        if not warmstart.is_file():
            raise FileNotFoundError(f"missing robust warmstart: {warmstart}")

    config = build_config(args.role, seed, args.mode)
    train(
        config, Path(run_dir), Path(bundle_dir), resume=bool(args.resume),
        warmstart_path=Path(warmstart) if warmstart else None)


if __name__ == "__main__":
    main()
