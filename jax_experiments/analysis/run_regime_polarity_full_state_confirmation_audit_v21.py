"""Strict V21 audit of full-state BAPR and matched baselines."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_full_state_specialist_v21 as specialist_protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as v5_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_baseline_v21 as trainer,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_specialist_v21 as specialist_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_stability_audit_v19 as bank_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18 as base,
)


def _load_frozen_controllers(seed: int):
    specialist_runner._bind()
    bank_audit.protocol = specialist_protocol
    bank_audit.producer = specialist_runner
    v5_model.load_model(17, 6)
    return bank_audit._load_controllers(
        specialist_protocol.SPECIALIST_VARIANT, seed)


def _bind() -> None:
    base.protocol = protocol
    base.trainer = trainer
    base._load_frozen_controllers = _load_frozen_controllers
    trainer._bind()


def evaluate(seed: int):
    _bind()
    return base.evaluate(seed)


def validate_result(payload, seed: int) -> None:
    _bind()
    base.validate_result(payload, seed)


def validate_audit(seed: int):
    _bind()
    return base.validate_audit(seed)


def run(seed: int) -> None:
    _bind()
    base.run(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed)


if __name__ == "__main__":
    main()
