"""Audit the frozen mode-head student on one untouched event seed."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_confirmation_v2 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_audit as base_audit,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)


def _bind_confirmation() -> None:
    base_audit.protocol = protocol
    base_audit.model_lib = model_lib
    base_audit.trainer = trainer


_bind_confirmation()
robust_label = base_audit.robust_label
arm_labels = base_audit.arm_labels


def validate_audit(event_seed: int):
    protocol.validate_frozen_candidate()
    _bind_confirmation()
    return base_audit.validate_audit(
        protocol.TEACHER_GROUP,
        protocol.STUDENT_SEED,
        protocol.require_audit_event_seed(event_seed),
    )


def run(event_seed: int) -> None:
    protocol.validate_frozen_candidate()
    _bind_confirmation()
    base_audit.run(
        protocol.TEACHER_GROUP,
        protocol.STUDENT_SEED,
        protocol.require_audit_event_seed(event_seed),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-seed",
        type=int,
        choices=protocol.AUDIT_EVENT_SEEDS,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()

