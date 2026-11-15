"""Strict CPU audit for one closed-loop compression v2 student."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_audit as base_audit,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)


def _bind_v2() -> None:
    base_audit.protocol = protocol
    base_audit.model_lib = model_lib
    base_audit.trainer = trainer


_bind_v2()
robust_label = base_audit.robust_label
arm_labels = base_audit.arm_labels


def validate_audit(variant: str, student_seed: int, event_seed: int):
    _bind_v2()
    return base_audit.validate_audit(variant, student_seed, event_seed)


def run(variant: str, student_seed: int, event_seed: int) -> None:
    _bind_v2()
    base_audit.run(variant, student_seed, event_seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--student-seed", type=int, choices=protocol.STUDENT_SEEDS,
        required=True)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.AUDIT_EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.variant, args.student_seed, args.event_seed)


if __name__ == "__main__":
    main()
