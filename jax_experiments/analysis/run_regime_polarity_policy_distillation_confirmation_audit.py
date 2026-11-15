"""Audit the frozen selected distillation student on one new event seed."""
from __future__ import annotations

import argparse
from contextlib import contextmanager

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_confirmation as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_audit as base,
)


@contextmanager
def _confirmation_protocol():
    original = base.protocol
    base.protocol = protocol
    try:
        yield
    finally:
        base.protocol = original


def validate_audit(event_seed: int):
    protocol.validate_frozen_candidate()
    with _confirmation_protocol():
        return base.validate_audit(
            protocol.TEACHER_GROUP,
            protocol.STUDENT_SEED,
            protocol.require_audit_event_seed(event_seed),
        )


def run(event_seed: int) -> None:
    protocol.validate_frozen_candidate()
    with _confirmation_protocol():
        base.run(
            protocol.TEACHER_GROUP,
            protocol.STUDENT_SEED,
            protocol.require_audit_event_seed(event_seed),
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.AUDIT_EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
