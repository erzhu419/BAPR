"""Run one strict event-matched audit in the fresh confirmation cohort."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_deployment_confirmation_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_audit_v2 as corrected_auditor,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_baseline_v2 as corrected_trainer,
)
from jax_experiments.analysis import (
    run_regime_polarity_fallback_final_audit_v1 as final_auditor,
)
from jax_experiments.analysis import (
    run_regime_polarity_fallback_final_baseline_v1 as sac_trainer,
)


def _bind() -> None:
    final_auditor.protocol = protocol
    final_auditor.baseline_runner.protocol = protocol
    final_auditor.model_lib.protocol = protocol
    sac_trainer.protocol = protocol
    corrected_auditor.protocol = protocol
    corrected_auditor.trainer.protocol = protocol
    corrected_trainer.protocol = protocol


def validate_manifest(method: str, seed: int):
    _bind()
    if method in ("bapr", "sac"):
        return final_auditor.validate_manifest(method, seed)
    return corrected_auditor.validate_manifest(method, seed)


def run(method: str, seed: int) -> None:
    method = protocol.require_method(method)
    seed = protocol.require_seed(seed)
    protocol.validate_mechanism_release()
    _bind()
    if method in ("bapr", "sac"):
        final_auditor.run(method, seed)
    else:
        corrected_auditor.run(method, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=protocol.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=protocol.TRAINING_SEEDS,
                        required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.method, args.seed)


if __name__ == "__main__":
    main()

