"""Run the preregistered V32 canonical-compensation power audit."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_action_compensation_reference_v32 as reference_protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as estimator_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_baseline_v32 as baseline_trainer,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_confirmation_audit_v31 as frozen,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_reference_v32 as reference_runner,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
)


_FROZEN_LOAD_POLICY_STACKS = frozen._load_policy_stacks


def load_policy_tree(path: Path) -> dict[str, Any] | nnx.State:
    """Load either supported Flax parameter-tree representation."""
    _patch_flax_variablestate_unpickle()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (dict, nnx.State)):
        raise ValueError(f"expected policy parameter tree: {path}")
    return payload


def load_policy_stacks(seed: int):
    policy_stack, robust_long_stack, baselines = _FROZEN_LOAD_POLICY_STACKS(seed)
    actions = dict(policy_stack["actions"])
    actions[f"specialist_{protocol.REFERENCE_MODE}"] = actions[
        "canonical_reference"
    ]
    policy_stack = dict(policy_stack)
    policy_stack["actions"] = actions
    return policy_stack, robust_long_stack, baselines


def identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "power_confirmation_canonical_compensation",
        "training_seed": protocol.require_training_seed(seed),
        "reference_mode": protocol.REFERENCE_MODE,
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
        ),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
        "arms": list(protocol.ARMS),
    }


def install_runtime_bindings() -> None:
    frozen.protocol = protocol
    frozen.reference_protocol = reference_protocol
    frozen.reference_runner = reference_runner
    frozen.baseline_trainer = baseline_trainer
    frozen.baseline_eval.protocol = protocol
    frozen.baseline_eval.trainer = baseline_trainer
    frozen.compensation.protocol = protocol
    frozen.compensation.base = frozen.baseline_eval
    frozen.compensation.v5_model = estimator_model
    frozen._load_pickle = load_policy_tree
    frozen._load_policy_stacks = load_policy_stacks
    frozen._identity = identity


def validate_audit(seed: int) -> dict[str, Any]:
    install_runtime_bindings()
    return frozen.validate_audit(seed)


def run(seed: int) -> None:
    _patch_flax_variablestate_unpickle()
    protocol.validate_registration()
    install_runtime_bindings()
    frozen.run(protocol.require_training_seed(seed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for idempotent execution")
    run(args.seed)


if __name__ == "__main__":
    main()
