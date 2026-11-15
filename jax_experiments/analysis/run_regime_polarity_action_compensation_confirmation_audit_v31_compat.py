"""Compatibility entry point for the frozen V31 audit implementation.

The preregistered reference publisher serializes policy parameters as an
``nnx.State``.  The frozen audit's loader accidentally restricted that valid
payload to ``dict``.  This entry point repairs only that serialization contract
while leaving the registered protocol, rollouts, arms, and decision gates
unchanged.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_action_compensation_confirmation_v31 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_confirmation_audit_v31 as frozen,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as estimator_model,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
)


_FROZEN_LOAD_POLICY_STACKS = frozen._load_policy_stacks
AMENDMENT_PATH = protocol.REGISTRATION_ROOT / "audit_implementation_amendment1.json"
AMENDMENT_SOURCE = Path(__file__).resolve()


def load_policy_tree(path: Path) -> dict[str, Any] | nnx.State:
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
        "canonical_reference"]
    policy_stack = dict(policy_stack)
    policy_stack["actions"] = actions
    return policy_stack, robust_long_stack, baselines


def amendment_payload() -> dict[str, Any]:
    return {
        "schema": "bapr.canonical-compensation-audit-amendment.v31.1",
        "status": "registered",
        "created_after_failed_execution_before_successful_audit": True,
        "parent_registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "failed_root_task_ids": {
            "nnx_state_loader": [
                "t93130", "t93136", "t93142", "t93148", "t93154"],
            "canonical_reference_alias": [
                "t93189", "t93190", "t93191", "t93192", "t93193"],
            "v5_estimator_binding": [
                "t93218", "t93219", "t93220", "t93221", "t93222"],
        },
        "corrections": [
            "Accept the nnx.State tree emitted by the registered reference publisher.",
            "Alias canonical_reference to the specialist_0 key expected by the frozen V28 helper.",
            "Bind the frozen helper to the registered v5 estimator implementation module.",
        ],
        "discarded_outputs": (
            "All failed attempts exited before writing an audit result or manifest."),
        "scientific_protocol_changes": [],
        "source_records": {
            "compatibility_runner": protocol.file_record(AMENDMENT_SOURCE),
            "estimator_implementation": protocol.file_record(
                Path(estimator_model.__file__).resolve()),
        },
    }


def create_amendment_registration() -> dict[str, Any]:
    protocol.validate_registration()
    payload = amendment_payload()
    if AMENDMENT_PATH.is_file():
        if protocol.read_json(AMENDMENT_PATH) != payload:
            raise ValueError("existing V31 audit amendment changed")
    else:
        protocol.write_json_atomic(AMENDMENT_PATH, payload)
    return payload


def install_runtime_bindings() -> None:
    frozen._load_pickle = load_policy_tree
    frozen._load_policy_stacks = load_policy_stacks
    frozen.compensation.v5_model = estimator_model


def annotate_audit(seed: int) -> None:
    result_path = protocol.audit_result(seed)
    manifest_path = protocol.audit_manifest(seed)
    payload = protocol.read_json(result_path)
    manifest = protocol.read_json(manifest_path)
    amendment = protocol.file_record(AMENDMENT_PATH)
    payload["execution_amendment"] = amendment
    protocol.write_json_atomic(result_path, payload)
    manifest["execution_amendment"] = amendment
    manifest["audit"] = protocol.file_record(result_path)
    protocol.write_json_atomic(manifest_path, manifest)
    frozen.validate_audit(seed)


def run(seed: int) -> None:
    _patch_flax_variablestate_unpickle()
    create_amendment_registration()
    install_runtime_bindings()
    seed = protocol.require_training_seed(seed)
    frozen.run(seed)
    annotate_audit(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for idempotent execution")
    run(args.seed)


if __name__ == "__main__":
    main()
