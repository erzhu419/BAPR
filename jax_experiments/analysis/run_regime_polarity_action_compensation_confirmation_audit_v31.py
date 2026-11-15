"""Run the fresh-policy V31 canonical-compensation audit."""
from __future__ import annotations

import argparse
import copy
import math
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_action_compensation_confirmation_v31 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_action_compensation_reference_v31 as reference_protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_audit_v28 as compensation,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_baseline_v31 as baseline_trainer,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_reference_v31 as reference_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as stack_lib,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18 as baseline_eval,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
)


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _bind() -> None:
    reference_runner._bind()
    baseline_trainer._bind()
    baseline_eval.protocol = protocol
    baseline_eval.trainer = baseline_trainer
    compensation.protocol = protocol
    compensation.base = baseline_eval


def _load_pickle(path: Path) -> dict[str, Any]:
    _patch_flax_variablestate_unpickle()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected parameter dictionary: {path}")
    return payload


def _load_policy_stacks(seed: int):
    _bind()
    seed = protocol.require_training_seed(seed)
    config, source_agent = reference_runner._load_source(seed)
    source_params = nnx.state(source_agent.policy, nnx.Param)
    robust = {
        "config": config,
        "agent": source_agent,
        "policy_graphdef": nnx.graphdef(source_agent.policy),
        "policy_params": source_params,
        "context_graphdef": None,
        "context_params": None,
    }
    reference_runner.validate_bundle(
        reference_protocol.SPECIALIST_VARIANT,
        seed,
        protocol.REFERENCE_MODE,
    )
    raw = _load_pickle(
        protocol.reference_bundle_dir(seed) / "policy" / protocol.POLICY_NAME)
    reference = {
        "config": copy.deepcopy(config),
        "agent": source_agent,
        "policy_graphdef": nnx.graphdef(source_agent.policy),
        "policy_params": _restore_tree_like(
            source_params,
            raw,
            "V31 canonical reference policy",
            allow_fallback=False,
        ),
        "context_graphdef": None,
        "context_params": None,
    }
    policy_stack = stack_lib._stack_from_controllers({
        "robust_sac": robust,
        "canonical_reference": reference,
    })
    policy_stack["reference_agent"] = source_agent

    robust_long = baseline_eval._load_sac_replica(
        seed, protocol.LONG_SAC_SLOT, robust)
    robust_long_stack = stack_lib._stack_from_controllers({
        "robust_sac": robust_long,
    })
    robust_long_stack["reference_agent"] = robust_long["agent"]
    baselines = {
        method: baseline_eval._load_baseline_runtime(method, seed)
        for method in protocol.TRAINED_METHODS
    }
    return policy_stack, robust_long_stack, baselines


def _stationary_holdout(
    policy_stack: dict[str, Any],
    robust_long_stack: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    events = {}
    actions = policy_stack["actions"]
    long_action = robust_long_stack["actions"]["robust_sac"]
    for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS:
        arms = {arm: {} for arm in protocol.ARMS}
        for mode in protocol.MODES:
            key = str(mode)
            arms[protocol.ROBUST_SOURCE_ARM][key] = (
                baseline_eval._stationary_direct(
                    policy_stack["config"], actions["robust_sac"],
                    event_seed, mode, protocol.STATIONARY_EPISODES_PER_MODE))
            arms[protocol.ROBUST_LONG_ARM][key] = (
                baseline_eval._stationary_direct(
                    robust_long_stack["config"], long_action,
                    event_seed, mode, protocol.STATIONARY_EPISODES_PER_MODE))
            arms[protocol.NO_COMPENSATION_ARM][key] = (
                baseline_eval._stationary_direct(
                    policy_stack["config"], actions["canonical_reference"],
                    event_seed, mode, protocol.STATIONARY_EPISODES_PER_MODE))
            arms[protocol.ORACLE_COMPENSATION_ARM][key] = (
                compensation._compensated_stationary(
                    policy_stack, protocol.REFERENCE_MODE, "true_mode",
                    event_seed, mode))
            arms[protocol.CAUSAL_COMPENSATION_ARM][key] = (
                compensation._compensated_stationary(
                    policy_stack, protocol.REFERENCE_MODE, "v5_posterior",
                    event_seed, mode))
            escp_config, escp_runtime = baselines["escp_recurrent"]
            arms[protocol.ESCP_ARM][key] = baseline_eval._stationary_runtime(
                escp_config, escp_runtime, event_seed, mode)
            resac_config, resac_runtime = baselines["resac_b0"]
            arms[protocol.RESAC_ARM][key] = baseline_eval._stationary_runtime(
                resac_config, resac_runtime, event_seed, mode)
        events[str(event_seed)] = arms
        print(f"V31 stationary event={event_seed} complete", flush=True)
    return events


def _switching_holdout(
    policy_stack: dict[str, Any],
    robust_long_stack: dict[str, Any],
    baselines: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = {}
    equivalence = {}
    event_hashes = set()
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        escp_config, escp_runtime = baselines["escp_recurrent"]
        resac_config, resac_runtime = baselines["resac_b0"]
        rows = {
            protocol.ROBUST_SOURCE_ARM: baseline_eval._switching_bank(
                policy_stack, {}, "fixed", event_seed,
                fixed_controller="robust_sac"),
            protocol.ROBUST_LONG_ARM: baseline_eval._switching_bank(
                robust_long_stack, {}, "fixed", event_seed,
                fixed_controller="robust_sac"),
            protocol.NO_COMPENSATION_ARM: baseline_eval._switching_bank(
                policy_stack, {}, "fixed", event_seed,
                fixed_controller="canonical_reference"),
            protocol.ORACLE_COMPENSATION_ARM: (
                compensation._compensated_switching(
                    policy_stack, protocol.REFERENCE_MODE, "true_mode",
                    event_seed)),
            protocol.CAUSAL_COMPENSATION_ARM: (
                compensation._compensated_switching(
                    policy_stack, protocol.REFERENCE_MODE, "v5_posterior",
                    event_seed)),
            protocol.ESCP_ARM: baseline_eval._switching_runtime(
                escp_config, escp_runtime, event_seed),
            protocol.RESAC_ARM: baseline_eval._switching_runtime(
                resac_config, resac_runtime, event_seed),
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("V31 switching arms used different mode streams")
        event_hashes.update(hashes)
        native = baseline_eval._stationary_direct(
            policy_stack["config"],
            policy_stack["actions"]["canonical_reference"],
            event_seed,
            protocol.REFERENCE_MODE,
            protocol.SWITCHING_EPISODES,
        )
        oracle_returns = np.asarray(
            rows[protocol.ORACLE_COMPENSATION_ARM]["returns"],
            dtype=np.float64,
        )
        native_returns = np.asarray(native["returns"], dtype=np.float64)
        max_error = float(np.max(np.abs(oracle_returns - native_returns)))
        equivalence[str(event_seed)] = {
            "native_reference_mode": protocol.REFERENCE_MODE,
            "native_returns": native["returns"],
            "oracle_compensation_returns": (
                rows[protocol.ORACLE_COMPENSATION_ARM]["returns"]),
            "max_abs_return_error": max_error,
            "pass": bool(max_error <= protocol.EXACT_RETURN_ATOL),
        }
        events[str(event_seed)] = rows
        print(f"V31 switching event={event_seed} complete", flush=True)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("V31 switching streams are not distinct")
    return events, equivalence


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "fresh_policy_canonical_compensation_confirmation",
        "training_seed": protocol.require_training_seed(seed),
        "reference_mode": protocol.REFERENCE_MODE,
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
        "arms": list(protocol.ARMS),
    }


def evaluate(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    policy_stack, robust_long_stack, baselines = _load_policy_stacks(seed)
    switching, equivalence = _switching_holdout(
        policy_stack, robust_long_stack, baselines)
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "frozen_inputs": protocol.frozen_input_records(seed),
        "stationary_holdout": _stationary_holdout(
            policy_stack, robust_long_stack, baselines),
        "switching_holdout": switching,
        "oracle_equivalence": equivalence,
    }


def validate_result(payload: dict[str, Any], seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    if (
        payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("frozen_inputs")
        != protocol.frozen_input_records(seed)
    ):
        raise ValueError("invalid V31 audit identity")
    stationary = payload.get("stationary_holdout") or {}
    if set(stationary) != {
        str(value) for value in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
    }:
        raise ValueError("incomplete V31 stationary holdout")
    for event in stationary.values():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V31 stationary arms")
        for arm in protocol.ARMS:
            if set(event[arm]) != {str(mode) for mode in protocol.MODES}:
                raise ValueError("incomplete V31 stationary modes")
            for row in event[arm].values():
                returns = row.get("returns") or []
                if (
                    len(returns) != protocol.STATIONARY_EPISODES_PER_MODE
                    or not all(math.isfinite(float(value)) for value in returns)
                    or int(row.get("total_actions", -1))
                    != (protocol.STATIONARY_EPISODES_PER_MODE
                        * protocol.MAX_EPISODE_STEPS)
                ):
                    raise ValueError("invalid V31 stationary rollout")

    switching = payload.get("switching_holdout") or {}
    equivalence = payload.get("oracle_equivalence") or {}
    expected_events = {str(value) for value in protocol.SWITCHING_EVENT_SEEDS}
    if set(switching) != expected_events or set(equivalence) != expected_events:
        raise ValueError("incomplete V31 switching holdout")
    expected_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    event_hashes = set()
    for event_seed, event in switching.items():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V31 switching arms")
        hashes = set()
        for row in event.values():
            returns = row.get("returns") or []
            if (
                len(returns) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value)) for value in returns)
                or int(row.get("total_actions", -1))
                != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
                or row.get("mode_counts") != {
                    str(mode): expected_count for mode in protocol.MODES}
                or row.get("base_schedule")
                != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            ):
                raise ValueError("invalid V31 switching rollout")
            hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(hashes) != 1 or "" in hashes:
            raise ValueError("V31 switching arms differ in event stream")
        event_hashes.update(hashes)
        oracle = event[protocol.ORACLE_COMPENSATION_ARM]
        causal = event[protocol.CAUSAL_COMPENSATION_ARM]
        if (
            oracle.get("routing_mode_accuracy") != 1.0
            or float(oracle.get("max_abs_execution_signal_error", math.inf))
            > protocol.EXACT_ACTION_ATOL
            or "posterior_metrics" not in causal
        ):
            raise ValueError("invalid V31 compensation diagnostics")
        if (
            equivalence[event_seed].get("pass") is not True
            or float(equivalence[event_seed].get(
                "max_abs_return_error", math.inf))
            > protocol.EXACT_RETURN_ATOL
        ):
            raise ValueError("V31 oracle trajectory equivalence failed")
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("V31 switching streams are not distinct")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    payload = protocol.read_json(protocol.audit_result(seed))
    validate_result(payload, seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != _identity(seed)
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("frozen_inputs")
        != protocol.frozen_input_records(seed)
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(seed))
    ):
        raise ValueError("invalid V31 audit manifest")
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V31 AUDIT ALREADY COMPLETE: seed={seed}", flush=True)
            return
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        payload = evaluate(seed)
        validate_result(payload, seed)
        protocol.write_json_atomic(temporary / "audit.json", payload)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": _identity(seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "frozen_inputs": protocol.frozen_input_records(seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V31 AUDIT COMPLETE: seed={seed}", flush=True)


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
