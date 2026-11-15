"""Audit sticky specialist options on frozen controllers and estimator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_specialist_sticky_router_v3 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_audit_v1 as source_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_diagnostic_v2 as parent_audit,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def _posterior_metrics(rows, labels) -> dict[str, float] | None:
    if not rows:
        return None
    posterior = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    target = np.eye(len(protocol.MODES), dtype=np.float64)[labels]
    return {
        "mode_accuracy": float(np.mean(np.argmax(posterior, axis=1) == labels)),
        "brier_score": float(np.mean(np.sum((posterior - target) ** 2, axis=1))),
    }


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "sticky_specialist_router_screen",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "arms": list(protocol.ARMS),
        "primary_arm": protocol.PRIMARY_ARM,
    }


def _sticky_arm(stack, arm: str, event_seed: int) -> dict[str, Any]:
    confirmation_steps = protocol.STICKY_ARMS[protocol.require_arm(arm)]
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    router = protocol.StickySpecialistOption(
        actions["robust_sac"],
        tuple(actions[f"specialist_{mode}"] for mode in protocol.MODES),
        stack["estimator_factory"](),
        confirmation_steps,
    )
    returns = []
    terminated = []
    trace = []
    posterior_rows = []
    posterior_labels = []
    robust_actions = 0
    adaptive_correct = 0
    adaptive_actions = 0
    switch_counts = []
    total_actions = 0
    try:
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, config, protocol.DWELL_STEPS)
            observation = env.reset()
            router.reset()
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                posterior_rows.append(router.posterior.copy())
                posterior_labels.append(mode)
                step = router.select_action(observation)
                next_observation, reward, done, info = env.step(step.action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("sticky-router mode misaligned")
                router.observe_transition(
                    observation, step.action, reward, next_observation)
                robust_actions += int(step.source == "robust")
                if step.source == "specialist":
                    adaptive_correct += int(step.selected_mode == mode)
                    adaptive_actions += 1
                total_actions += 1
                episode_return += float(reward)
                observation = next_observation
                if done:
                    episode_terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminated.append(float(episode_terminated))
            switch_counts.append(router.switch_count)
    finally:
        if hasattr(env, "close"):
            env.close()

    return {
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminated)),
        "total_actions": int(total_actions),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "fallback_action_fraction": float(
            robust_actions / max(total_actions, 1)),
        "adaptive_mode_accuracy": float(
            adaptive_correct / max(adaptive_actions, 1)),
        "posterior_metrics": _posterior_metrics(
            posterior_rows, posterior_labels),
        "option_switch_count_mean": float(np.mean(switch_counts)),
        "switch_confirmation_steps": int(confirmation_steps),
    }


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    stack = source_audit._load_stack(seed)
    switching = {}
    for arm in protocol.ARMS:
        if arm in protocol.STICKY_ARMS:
            switching[arm] = _sticky_arm(stack, arm, event_seed)
        else:
            switching[arm] = parent_audit._switching_arm(
                stack, arm, event_seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "source_bundles": protocol.source_records(seed),
        "estimator": protocol.estimator_records(),
        "switching": switching,
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("source_bundles") != protocol.source_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid sticky-router event")
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        values = row.get("returns") or []
        if (
            len(values) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in values)
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        ):
            raise ValueError("invalid sticky-router result")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("sticky-router arms used different mode streams")
    if payload["switching"]["dynamic_oracle"][
        "adaptive_mode_accuracy"] != 1.0:
        raise ValueError("dynamic oracle used a wrong specialist")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected_files = {
        str(event_seed): protocol.file_record(
            protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    }
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "training_seed": seed,
            "event_seeds": list(protocol.EVENT_SEEDS),
        }
        or manifest.get("event_files") != expected_files
    ):
        raise ValueError("invalid sticky-router audit")
    for event_seed in protocol.EVENT_SEEDS:
        validate_event(
            protocol.read_json(protocol.event_result(seed, event_seed)),
            seed,
            event_seed,
        )
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"STICKY ROUTER ALREADY COMPLETE: {destination}")
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
        records = {}
        for event_seed in protocol.EVENT_SEEDS:
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(result, evaluate(seed, event_seed))
            records[str(event_seed)] = protocol.file_record(result)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "training_seed": seed,
                    "event_seeds": list(protocol.EVENT_SEEDS),
                },
                "event_files": records,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"STICKY ROUTER COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
