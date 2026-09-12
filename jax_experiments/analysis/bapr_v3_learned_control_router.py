"""Protocol helpers for the learned control-equivalence router."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import bapr_v3_control_equivalence as control


ROOT = control.ROOT
FAMILY = control.FAMILY
ENV = control.ENV
TRAIN_EVENT_SEEDS = (3100, 3200)
VALIDATION_EVENT_SEEDS = (4100, 4200)
HOLDOUT_EVENT_SEEDS = (5100, 5200, 5300, 5400, 5500)
BEHAVIOR_CONTROLLERS = (
    "robust", "fixed_mode_0", "fixed_mode_2", "fixed_mode_3")
MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_learned_router_v1")
MODEL_PATH = MODEL_ROOT / "router_params.npz"
MANIFEST_PATH = MODEL_ROOT / "router_manifest.json"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_learned_router_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_learned_router_analysis_v1")

MODEL_SCHEMA = "bapr.v3-control-equivalence-router.v1"
AUDIT_SCHEMA = "bapr.v3-control-equivalence-router-audit-group.v1"
ANALYSIS_SCHEMA = "bapr.v3-control-equivalence-router-analysis.v1"


@dataclass(frozen=True)
class RouterConfig:
    """Frozen filtering and robust-fallback decision parameters."""

    hazard_rate: float
    evidence_scale: float
    confidence_threshold: float
    margin_threshold: float
    min_history: int
    hysteresis_margin: float = 0.02
    posterior_decay: float = 1.0
    change_reset_threshold: float = 0.0
    change_reset_alpha: float = 0.25
    change_reset_mix: float = 1.0
    change_cusum_threshold: float = 0.0
    change_cusum_drift: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RouterConfig":
        payload = dict(value)
        # Manifests created before bounded-memory routing are exact decay=1
        # models. Keep those frozen artifacts loadable without rewriting them.
        payload.setdefault("posterior_decay", 1.0)
        payload.setdefault("change_reset_threshold", 0.0)
        payload.setdefault("change_reset_alpha", 0.25)
        payload.setdefault("change_reset_mix", 1.0)
        payload.setdefault("change_cusum_threshold", 0.0)
        payload.setdefault("change_cusum_drift", 0.0)
        return cls(**{
            field: payload[field]
            for field in cls.__dataclass_fields__
        })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure() -> None:
    control.configure()


def audit_group_path(event_seed: int) -> Path:
    return (
        AUDIT_ROOT / FAMILY / "HalfCheetah"
        / f"event_seed_{int(event_seed)}" / "group.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_parameter_state(path: Path, state: Any) -> list[dict[str, Any]]:
    """Save a fixed-architecture NNX parameter state without pickle."""
    leaves = [np.asarray(value) for value in jax.tree.leaves(state)]
    arrays = {f"leaf_{index:05d}": value
              for index, value in enumerate(leaves)}
    metadata = [
        {
            "key": key,
            "shape": list(arrays[key].shape),
            "dtype": str(arrays[key].dtype),
        }
        for key in sorted(arrays)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return metadata


def load_parameter_state(path: Path, template: Any,
                         metadata: list[dict[str, Any]]) -> Any:
    """Restore arrays into the current runtime's NNX state structure."""
    template_leaves, treedef = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        if len(keys) != len(template_leaves) or len(keys) != len(metadata):
            raise ValueError("router parameter leaf count changed")
        leaves = []
        for index, (key, expected, template_value) in enumerate(
                zip(keys, metadata, template_leaves)):
            if key != f"leaf_{index:05d}" or key != expected.get("key"):
                raise ValueError("router parameter ordering changed")
            value = np.asarray(archive[key])
            shape = tuple(int(item) for item in expected.get("shape", []))
            if (value.shape != shape
                    or value.shape != tuple(template_value.shape)
                    or str(value.dtype) != str(expected.get("dtype"))):
                raise ValueError(f"router parameter mismatch at {key}")
            leaves.append(jnp.asarray(value, dtype=template_value.dtype))
    return jax.tree.unflatten(treedef, leaves)


def aggregate_controller_probabilities(
    posterior: np.ndarray | jax.Array,
    controller_map: Iterable[int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Marginalize physical-mode belief into control-equivalence classes."""
    posterior_np = np.asarray(posterior, dtype=np.float64)
    mapping = tuple(int(value) for value in controller_map)
    if posterior_np.shape != (len(mapping),):
        raise ValueError(
            f"posterior shape {posterior_np.shape} does not match map {mapping}")
    if (not np.all(np.isfinite(posterior_np))
            or np.any(posterior_np < 0.0)
            or float(np.sum(posterior_np)) <= 0.0):
        raise ValueError("posterior must be finite, nonnegative, and nonempty")
    posterior_np = posterior_np / np.sum(posterior_np)
    controllers = tuple(sorted(set(mapping)))
    probabilities = np.asarray([
        np.sum(posterior_np[
            np.asarray(mapping, dtype=np.int32) == controller])
        for controller in controllers
    ], dtype=np.float64)
    return probabilities, controllers


def select_controller(
    posterior: np.ndarray | jax.Array,
    count: int,
    controller_map: Iterable[int],
    config: RouterConfig,
    previous_controller: int = -1,
) -> tuple[int, dict[str, float]]:
    """Choose a specialist or the robust fallback (`-1`)."""
    probabilities, controllers = aggregate_controller_probabilities(
        posterior, controller_map)
    order = np.argsort(-probabilities, kind="stable")
    top_index = int(order[0])
    top_controller = int(controllers[top_index])
    top_probability = float(probabilities[top_index])
    second_probability = float(probabilities[order[1]])
    margin = top_probability - second_probability

    if previous_controller in controllers:
        previous_index = controllers.index(int(previous_controller))
        previous_probability = float(probabilities[previous_index])
        if (previous_probability >= config.confidence_threshold
                and previous_probability + config.hysteresis_margin
                >= top_probability):
            top_controller = int(previous_controller)
            top_probability = previous_probability
            competitor = np.delete(probabilities, previous_index)
            second_probability = float(np.max(competitor))
            margin = top_probability - second_probability

    eligible = (
        int(count) >= int(config.min_history)
        and top_probability >= float(config.confidence_threshold)
        and margin >= float(config.margin_threshold)
    )
    selected = top_controller if eligible else -1
    return selected, {
        "confidence": top_probability,
        "margin": margin,
        "fallback": float(not eligible),
    }


def sticky_filter(log_likelihoods: np.ndarray, config: RouterConfig,
                  initial: np.ndarray | None = None) -> np.ndarray:
    """Run the causal sticky filter over precomputed emission evidence."""
    evidence = np.asarray(log_likelihoods, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] < 2:
        raise ValueError("log_likelihoods must have shape [time, modes>=2]")
    if not np.all(np.isfinite(evidence)):
        raise ValueError("log_likelihoods contain non-finite values")
    modes = evidence.shape[1]
    posterior = (
        np.full((modes,), 1.0 / modes, dtype=np.float64)
        if initial is None else np.asarray(initial, dtype=np.float64).copy())
    posterior /= np.sum(posterior)
    outputs = []
    switch_probability = config.hazard_rate / float(modes - 1)
    for row in evidence:
        prior = (
            (1.0 - config.hazard_rate) * posterior
            + switch_probability * (1.0 - posterior))
        logits = (
            config.posterior_decay * np.log(np.clip(prior, 1e-12, 1.0))
            + config.evidence_scale * (row - np.max(row)))
        logits -= np.max(logits)
        posterior = np.exp(logits)
        posterior /= np.sum(posterior)
        outputs.append(posterior.copy())
    return np.asarray(outputs, dtype=np.float64)


def route_trace(posteriors: np.ndarray, controller_map: Iterable[int],
                config: RouterConfig) -> tuple[np.ndarray, list[dict[str, float]]]:
    decisions = []
    diagnostics = []
    previous = -1
    for count, posterior in enumerate(np.asarray(posteriors), start=1):
        selected, row = select_controller(
            posterior, count, controller_map, config, previous)
        decisions.append(selected)
        diagnostics.append(row)
        previous = selected
    return np.asarray(decisions, dtype=np.int32), diagnostics


def causal_route_trace(
    log_likelihoods: np.ndarray,
    controller_map: Iterable[int],
    config: RouterConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Return action-time routes using evidence only from earlier transitions."""
    posteriors = sticky_filter(log_likelihoods, config)
    modes = posteriors.shape[1]
    before_action = np.concatenate([
        np.full((1, modes), 1.0 / modes, dtype=np.float64),
        posteriors[:-1],
    ], axis=0)
    decisions = []
    diagnostics = []
    previous = -1
    for count, posterior in enumerate(before_action):
        selected, row = select_controller(
            posterior, count, controller_map, config, previous)
        decisions.append(selected)
        diagnostics.append(row)
        previous = selected
    return (
        np.asarray(decisions, dtype=np.int32), posteriors, diagnostics)


def routing_metrics(decisions: np.ndarray, true_modes: np.ndarray,
                    controller_map: Iterable[int], stability: int = 8,
                    burnin: int = 32) -> dict[str, Any]:
    """Measure control-label accuracy and post-switch detection delay."""
    decisions = np.asarray(decisions, dtype=np.int32)
    true_modes = np.asarray(true_modes, dtype=np.int32)
    mapping = np.asarray(tuple(controller_map), dtype=np.int32)
    if decisions.shape != true_modes.shape or decisions.ndim != 1:
        raise ValueError("routing decisions and true_modes must be 1-D peers")
    expected = mapping[true_modes]
    index = np.arange(len(decisions)) >= int(burnin)
    adaptive = decisions >= 0
    correct = decisions == expected
    considered = index & adaptive
    coverage = float(np.mean(adaptive[index])) if np.any(index) else 0.0
    conditional_accuracy = (
        float(np.mean(correct[considered])) if np.any(considered) else 0.0)
    wrong_rate = (
        float(np.mean((adaptive & ~correct)[index])) if np.any(index) else 0.0)
    effective_accuracy = (
        float(np.mean(correct[index])) if np.any(index) else 0.0)

    switch_points = np.flatnonzero(true_modes[1:] != true_modes[:-1]) + 1
    delays = []
    for switch in switch_points:
        next_switches = switch_points[switch_points > switch]
        end = int(next_switches[0]) if len(next_switches) else len(decisions)
        wanted = int(expected[switch])
        delay = end - switch
        for start in range(switch, max(switch, end - stability + 1)):
            if np.all(decisions[start:start + stability] == wanted):
                delay = start - switch
                break
        delays.append(int(delay))
    return {
        "coverage": coverage,
        "conditional_accuracy": conditional_accuracy,
        "wrong_route_rate": wrong_rate,
        "effective_accuracy": effective_accuracy,
        "switch_count": int(len(switch_points)),
        "switch_delays": delays,
        "median_switch_delay": (
            float(np.median(delays)) if delays else 0.0),
    }


def load_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mapping, mapping_payload = control.load_controller_map()
    if (payload.get("schema") != MODEL_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != FAMILY
            or payload.get("env") != ENV
            or tuple(payload.get("controller_map", [])) != mapping
            or payload.get("controller_map_file")
            != file_record(control.MAPPING_PATH)
            or payload.get("controller_map_schema")
            != mapping_payload.get("schema")
            or payload.get("parameter_file") != file_record(MODEL_PATH)):
        raise ValueError("invalid or stale learned-router manifest")
    RouterConfig.from_dict(payload["router_config"])
    return payload
