"""Protocol and model for a nonlinear posterior-conditioned residual policy."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import bapr_v3_posterior_residual as screen


ROOT = screen.ROOT
FAMILY = screen.FAMILY
ENV = screen.ENV
DECISION_VARIANT = screen.DECISION_VARIANT

TRAIN_EVENT_SEEDS = estimator.TRAIN_EVENT_SEEDS
MODEL_SELECTION_EVENT_SEEDS = estimator.VALIDATION_EVENT_SEEDS
DEVELOPMENT_RETURN_EVENT_SEEDS = screen.DEVELOPMENT_EVENT_SEEDS
SEALED_CONFIRMATION_EVENT_SEEDS = screen.SEALED_CONFIRMATION_EVENT_SEEDS

BEHAVIOR_CONTROLLERS = (
    "robust", "dynamic_utility_oracle", "learned_utility_router")
FULL_CYCLE_SEQUENCES = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (0, 2, 1, 3),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
)
FULL_CYCLE_DWELL_STEPS = 250

MODEL_CONFIG = {
    "hidden_dim": 128,
    "residual_bound": 2.0,
    "learning_rate": 3e-4,
    "batch_size": 512,
    "initial_updates": 8000,
    "dagger_updates": 4000,
    "initialization_seeds": [0, 1, 2],
    "teacher_loss_weight": 1.0,
    "entropy_anchor_weight": 1.0,
    "dagger_rounds": 1,
}

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_posterior_residual_policy_v1"
)
MODEL_PATH = MODEL_ROOT / "policy_params.npz"
NORMALIZATION_PATH = MODEL_ROOT / "normalization.npz"
MANIFEST_PATH = MODEL_ROOT / "policy_manifest.json"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_posterior_residual_policy_audit_v1"
)
ANALYSIS_ROOT = AUDIT_ROOT / "analysis"
ANALYSIS_JSON = ANALYSIS_ROOT / "summary.json"
ANALYSIS_REPORT = ANALYSIS_ROOT / "report.md"

MODEL_SCHEMA = "bapr.v3-posterior-residual-policy.v1"
GROUP_SCHEMA = "bapr.v3-posterior-residual-policy-audit-group.v1"
ANALYSIS_SCHEMA = "bapr.v3-posterior-residual-policy-analysis.v1"


def configure() -> None:
    screen.configure()


def audit_group_path(event_seed: int) -> Path:
    if int(event_seed) not in DEVELOPMENT_RETURN_EVENT_SEEDS:
        raise ValueError(f"unregistered nonlinear residual seed {event_seed}")
    return AUDIT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def file_record(path: Path):
    return estimator.file_record(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    estimator.write_json_atomic(path, payload)


def validate_source_records(records: dict[str, Any]) -> None:
    if not records:
        raise ValueError("residual policy manifest has no source records")
    for relative, expected in records.items():
        path = ROOT / relative
        if not path.is_file() or file_record(path) != expected:
            raise ValueError(f"residual policy source changed: {relative}")


def utility_constants(table: dict[str, Any]):
    matrix, controllers = screen.utility.utility_matrix(table)
    robust_index = controllers.index(screen.utility.ROBUST_CONTROLLER)
    specialist_indices = tuple(
        index for index, controller in enumerate(controllers)
        if controller != screen.utility.ROBUST_CONTROLLER)
    if not specialist_indices:
        raise ValueError("utility table has no specialist controller")
    return (
        np.asarray(matrix, dtype=np.float32),
        int(robust_index),
        specialist_indices,
        float(screen.residual_advantage_scale(table)),
    )


def posterior_strength(
    posterior,
    utility_matrix,
    robust_index: int,
    specialist_indices: Iterable[int],
    advantage_scale: float,
):
    """Continuous utility support for adaptation, with no threshold."""
    probabilities = jnp.clip(jnp.asarray(posterior), 0.0)
    probabilities = probabilities / jnp.maximum(
        jnp.sum(probabilities, axis=-1, keepdims=True), 1e-8)
    expected = probabilities @ jnp.asarray(utility_matrix)
    robust = expected[..., int(robust_index)]
    specialist = jnp.max(
        jnp.take(
            expected,
            jnp.asarray(tuple(specialist_indices), dtype=jnp.int32),
            axis=-1),
        axis=-1,
    )
    return jnp.clip(
        (specialist - robust) / float(advantage_scale), 0.0, 1.0)


def normalized_entropy(posterior):
    probabilities = jnp.clip(jnp.asarray(posterior), 1e-8, 1.0)
    probabilities = probabilities / jnp.sum(
        probabilities, axis=-1, keepdims=True)
    return -jnp.sum(
        probabilities * jnp.log(probabilities), axis=-1) / jnp.log(4.0)


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


class PosteriorResidualPolicy(nnx.Module):
    """Nonlinear action residual anchored exactly to a frozen robust action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 residual_bound: float, *, rngs: nnx.Rngs):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.residual_bound = float(residual_bound)
        layers = []
        width = self.obs_dim + 4 + self.act_dim
        for _ in range(2):
            layers.append(nnx.Linear(width, hidden_dim, rngs=rngs))
            width = hidden_dim
        self.layers = _module_list(layers)
        self.output = nnx.Linear(width, self.act_dim, rngs=rngs)

    def __call__(self, obs, posterior, robust_action, strength,
                 obs_mean, obs_std):
        normalized_obs = jnp.clip(
            (obs - obs_mean) / jnp.maximum(obs_std, 1e-4), -10.0, 10.0)
        hidden = jnp.concatenate(
            [normalized_obs, posterior, robust_action], axis=-1)
        for layer in self.layers:
            hidden = jax.nn.silu(layer(hidden))
        residual = self.residual_bound * jnp.tanh(self.output(hidden))
        return jnp.clip(
            robust_action + strength[..., None] * residual, -1.0, 1.0)


def make_policy(obs_dim: int, act_dim: int, seed: int):
    return PosteriorResidualPolicy(
        obs_dim,
        act_dim,
        hidden_dim=int(MODEL_CONFIG["hidden_dim"]),
        residual_bound=float(MODEL_CONFIG["residual_bound"]),
        rngs=nnx.Rngs(int(seed)),
    )


def save_normalization(path: Path, obs_mean, obs_std) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                obs_mean=np.asarray(obs_mean, dtype=np.float32),
                obs_std=np.asarray(obs_std, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_normalization(path: Path, obs_dim: int):
    with np.load(path, allow_pickle=False) as archive:
        mean = np.asarray(archive["obs_mean"], dtype=np.float32)
        std = np.asarray(archive["obs_std"], dtype=np.float32)
    if (mean.shape != (int(obs_dim),)
            or std.shape != (int(obs_dim),)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std))
            or np.any(std <= 0.0)):
        raise ValueError("invalid posterior residual normalization")
    return mean, std


def load_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (payload.get("schema") != MODEL_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != FAMILY
            or payload.get("env") != ENV
            or payload.get("decision_variant") != DECISION_VARIANT
            or payload.get("model_config") != MODEL_CONFIG
            or tuple(payload.get("train_event_seeds", []))
            != TRAIN_EVENT_SEEDS
            or tuple(payload.get("model_selection_event_seeds", []))
            != MODEL_SELECTION_EVENT_SEEDS
            or payload.get("parameter_file") != file_record(MODEL_PATH)
            or payload.get("normalization_file")
            != file_record(NORMALIZATION_PATH)):
        raise ValueError(f"invalid nonlinear residual manifest: {MANIFEST_PATH}")
    obs_dim = int(payload.get("obs_dim", -1))
    act_dim = int(payload.get("act_dim", -1))
    if obs_dim <= 0 or act_dim <= 0:
        raise ValueError("invalid nonlinear residual dimensions")
    validate_source_records(payload.get("source_files") or {})
    if (payload.get("utility_table_file")
            != file_record(screen.utility.TABLE_PATH)
            or payload.get("estimator_manifest_file")
            != file_record(estimator.MANIFEST_PATH)
            or payload.get("estimator_parameter_file")
            != file_record(estimator.MODEL_PATH)):
        raise ValueError("nonlinear residual frozen inputs changed")
    return payload


def load_policy():
    manifest = load_manifest()
    model = make_policy(
        int(manifest["obs_dim"]), int(manifest["act_dim"]), seed=0)
    template = nnx.state(model, nnx.Param)
    params = estimator.load_parameter_state(
        MODEL_PATH, template, manifest["parameter_metadata"])
    mean, std = load_normalization(
        NORMALIZATION_PATH, int(manifest["obs_dim"]))
    return model, params, mean, std, manifest


def validate_finite_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite residual policy metric {key}")
