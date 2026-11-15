"""Protocol helpers for the frozen-base, independent-adapter fork.

The development screen starts from one completed RegimeSAC robust controller.
One branch continues robust training for 700 iterations.  Each adapter family
uses four independent fixed-mode BAPRRegime continuations of 175 iterations,
so the four adapters consume the same aggregate 2.8M transitions as the robust
continuation.  This is a common-controller rebootstrap, not an exact trajectory
continuation: replay is empty and optimizer states are reset in every branch.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import regime_control_headroom as source


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1"
ENV = "HalfCheetah-v2"
FAMILY = source.FAMILY
MODES = source.MODES
TRAINING_SEEDS = source.TRAINING_SEEDS
DEVELOPMENT_SEEDS = (8,)
RESIDUAL_DELTAS = (0.25, 0.5, 1.0)
CALIBRATION_EVENT_SEEDS = (74100, 74200)
AUDIT_EVENT_SEEDS = (75100, 75200, 75300, 75400, 75500)

SOURCE_NEXT_ITERATION = source.MAX_ITERS
SOURCE_TOTAL_STEPS = source.FINAL_TOTAL_STEPS
SOURCE_UPDATE_COUNT = source.FINAL_UPDATE_COUNT
SAMPLES_PER_ITER = source.SAMPLES_PER_ITER
UPDATES_PER_ITER = source.UPDATES_PER_ITER
ROBUST_EXTRA_ITERS = 700
ADAPTER_EXTRA_ITERS_PER_MODE = ROBUST_EXTRA_ITERS // len(MODES)
if ADAPTER_EXTRA_ITERS_PER_MODE * len(MODES) != ROBUST_EXTRA_ITERS:
    raise RuntimeError("adapter allocation must exactly match robust budget")
ROBUST_FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + ROBUST_EXTRA_ITERS
ROBUST_FINAL_TOTAL_STEPS = (
    SOURCE_TOTAL_STEPS + ROBUST_EXTRA_ITERS * SAMPLES_PER_ITER)
ROBUST_FINAL_UPDATE_COUNT = (
    SOURCE_UPDATE_COUNT + ROBUST_EXTRA_ITERS * UPDATES_PER_ITER)
ADAPTER_FINAL_NEXT_ITERATION = (
    SOURCE_NEXT_ITERATION + ADAPTER_EXTRA_ITERS_PER_MODE)
ADAPTER_FINAL_TOTAL_STEPS = (
    SOURCE_TOTAL_STEPS + ADAPTER_EXTRA_ITERS_PER_MODE * SAMPLES_PER_ITER)
ADAPTER_FINAL_UPDATE_COUNT = (
    SOURCE_UPDATE_COUNT + ADAPTER_EXTRA_ITERS_PER_MODE * UPDATES_PER_ITER)
MAX_EPISODE_STEPS = source.MAX_EPISODE_STEPS
DWELL_STEPS = source.DWELL_STEPS
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5

RUN_ROOT = ROOT / "jax_experiments" / "results_regime_adapter_fork_v1"
BUNDLE_ROOT = (
    ROOT / "jax_experiments" / "eval_bundles_regime_adapter_fork_v1")
CALIBRATION_ROOT = (
    ROOT / "jax_experiments" / "results_regime_adapter_calibration_v1")
AUDIT_ROOT = ROOT / "jax_experiments" / "results_regime_adapter_audit_v1"
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_regime_adapter_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports" / "regime_adapter_fork_protocol_2026-07-22.md")

BOOTSTRAP_NAME = "regime_adapter_bootstrap.json"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
CALIBRATION_MANIFEST_NAME = "calibration_manifest.json"
AUDIT_MANIFEST_NAME = "audit_manifest.json"
BUNDLE_SCHEMA = "bapr.regime-adapter-bundle.v1"
CALIBRATION_SCHEMA = "bapr.regime-adapter-calibration.v1"
AUDIT_SCHEMA = "bapr.regime-adapter-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-adapter-analysis.v1"


def require_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown regime-adapter training seed {seed}")
    return seed


def require_mode(mode: int) -> int:
    mode = int(mode)
    if mode not in MODES:
        raise ValueError(f"unknown regime-adapter mode {mode}")
    return mode


def require_delta(delta: float) -> float:
    delta = float(delta)
    if delta not in RESIDUAL_DELTAS:
        raise ValueError(f"unknown residual delta {delta}")
    return delta


def delta_slug(delta: float) -> str:
    return f"d{int(round(require_delta(delta) * 100)):03d}"


def source_bundle_dir(seed: int) -> Path:
    return source.bundle_dir(ENV, "robust", require_seed(seed))


def robust_run_dir(seed: int) -> Path:
    return RUN_ROOT / f"seed_{require_seed(seed)}" / "robust_continue"


def robust_bundle_dir(seed: int) -> Path:
    return BUNDLE_ROOT / f"seed_{require_seed(seed)}" / "robust_continue"


def adapter_run_dir(seed: int, delta: float, mode: int) -> Path:
    return (
        RUN_ROOT / f"seed_{require_seed(seed)}" / delta_slug(delta)
        / f"mode_{require_mode(mode)}")


def adapter_bundle_dir(seed: int, delta: float, mode: int) -> Path:
    return (
        BUNDLE_ROOT / f"seed_{require_seed(seed)}" / delta_slug(delta)
        / f"mode_{require_mode(mode)}")


def branch_bundle_dir(
        seed: int, role: str, delta: float | None = None,
        mode: int | None = None) -> Path:
    if role == "robust_continue":
        if delta is not None or mode is not None:
            raise ValueError("robust branch cannot have delta or mode")
        return robust_bundle_dir(seed)
    if role != "adapter":
        raise ValueError(f"unknown branch role {role!r}")
    if delta is None or mode is None:
        raise ValueError("adapter branch requires delta and mode")
    return adapter_bundle_dir(seed, delta, mode)


def bundle_manifest(directory: Path) -> Path:
    return directory / BUNDLE_MANIFEST_NAME


def required_bundle_paths(directory: Path) -> tuple[Path, ...]:
    return (
        directory / BUNDLE_MANIFEST_NAME,
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
        directory / "checkpoints" / BOOTSTRAP_NAME,
    )


def all_training_bundle_dirs(seed: int, delta: float) -> tuple[Path, ...]:
    return (robust_bundle_dir(seed),) + tuple(
        adapter_bundle_dir(seed, delta, mode) for mode in MODES)


def calibration_dir(seed: int, delta: float) -> Path:
    return (
        CALIBRATION_ROOT / f"seed_{require_seed(seed)}"
        / delta_slug(delta))


def calibration_manifest(seed: int, delta: float) -> Path:
    return calibration_dir(seed, delta) / CALIBRATION_MANIFEST_NAME


def audit_dir(seed: int, delta: float, event_seed: int) -> Path:
    event_seed = int(event_seed)
    if event_seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown sealed event seed {event_seed}")
    return (
        AUDIT_ROOT / f"seed_{require_seed(seed)}" / delta_slug(delta)
        / f"event_{event_seed}")


def audit_manifest(seed: int, delta: float, event_seed: int) -> Path:
    return audit_dir(seed, delta, event_seed) / AUDIT_MANIFEST_NAME


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(
        seed: int, role: str, delta: float | None = None,
        mode: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_seed(seed),
        "role": role,
    }
    if role == "robust_continue":
        payload["algo"] = "regime_sac"
    elif role == "adapter":
        payload.update({
            "algo": "bapr_regime",
            "residual_delta": require_delta(float(delta)),
            "fixed_mode": require_mode(int(mode)),
        })
    else:
        raise ValueError(f"unknown role {role!r}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def file_record(path: Path) -> dict[str, Any]:
    return source.file_record(path)


def _hash_arrays(values: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps({
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(header)
        digest.update(b"\0")
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def base_policy_sha256(policy) -> str:
    values: list[tuple[str, Any]] = []
    for index, layer in enumerate(policy.base_layers):
        values.extend([
            (f"base_layers.{index}.kernel", layer.kernel.value),
            (f"base_layers.{index}.bias", layer.bias.value),
        ])
    values.extend([
        ("base_mean.kernel", policy.base_mean.kernel.value),
        ("base_mean.bias", policy.base_mean.bias.value),
        ("base_log_std.kernel", policy.base_log_std.kernel.value),
        ("base_log_std.bias", policy.base_log_std.bias.value),
    ])
    return _hash_arrays(values)


def residual_policy_sha256(policy) -> str:
    values: list[tuple[str, Any]] = []
    for index, layer in enumerate(policy.residual_layers):
        values.extend([
            (f"residual_layers.{index}.kernel", layer.kernel.value),
            (f"residual_layers.{index}.bias", layer.bias.value),
        ])
    values.extend([
        ("residual_mean.kernel", policy.residual_mean.kernel.value),
        ("residual_mean.bias", policy.residual_mean.bias.value),
    ])
    return _hash_arrays(values)


def critic_sha256(critic) -> str:
    values: list[tuple[str, Any]] = []
    for index, layer in enumerate(critic.layers):
        values.extend([
            (f"layers.{index}.kernel", layer.kernel.value),
            (f"layers.{index}.bias", layer.bias.value),
        ])
    return _hash_arrays(values)


def source_policy_sha256(policy) -> str:
    values: list[tuple[str, Any]] = []
    for index, layer in enumerate(policy.layers):
        values.extend([
            (f"layers.{index}.kernel", layer.kernel.value),
            (f"layers.{index}.bias", layer.bias.value),
        ])
    values.extend([
        ("mean_head.kernel", policy.mean_head.kernel.value),
        ("mean_head.bias", policy.mean_head.bias.value),
        ("log_std_head.kernel", policy.log_std_head.kernel.value),
        ("log_std_head.bias", policy.log_std_head.bias.value),
    ])
    return _hash_arrays(values)


def _copy_value(destination, source_value) -> None:
    destination.value = jnp.asarray(source_value, dtype=destination.value.dtype)


def copy_source_policy_to_frozen_base(source_agent, target_agent) -> None:
    """Copy the zero-context RegimeSAC actor into the obs-only frozen base."""
    source_policy = source_agent.policy
    target_policy = target_agent.policy
    if len(source_policy.layers) != len(target_policy.base_layers):
        raise ValueError("source and target actor depths differ")
    for index, (source_layer, target_layer) in enumerate(zip(
            source_policy.layers, target_policy.base_layers)):
        kernel = source_layer.kernel.value
        if index == 0:
            if kernel.shape[0] != (
                    target_agent.obs_dim + source_agent.context_dim):
                raise ValueError("source policy input width is inconsistent")
            kernel = kernel[:target_agent.obs_dim]
        if kernel.shape != target_layer.kernel.value.shape:
            raise ValueError(
                f"policy layer {index} shape mismatch: "
                f"{kernel.shape} != {target_layer.kernel.value.shape}")
        _copy_value(target_layer.kernel, kernel)
        _copy_value(target_layer.bias, source_layer.bias.value)
    for source_head, target_head in (
            (source_policy.mean_head, target_policy.base_mean),
            (source_policy.log_std_head, target_policy.base_log_std)):
        _copy_value(target_head.kernel, source_head.kernel.value)
        _copy_value(target_head.bias, source_head.bias.value)
    if not target_policy.zero_residual_output():
        raise ValueError("target policy is not a residual policy")


def copy_source_critic(source_critic, target_critic, *, obs_dim: int,
                       source_context_dim: int,
                       target_context_dim: int) -> None:
    """Embed a zero-context critic into the wider adapter context input."""
    if len(source_critic.layers) != len(target_critic.layers):
        raise ValueError("source and target critic depths differ")
    for index, (source_layer, target_layer) in enumerate(zip(
            source_critic.layers, target_critic.layers)):
        source_kernel = source_layer.kernel.value
        if index == 0:
            target_kernel = jnp.zeros_like(target_layer.kernel.value)
            source_action_start = obs_dim + source_context_dim
            target_action_start = obs_dim + target_context_dim
            action_dim = source_kernel.shape[1] - source_action_start
            if target_kernel.shape[1] - target_action_start != action_dim:
                raise ValueError("source and target critic action widths differ")
            target_kernel = target_kernel.at[:, :obs_dim].set(
                source_kernel[:, :obs_dim])
            target_kernel = target_kernel.at[
                :, target_action_start:target_action_start + action_dim
            ].set(source_kernel[:, source_action_start:])
        else:
            target_kernel = source_kernel
        if target_kernel.shape != target_layer.kernel.value.shape:
            raise ValueError(
                f"critic layer {index} shape mismatch: "
                f"{target_kernel.shape} != {target_layer.kernel.value.shape}")
        _copy_value(target_layer.kernel, target_kernel)
        _copy_value(target_layer.bias, source_layer.bias.value)


def copy_source_controller_to_adapter(source_agent, target_agent) -> None:
    copy_source_policy_to_frozen_base(source_agent, target_agent)
    copy_source_critic(
        source_agent.critic, target_agent.critic,
        obs_dim=source_agent.obs_dim,
        source_context_dim=source_agent.context_dim,
        target_context_dim=target_agent.context_dim)
    copy_source_critic(
        source_agent.target_critic, target_agent.target_critic,
        obs_dim=source_agent.obs_dim,
        source_context_dim=source_agent.context_dim,
        target_context_dim=target_agent.context_dim)
    target_agent.log_alpha = jnp.asarray(source_agent.log_alpha)
    target_agent.update_count = int(source_agent.update_count)


def expected_checkpoint(role: str) -> dict[str, int | str]:
    if role == "robust_continue":
        return {
            "iteration": ROBUST_FINAL_NEXT_ITERATION - 1,
            "next_iteration": ROBUST_FINAL_NEXT_ITERATION,
            "total_steps": ROBUST_FINAL_TOTAL_STEPS,
            "update_count": ROBUST_FINAL_UPDATE_COUNT,
            "algo": "regime_sac",
        }
    if role == "adapter":
        return {
            "iteration": ADAPTER_FINAL_NEXT_ITERATION - 1,
            "next_iteration": ADAPTER_FINAL_NEXT_ITERATION,
            "total_steps": ADAPTER_FINAL_TOTAL_STEPS,
            "update_count": ADAPTER_FINAL_UPDATE_COUNT,
            "algo": "bapr_regime",
        }
    raise ValueError(f"unknown role {role!r}")

