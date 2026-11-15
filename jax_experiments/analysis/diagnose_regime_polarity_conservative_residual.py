"""Run one exact-budget iteration to diagnose conservative actor rejection."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_conservative_residual as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_anchor_branch as runner,
)


DIAGNOSTIC_VERSION = "v2-stable-lcb-gradient"
DIAGNOSTIC_ITERS = 1
RUN_ROOT = (
    protocol.ROOT / "jax_experiments"
    / "results_regime_polarity_conservative_residual_diagnostic_v2")
OUTPUT_ROOT = (
    protocol.ROOT / "jax_experiments"
    / "results_regime_polarity_conservative_residual_diagnostic_analysis_v2")
METRICS = (
    "v2_train_update_accept_rate",
    "v2_train_candidate_advantage_min",
    "v2_train_candidate_advantage_mean",
    "v2_train_candidate_regression_margin_min",
    "v2_train_candidate_floor_margin_min",
    "v2_train_candidate_nonfinite_rate",
    "v2_train_update_reject_regression_rate",
    "v2_train_update_reject_floor_rate",
)


def run_dir(variant: str, seed: int) -> Path:
    return (
        RUN_ROOT / protocol.env_slug(protocol.ENV) / "branches"
        / protocol.require_variant(variant)
        / f"seed_{protocol.require_training_seed(seed)}")


def output_dir(variant: str, seed: int) -> Path:
    return (
        OUTPUT_ROOT / protocol.require_variant(variant)
        / f"seed_{protocol.require_training_seed(seed)}")


def output_path(variant: str, seed: int) -> Path:
    return output_dir(variant, seed) / "diagnostic.json"


def _bind_short_protocol() -> None:
    runner.protocol = protocol
    protocol.RUN_ROOT = RUN_ROOT
    protocol.BRANCH_EXTRA_ITERS = DIAGNOSTIC_ITERS
    protocol.BRANCH_FINAL_NEXT_ITERATION = (
        protocol.SOURCE_NEXT_ITERATION + DIAGNOSTIC_ITERS)
    protocol.BRANCH_FINAL_ITERATION = (
        protocol.BRANCH_FINAL_NEXT_ITERATION - 1)
    protocol.BRANCH_TOTAL_STEPS = (
        protocol.SOURCE_TOTAL_STEPS
        + DIAGNOSTIC_ITERS * protocol.SAMPLES_PER_ITER)
    protocol.BRANCH_UPDATE_COUNT = (
        protocol.SOURCE_UPDATE_COUNT
        + DIAGNOSTIC_ITERS * protocol.UPDATES_PER_ITER)


def _read_metric(run_dir: Path, name: str) -> np.ndarray:
    path = run_dir / "logs" / f"{name}.npy"
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    values = values.reshape(-1)[-DIAGNOSTIC_ITERS:]
    if values.size != DIAGNOSTIC_ITERS:
        raise ValueError(f"incomplete diagnostic metric {name}: {path}")
    return values


def _summary(seed: int, variant: str, run_dir: Path) -> dict:
    overall = {
        name: float(np.mean(_read_metric(run_dir, name)))
        for name in METRICS
    }
    modes = {}
    for mode in protocol.MODES:
        prefix = f"v2_train_candidate_mode_{mode}"
        modes[str(mode)] = {
            "advantage": float(np.mean(_read_metric(
                run_dir, f"{prefix}_advantage"))),
            "regression_margin": float(np.mean(_read_metric(
                run_dir, f"{prefix}_regression_margin"))),
            "floor_margin": float(np.mean(_read_metric(
                run_dir, f"{prefix}_floor_margin"))),
            "represented_rate": float(np.mean(_read_metric(
                run_dir, f"{prefix}_represented_rate"))),
            "nonfinite_rate": float(np.mean(_read_metric(
                run_dir, f"{prefix}_nonfinite_rate"))),
            "reject_regression_rate": float(np.mean(_read_metric(
                run_dir, f"{prefix}_reject_regression_rate"))),
            "reject_floor_rate": float(np.mean(_read_metric(
                run_dir, f"{prefix}_reject_floor_rate"))),
        }

    config, env, agent, next_iteration, total_steps, temporary = (
        runner._load_final(run_dir))
    try:
        components = runner._component_hashes(agent, variant)
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()
    bootstrap = protocol.read_json(
        run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME)
    initial = bootstrap["initial_components"]
    return {
        "schema": "bapr.regime-polarity-conservative-diagnostic.v2",
        "status": "complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "identity": {
            "variant": variant,
            "training_seed": seed,
            "source_next_iteration": protocol.SOURCE_NEXT_ITERATION,
            "diagnostic_iterations": DIAGNOSTIC_ITERS,
        },
        "checkpoint": {
            "next_iteration": int(next_iteration),
            "total_steps": int(total_steps),
            "update_count": int(agent.update_count),
        },
        "actor": {
            "base_unchanged": (
                components["base_policy"] == initial["base_policy"]),
            "adaptive_changed": (
                components["adaptive_policy"]
                != initial["adaptive_policy"]),
            "initial": initial,
            "final": components,
        },
        "overall": overall,
        "modes": modes,
    }


def run(seed: int, variant: str) -> dict:
    seed = protocol.require_training_seed(seed)
    variant = protocol.require_variant(variant)
    _bind_short_protocol()
    destination = output_path(variant, seed)
    if destination.is_file():
        payload = protocol.read_json(destination)
        if (
            payload.get("status") == "complete"
            and payload.get("diagnostic_version") == DIAGNOSTIC_VERSION
        ):
            print(f"DIAGNOSTIC ALREADY COMPLETE: {destination}")
            return payload
        raise ValueError(f"invalid existing diagnostic: {destination}")

    branch_dir = run_dir(variant, seed)
    runner._bootstrap(seed, variant, branch_dir)
    expected_next = protocol.BRANCH_FINAL_NEXT_ITERATION
    checkpoint = protocol.checkpoint_record(branch_dir)
    if checkpoint["next_iteration"] < expected_next:
        command = runner._training_command(seed, variant, branch_dir)
        print("CONSERVATIVE DIAGNOSTIC TRAIN:", " ".join(command), flush=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(protocol.ROOT)
        subprocess.run(
            command,
            cwd=protocol.ROOT,
            env=environment,
            check=True,
        )
    elif checkpoint["next_iteration"] > expected_next:
        raise ValueError(
            f"diagnostic checkpoint exceeds budget: {checkpoint}")

    payload = _summary(seed, variant, branch_dir)
    expected = {
        "next_iteration": expected_next,
        "total_steps": protocol.BRANCH_TOTAL_STEPS,
        "update_count": protocol.BRANCH_UPDATE_COUNT,
    }
    actual = payload["checkpoint"]
    if actual != expected:
        raise ValueError(
            f"diagnostic checkpoint mismatch: {actual} != {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(destination, payload)
    replay = branch_dir / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument(
        "--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed, args.variant)


if __name__ == "__main__":
    main()
