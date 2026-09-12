"""Fit a causal affine calibration over the frozen v1 likelihood heads."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from flax import nnx

from jax_experiments.analysis import regime_polarity_evidence_calibration as protocol
from jax_experiments.analysis import regime_polarity_headroom as exploratory
from jax_experiments.analysis import regime_polarity_posterior as v1
from jax_experiments.analysis import regime_polarity_posterior_model as model_lib
from jax_experiments.analysis import train_regime_polarity_posterior as trainer


def _emission_sequences(
    source_sequences,
    emission_fn,
    model_params,
):
    prepared = []
    for sequence in source_sequences:
        log_likelihood, aleatoric, epistemic = (
            trainer._sequence_emissions(
                emission_fn, model_params, sequence))
        prepared.append({
            "log_likelihood": log_likelihood,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "mode_id": np.asarray(sequence["mode_id"], dtype=np.int32),
            "identity": dict(sequence["identity"]),
        })
    return prepared


def _fit_weighted_ridge(
    features: list[np.ndarray],
    labels: list[np.ndarray],
    ridge: float,
):
    x = np.concatenate(features, axis=0).astype(np.float64)
    y = np.concatenate(labels, axis=0).astype(np.int32)
    counts = np.bincount(y, minlength=len(protocol.MODES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("calibration training data is missing a mode")
    sample_weight = 1.0 / counts[y]
    sample_weight *= len(sample_weight) / np.sum(sample_weight)
    normalizer = np.sum(sample_weight)
    feature_mean = np.sum(
        sample_weight[:, None] * x, axis=0) / normalizer
    feature_var = np.sum(
        sample_weight[:, None] * np.square(x - feature_mean),
        axis=0,
    ) / normalizer
    feature_scale = np.sqrt(np.maximum(feature_var, 1e-8))
    standardized = (x - feature_mean) / feature_scale
    design = np.concatenate([
        standardized,
        np.ones((len(standardized), 1), dtype=np.float64),
    ], axis=-1)
    targets = np.eye(len(protocol.MODES), dtype=np.float64)[y]
    weighted_design = sample_weight[:, None] * design
    gram = design.T @ weighted_design / normalizer
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = design.T @ (sample_weight[:, None] * targets) / normalizer
    coefficients = np.linalg.solve(
        gram + float(ridge) * penalty, rhs)
    return (
        feature_mean,
        feature_scale,
        coefficients[:-1],
        coefficients[-1],
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return trainer._aggregate_metrics(rows)


def _evaluate(calibrator, sequences):
    by_kind = {"stationary": [], "switching": []}
    for sequence in sequences:
        posterior = calibrator.action_posteriors(
            sequence["log_likelihood"])
        metrics = v1.posterior_metrics(
            posterior, sequence["mode_id"])
        by_kind[sequence["identity"]["kind"]].append(metrics)
    return {
        kind: _aggregate(rows)
        for kind, rows in by_kind.items()
    }


def fit_and_select(train_sequences, validation_sequences):
    candidates = []
    fit_cache = {}
    for config in protocol.candidate_configs():
        fit_key = (config.ema_alpha, config.ridge)
        if fit_key not in fit_cache:
            features = [
                protocol.temporal_features(
                    row["log_likelihood"], config.ema_alpha)
                for row in train_sequences
            ]
            labels = [row["mode_id"] for row in train_sequences]
            fit_cache[fit_key] = _fit_weighted_ridge(
                features, labels, config.ridge)
        fitted = fit_cache[fit_key]
        calibrator = protocol.CausalEvidenceCalibrator(
            config, *fitted)
        metrics = _evaluate(calibrator, validation_sequences)
        stationary = metrics["stationary"]
        switching = metrics["switching"]
        gate = (
            stationary["mode_accuracy"] >= protocol.MIN_MODE_ACCURACY
            and switching["mode_accuracy"] >= protocol.MIN_MODE_ACCURACY
            and switching["median_switch_delay"]
            <= protocol.MAX_MEDIAN_SWITCH_DELAY
            and switching["p90_switch_delay"]
            <= protocol.MAX_P90_SWITCH_DELAY
            and switching["brier_score"] <= protocol.MAX_BRIER_SCORE
        )
        score = (
            0.35 * stationary["mode_accuracy"]
            + 0.65 * switching["mode_accuracy"]
            - 0.25 * switching["brier_score"]
            - 0.002 * switching["median_switch_delay"]
            - 0.001 * switching["p90_switch_delay"]
        )
        candidates.append({
            "calibrator_config": config.to_dict(),
            "stationary": stationary,
            "switching": switching,
            "validation_gate_pass": bool(gate),
            "selection_score": float(score),
            "_fitted": fitted,
        })
    candidates.sort(key=lambda row: (
        not row["validation_gate_pass"],
        -row["selection_score"],
        row["calibrator_config"]["ema_alpha"],
        row["calibrator_config"]["ridge"],
        row["calibrator_config"]["temperature"],
    ))
    selected = candidates[0]
    public = [
        {key: value for key, value in row.items() if key != "_fitted"}
        for row in candidates
    ]
    return selected, public


def _source_records() -> dict[str, dict[str, Any]]:
    root = protocol.ROOT
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(model_lib.__file__).resolve(),
        Path(trainer.__file__).resolve(),
    )
    return {
        str(path.relative_to(root)): protocol.file_record(path)
        for path in paths
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stationary-steps",
        type=int,
        default=protocol.STATIONARY_STEPS,
    )
    parser.add_argument(
        "--switching-episodes",
        type=int,
        default=protocol.SWITCHING_EPISODES,
    )
    args = parser.parse_args()
    if args.stationary_steps <= 0 or args.switching_episodes <= 0:
        parser.error("data collection controls must be positive")
    return args


def main() -> None:
    args = parse_args()
    if protocol.MODEL_MANIFEST.is_file():
        protocol.load_calibrator()
        print(
            "POLARITY EVIDENCE CALIBRATOR ALREADY COMPLETE: "
            f"{protocol.MODEL_ROOT}",
            flush=True,
        )
        return
    if (args.stationary_steps != protocol.STATIONARY_STEPS
            or args.switching_episodes != protocol.SWITCHING_EPISODES):
        raise ValueError("production calibrator requires sealed data budget")
    if not v1.MODEL_MANIFEST.is_file() or not v1.MODEL_PATH.is_file():
        raise FileNotFoundError("frozen v1 posterior model is incomplete")

    reference_config, reference_agent, _ = trainer._load_controller(
        exploratory, protocol.TRAIN_CONTROLLER_SEEDS[0], "robust")
    del reference_config
    forward_model, _, source_manifest = model_lib.load_model(
        reference_agent.obs_dim, reference_agent.act_dim)
    emission_fn = model_lib.build_emission_fn(forward_model)
    model_params = nnx.state(forward_model, nnx.Param)

    train_source = (
        trainer._collect_stationary(
            exploratory,
            protocol.TRAIN_CONTROLLER_SEEDS,
            protocol.TRAIN_EVENT_SEEDS,
            args.stationary_steps,
        )
        + trainer._collect_switching(
            exploratory,
            protocol.TRAIN_CONTROLLER_SEEDS,
            protocol.TRAIN_EVENT_SEEDS,
            args.switching_episodes,
        )
    )
    validation_source = (
        trainer._collect_stationary(
            exploratory,
            protocol.VALIDATION_CONTROLLER_SEEDS,
            protocol.VALIDATION_EVENT_SEEDS,
            args.stationary_steps,
        )
        + trainer._collect_switching(
            exploratory,
            protocol.VALIDATION_CONTROLLER_SEEDS,
            protocol.VALIDATION_EVENT_SEEDS,
            args.switching_episodes,
        )
    )
    train_sequences = _emission_sequences(
        train_source, emission_fn, model_params)
    validation_sequences = _emission_sequences(
        validation_source, emission_fn, model_params)
    selected, candidates = fit_and_select(
        train_sequences, validation_sequences)
    config = protocol.CalibratorConfig.from_dict(
        selected["calibrator_config"])
    protocol.save_calibrator(
        protocol.MODEL_PATH, *selected["_fitted"])
    public_selected = {
        key: value for key, value in selected.items() if key != "_fitted"
    }
    manifest = {
        "schema": protocol.MODEL_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "calibrator_config": config.to_dict(),
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "source_model_manifest": protocol.file_record(v1.MODEL_MANIFEST),
        "source_model_parameters": protocol.file_record(v1.MODEL_PATH),
        "source_model_validation_gate_pass": bool(
            source_manifest["validation_gate_pass"]),
        "training_controller_seeds": list(
            protocol.TRAIN_CONTROLLER_SEEDS),
        "validation_controller_seeds": list(
            protocol.VALIDATION_CONTROLLER_SEEDS),
        "test_controller_seeds": list(
            protocol.TEST_CONTROLLER_SEEDS),
        "training_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(
            protocol.VALIDATION_EVENT_SEEDS),
        "test_event_seeds": list(protocol.TEST_EVENT_SEEDS),
        "training_data_policy": (
            "frozen v1 likelihood heads; affine labels use exploratory "
            "mode IDs only; sealed test seeds excluded"),
        "selected_validation": public_selected,
        "validation_candidates": candidates,
        "validation_gate_pass": bool(
            public_selected["validation_gate_pass"]),
        "source_files": _source_records(),
    }
    protocol.write_json_atomic(protocol.MODEL_MANIFEST, manifest)
    print(
        "POLARITY EVIDENCE CALIBRATION COMPLETE: "
        f"gate={manifest['validation_gate_pass']} "
        f"config={config.to_dict()} output={protocol.MODEL_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
