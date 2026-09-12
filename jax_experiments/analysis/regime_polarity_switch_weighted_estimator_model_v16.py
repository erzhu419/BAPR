"""Construction and loading for the v16 switch-weighted inverse model."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_v16 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_inverse_system_id_audit import (
    InverseSystemIDEstimator,
)
from jax_experiments.networks.executed_action_inverse import (
    ExecutedActionInverse,
    candidate_action_evidence,
)


def make_model(obs_dim: int, act_dim: int, seed: int):
    config = protocol.MODEL_CONFIG
    return ExecutedActionInverse(
        obs_dim,
        act_dim,
        hidden_dim=config["hidden_dim"],
        ensemble_size=config["ensemble_size"],
        n_layers=config["n_layers"],
        obs_scale=config["obs_scale"],
        delta_scale=config["delta_scale"],
        rngs=nnx.Rngs(seed),
    )


def build_prediction_fn(model):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def predict(params, obs, next_obs):
        return nnx.merge(graphdef, params).predict(obs, next_obs)

    return predict


def build_evidence_fn(model):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def evidence(
        params,
        obs,
        next_obs,
        commanded_action,
        gain_vectors,
        residual_variance,
    ):
        predicted = nnx.merge(graphdef, params).predict(obs, next_obs)
        return candidate_action_evidence(
            predicted,
            commanded_action,
            gain_vectors,
            residual_variance,
        )

    return evidence


def one_step_evidence(model):
    batched = build_evidence_fn(model)

    @jax.jit
    def apply(
        params,
        obs,
        next_obs,
        commanded_action,
        gain_vectors,
        residual_variance,
    ):
        values = batched(
            params,
            jnp.asarray(obs)[None, :],
            jnp.asarray(next_obs)[None, :],
            jnp.asarray(commanded_action)[None, :],
            gain_vectors,
            residual_variance,
        )
        return tuple(value[0] for value in values)

    return apply


def load_model(obs_dim: int, act_dim: int):
    manifest = protocol.read_json(protocol.MODEL_MANIFEST)
    if (
        manifest.get("schema") != protocol.MODEL_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("env") != protocol.ENV
        or manifest.get("family") != protocol.FAMILY
        or manifest.get("model_config") != protocol.MODEL_CONFIG
        or manifest.get("filter_config") != protocol.FILTER_CONFIG
        or manifest.get("parameter_file")
        != protocol.file_record(protocol.MODEL_PATH)
    ):
        raise ValueError("invalid v16 switch-weighted estimator model")
    expected_gains = protocol.mode_gain_vectors(act_dim)
    gains = np.asarray(manifest.get("mode_gain_vectors"), dtype=np.float32)
    variance = np.asarray(
        manifest.get("residual_variance"), dtype=np.float32)
    if (
        gains.shape != expected_gains.shape
        or not np.allclose(gains, expected_gains)
        or variance.shape != (act_dim,)
        or not np.all(np.isfinite(variance))
        or not np.all(variance > 0.0)
    ):
        raise ValueError("invalid v16 estimator calibration arrays")
    model = make_model(obs_dim, act_dim, int(manifest["model_seed"]))
    restored = protocol.load_parameter_state(
        protocol.MODEL_PATH,
        nnx.state(model, nnx.Param),
        manifest["parameter_leaves"],
    )
    nnx.update(model, restored)
    return (
        model,
        protocol.FilterConfig.from_dict(manifest["filter_config"]),
        jnp.asarray(gains),
        jnp.asarray(variance),
        manifest,
    )


def make_estimator(obs_dim: int, act_dim: int):
    model, filter_config, gains, variance, _ = load_model(obs_dim, act_dim)
    return InverseSystemIDEstimator(
        one_step_evidence(model),
        nnx.state(model, nnx.Param),
        gains,
        variance,
        filter_config,
    )
