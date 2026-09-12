"""Construction and sealed loading for the inverse system-ID model."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_inverse_system_id as protocol,
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
        current = nnx.merge(graphdef, params)
        return current.predict(obs, next_obs)

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
        current = nnx.merge(graphdef, params)
        predicted = current.predict(obs, next_obs)
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
    if (manifest.get("schema") != protocol.MODEL_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("env") != protocol.ENV
            or manifest.get("family") != protocol.FAMILY
            or manifest.get("model_config") != protocol.MODEL_CONFIG
            or manifest.get("parameter_file")
            != protocol.file_record(protocol.MODEL_PATH)):
        raise ValueError("invalid or stale inverse system-ID model")
    expected_gains = protocol.mode_gain_vectors(act_dim)
    gains = jnp.asarray(manifest.get("mode_gain_vectors"), dtype=jnp.float32)
    variance = jnp.asarray(
        manifest.get("residual_variance"), dtype=jnp.float32)
    if (gains.shape != expected_gains.shape
            or not jnp.allclose(gains, expected_gains)
            or variance.shape != (act_dim,)
            or not bool(jnp.all(jnp.isfinite(variance)))
            or not bool(jnp.all(variance > 0.0))):
        raise ValueError("invalid inverse system-ID calibration arrays")
    filter_config = protocol.FilterConfig.from_dict(
        manifest["filter_config"])
    model = make_model(obs_dim, act_dim, int(manifest["model_seed"]))
    restored = protocol.load_parameter_state(
        protocol.MODEL_PATH,
        nnx.state(model, nnx.Param),
        manifest["parameter_leaves"],
    )
    nnx.update(model, restored)
    return model, filter_config, gains, variance, manifest
