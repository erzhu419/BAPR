"""Model construction and sealed loading for the polarity posterior screen."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.analysis import regime_polarity_posterior as protocol
from jax_experiments.networks.probabilistic_regime_context import (
    ProbabilisticRegimeContext,
)


def make_model(obs_dim: int, act_dim: int, seed: int):
    config = protocol.MODEL_CONFIG
    return ProbabilisticRegimeContext(
        obs_dim,
        act_dim,
        num_modes=len(protocol.MODES),
        hidden_dim=config["hidden_dim"],
        ensemble_size=config["ensemble_size"],
        mode="supervised",
        likelihood="probabilistic",
        reward_scale=config["reward_scale"],
        delta_scale=config["delta_scale"],
        min_history=1,
        hazard_rate=0.004,
        evidence_scale=1.0,
        fixed_variance=config["fixed_variance"],
        variance_model=config["variance_model"],
        variance_floor=config["variance_floor"],
        variance_ceiling=config["variance_ceiling"],
        mean_loss_weight=config["mean_loss_weight"],
        variance_loss_weight=0.0,
        variance_prior_weight=0.0,
        evidence_clip=0.0,
        surprise_threshold=1e9,
        rngs=nnx.Rngs(seed),
    )


def build_emission_fn(model):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def emissions(params, obs, act, rew, next_obs):
        current = nnx.merge(graphdef, params)

        def one(observation, action, reward, following):
            _, statistics = current.transition_statistics(
                observation, action, reward, following,
                stop_variance_grad=True)
            return (
                statistics[0],
                statistics[2],
                statistics[3],
            )

        return jax.vmap(one)(obs, act, rew, next_obs)

    return emissions


def load_model(obs_dim: int, act_dim: int):
    manifest = protocol.read_json(protocol.MODEL_MANIFEST)
    if (manifest.get("schema") != protocol.MODEL_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("env") != protocol.ENV
            or manifest.get("family") != protocol.FAMILY
            or manifest.get("model_config") != protocol.MODEL_CONFIG
            or manifest.get("parameter_file")
            != protocol.file_record(protocol.MODEL_PATH)):
        raise ValueError("invalid or stale polarity posterior model")
    filter_config = protocol.FilterConfig.from_dict(
        manifest["filter_config"])
    model = make_model(obs_dim, act_dim, seed=int(manifest["model_seed"]))
    restored = protocol.load_parameter_state(
        protocol.MODEL_PATH,
        nnx.state(model, nnx.Param),
        manifest["parameter_leaves"],
    )
    nnx.update(model, restored)
    return model, filter_config, manifest


def one_step_emission(model):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def apply(params, obs, action, reward, next_obs):
        current = nnx.merge(graphdef, params)
        _, statistics = current.transition_statistics(
            jnp.asarray(obs),
            jnp.asarray(action),
            jnp.asarray(reward),
            jnp.asarray(next_obs),
            stop_variance_grad=True,
        )
        return statistics[0], statistics[2], statistics[3]

    return apply

