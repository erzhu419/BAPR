"""Teacher and student models for closed-loop policy compression v2."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_model as base_model,
)
from jax_experiments.networks.policy import GaussianPolicy


LoadedTeacher = base_model.LoadedTeacher
reduced_action = base_model.reduced_action
individual_action = base_model.individual_action
make_estimator = base_model.make_estimator


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


class ModeHeadPolicy(nnx.Module):
    """Shared observation trunk with posterior-mixed mode action heads."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_modes: int,
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        input_dim = int(obs_dim)
        for _ in range(int(n_layers)):
            layers.append(nnx.Linear(input_dim, int(hidden_dim), rngs=rngs))
            input_dim = int(hidden_dim)
        self.layers = _module_list(layers)
        self.mean_heads = _module_list([
            nnx.Linear(int(hidden_dim), int(act_dim), rngs=rngs)
            for _ in range(int(n_modes))
        ])
        self.n_modes = int(n_modes)
        self.act_dim = int(act_dim)

    def __call__(self, obs, context):
        hidden = obs
        for layer in self.layers:
            hidden = nnx.relu(layer(hidden))
        means = jnp.stack(
            [head(hidden) for head in self.mean_heads], axis=-2)
        weights = jnp.clip(context, 0.0, 1.0)
        total = jnp.sum(weights, axis=-1, keepdims=True)
        uniform = jnp.full_like(weights, 1.0 / float(self.n_modes))
        weights = jnp.where(total > 1e-6, weights / jnp.maximum(total, 1e-6), uniform)
        mean = jnp.sum(means * weights[..., :, None], axis=-2)
        return mean, jnp.zeros_like(mean)

    def deterministic(self, obs, context):
        mean, _ = self(obs, context)
        return jnp.tanh(mean)


def load_teacher(variant: str) -> LoadedTeacher:
    protocol.require_variant(variant)
    return base_model.load_teacher("combined")


def make_student(variant: str, obs_dim: int, act_dim: int, seed: int):
    variant = protocol.require_variant(variant)
    config = protocol.MODEL_CONFIG
    if variant == "wide_dagger":
        return GaussianPolicy(
            int(obs_dim),
            int(act_dim),
            int(config["wide_hidden_dim"]),
            ep_dim=len(protocol.MODES),
            n_layers=int(config["wide_n_layers"]),
            rngs=nnx.Rngs(int(seed)),
        )
    return ModeHeadPolicy(
        int(obs_dim),
        int(act_dim),
        int(config["mode_hidden_dim"]),
        int(config["mode_n_layers"]),
        len(protocol.MODES),
        rngs=nnx.Rngs(int(seed)),
    )


def build_student_action(model):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def action(params, observation, context):
        current = nnx.merge(graphdef, params)
        return current.deterministic(
            jnp.asarray(observation)[None],
            jnp.asarray(context)[None],
        )[0]

    return action


def build_student_batch_action(model):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def action(params, observations, contexts):
        current = nnx.merge(graphdef, params)
        return current.deterministic(
            jnp.asarray(observations), jnp.asarray(contexts))

    return action


def load_student(
    variant: str,
    student_seed: int,
    obs_dim: int,
    act_dim: int,
):
    destination = protocol.model_dir(variant, student_seed)
    manifest = protocol.read_json(destination / "model_manifest.json")
    if (manifest.get("schema") != protocol.MODEL_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity")
            != protocol.model_identity(variant, student_seed)
            or manifest.get("parameter_file")
            != protocol.file_record(destination / "student_params.npz")):
        raise ValueError(f"invalid closed-loop distilled student: {destination}")
    model = make_student(variant, obs_dim, act_dim, student_seed)
    state = protocol.load_parameter_state(
        destination / "student_params.npz",
        nnx.state(model, nnx.Param),
        manifest["parameter_leaves"],
    )
    return model, state, manifest


def source_records(variant: str):
    protocol.require_variant(variant)
    return base_model.source_records("combined")
