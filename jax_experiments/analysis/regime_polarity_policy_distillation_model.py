"""Frozen ensemble and student-model helpers for policy distillation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id_model as estimator_model,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_inverse_system_id_audit as inverse_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_ensemble_audit as ensemble_audit,
)
from jax_experiments.analysis import train_regime_polarity_posterior as common
from jax_experiments.networks.policy import GaussianPolicy


@dataclass(frozen=True)
class LoadedTeacher:
    config: Any
    obs_dim: int
    act_dim: int
    controller_keys: tuple[tuple[str, int], ...]
    action_groups: dict[str, tuple[Any, Any]]


def _compatible(reference, candidate, reference_agent, candidate_agent) -> bool:
    return (
        candidate.env_name == reference.env_name
        and candidate.env_type == reference.env_type
        and candidate.stochastic_mode_family
        == reference.stochastic_mode_family
        and candidate.stochastic_mode_dwell_steps
        == reference.stochastic_mode_dwell_steps
        and candidate.brax_backend == reference.brax_backend
        and candidate.task_num == reference.task_num
        and candidate.hidden_dim == reference.hidden_dim
        and candidate_agent.obs_dim == reference_agent.obs_dim
        and candidate_agent.act_dim == reference_agent.act_dim
    )


def load_teacher(group: str) -> LoadedTeacher:
    group = protocol.require_teacher_group(group)
    loaded = {"robust": [], "oracle": []}
    keys = []
    for source_group in protocol.source_groups(group):
        source = protocol.ensemble.source_protocol(source_group)
        for seed in protocol.ensemble.controller_seeds(source_group):
            keys.append((source_group, int(seed)))
            for role in ("robust", "oracle"):
                loaded[role].append(common._load_controller(
                    source, int(seed), role))
    if tuple(keys) != protocol.controller_keys(group):
        raise ValueError("distillation controller ordering changed")
    reference_config, reference_agent, _ = loaded["robust"][0]
    for role_rows in loaded.values():
        for config, agent, _ in role_rows:
            if not _compatible(
                    reference_config, config, reference_agent, agent):
                raise ValueError("distillation source controller configs differ")
    action_groups = {
        role: ensemble_audit._stacked_policy_action(
            [row[1] for row in role_rows],
            [row[2] for row in role_rows],
        )
        for role, role_rows in loaded.items()
    }
    return LoadedTeacher(
        config=reference_config,
        obs_dim=int(reference_agent.obs_dim),
        act_dim=int(reference_agent.act_dim),
        controller_keys=tuple(keys),
        action_groups=action_groups,
    )


def component_actions(
    teacher: LoadedTeacher,
    role: str,
    observation,
    context,
) -> np.ndarray:
    action_fn, policy_state = teacher.action_groups[str(role)]
    values = np.asarray(action_fn(
        policy_state,
        jnp.asarray(observation, dtype=jnp.float32),
        jnp.asarray(context, dtype=jnp.float32),
    ), dtype=np.float32)
    if (values.shape != (len(teacher.controller_keys), teacher.act_dim)
            or not np.all(np.isfinite(values))):
        raise ValueError("invalid frozen ensemble action matrix")
    return values


def reduced_action(
    teacher: LoadedTeacher,
    role: str,
    observation,
    context,
    reduction: str = protocol.REDUCTION,
) -> np.ndarray:
    return ensemble_audit._reduce_actions(
        component_actions(teacher, role, observation, context), reduction)


def individual_action(
    teacher: LoadedTeacher,
    role: str,
    member_index: int,
    observation,
    context,
) -> np.ndarray:
    return ensemble_audit._reduce_actions(
        component_actions(teacher, role, observation, context),
        "individual",
        int(member_index),
    )


def make_estimator(obs_dim: int, act_dim: int):
    model, filter_config, gains, variance, _ = estimator_model.load_model(
        int(obs_dim), int(act_dim))
    return inverse_audit.InverseSystemIDEstimator(
        estimator_model.one_step_evidence(model),
        nnx.state(model, nnx.Param),
        gains,
        variance,
        filter_config,
    )


def make_student(obs_dim: int, act_dim: int, seed: int) -> GaussianPolicy:
    return GaussianPolicy(
        int(obs_dim),
        int(act_dim),
        int(protocol.MODEL_CONFIG["hidden_dim"]),
        ep_dim=len(protocol.MODES),
        n_layers=int(protocol.MODEL_CONFIG["n_layers"]),
        rngs=nnx.Rngs(int(seed)),
    )


def build_student_action(model: GaussianPolicy):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def action(params, observation, context):
        current = nnx.merge(graphdef, params)
        return current.deterministic(
            jnp.asarray(observation)[None],
            jnp.asarray(context)[None],
        )[0]

    return action


def load_student(group: str, student_seed: int, obs_dim: int, act_dim: int):
    destination = protocol.model_dir(group, student_seed)
    manifest = protocol.read_json(destination / "model_manifest.json")
    if (manifest.get("schema") != protocol.MODEL_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity")
            != protocol.model_identity(group, student_seed)
            or manifest.get("parameter_file")
            != protocol.file_record(destination / "student_params.npz")):
        raise ValueError(f"invalid distilled student: {destination}")
    model = make_student(obs_dim, act_dim, student_seed)
    state = protocol.load_parameter_state(
        destination / "student_params.npz",
        nnx.state(model, nnx.Param),
        manifest["parameter_leaves"],
    )
    return model, state, manifest


def source_records(group: str) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for source_group in protocol.source_groups(group)
        for path in protocol.ensemble.source_bundle_manifests(source_group)
    }
