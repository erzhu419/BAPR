"""Audit independent polarity controllers and deterministic policy ensembles."""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_ensemble as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_inverse_system_id_audit as inverse_audit,
)
from jax_experiments.analysis import train_regime_polarity_posterior as common
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


InverseSystemIDEstimator = inverse_audit.InverseSystemIDEstimator


def _stacked_policy_action(agents, states):
    if not agents or len(agents) != len(states):
        raise ValueError("policy ensemble requires matched nonempty agents/states")
    obs_dim = agents[0].obs_dim
    act_dim = agents[0].act_dim
    if any(agent.obs_dim != obs_dim or agent.act_dim != act_dim for agent in agents):
        raise ValueError("policy ensemble controller dimensions differ")
    structures = [jax.tree_util.tree_structure(state) for state in states]
    if any(structure != structures[0] for structure in structures[1:]):
        raise ValueError("policy ensemble parameter structures differ")
    graphdef = nnx.graphdef(agents[0].policy)
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)

    @jax.jit
    def action(params, observation, context):
        def one(member_params):
            policy = nnx.merge(graphdef, member_params)
            return policy.deterministic(
                jnp.asarray(observation)[None],
                jnp.asarray(context)[None],
            )[0]

        return jax.vmap(one)(params)

    return action, stacked


def _reduce_actions(
    component_actions: np.ndarray,
    reduction: str,
    member_index: int | None = None,
) -> np.ndarray:
    component_actions = np.asarray(component_actions, dtype=np.float32)
    if component_actions.ndim != 2 or not np.all(np.isfinite(component_actions)):
        raise ValueError("invalid component action matrix")
    if reduction == "individual":
        if member_index is None:
            raise ValueError("individual reduction requires a member index")
        return component_actions[int(member_index)].copy()
    if reduction == "mean":
        return np.mean(component_actions, axis=0, dtype=np.float32)
    if reduction == "median":
        return np.median(component_actions, axis=0).astype(np.float32)
    raise ValueError(f"unknown policy-ensemble reduction {reduction!r}")


def _context(kind: str, physical_mode: int, posterior: np.ndarray) -> np.ndarray:
    if kind == "zero":
        return np.zeros((len(protocol.MODES),), dtype=np.float32)
    if kind == "true":
        return np.eye(len(protocol.MODES), dtype=np.float32)[physical_mode]
    if kind == "learned":
        return np.asarray(posterior, dtype=np.float32)
    raise ValueError(f"unknown ensemble context kind {kind!r}")


def _member_index(group: str, seed: int | None) -> int | None:
    if seed is None:
        return None
    return protocol.controller_seeds(group).index(int(seed))


def _action(
    group: str,
    arm,
    action_groups,
    observation,
    physical_mode: int,
    posterior: np.ndarray,
) -> tuple[np.ndarray, float]:
    context = _context(arm.context_kind, physical_mode, posterior)
    action_fn, policy_state = action_groups[arm.source_role]
    components = np.asarray(action_fn(policy_state, observation, context))
    selected = _reduce_actions(
        components,
        arm.reduction,
        _member_index(group, arm.controller_seed),
    )
    disagreement = float(np.mean(np.std(components, axis=0)))
    return selected, disagreement


def _stationary(
    group: str,
    config,
    arm,
    action_groups,
    estimator,
    event_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for mode in protocol.MODES:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            run_config,
            seed_offset=int(event_seed) - int(config.seed),
        )
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[mode])
        returns = []
        terminations = []
        disagreements = []
        posterior_rows = []
        labels = []
        for _ in range(protocol.EPISODES_PER_TASK):
            observation = env.reset()
            estimator_state = estimator.initial_state()
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                posterior = estimator.probabilities(estimator_state)
                action, disagreement = _action(
                    group,
                    arm,
                    action_groups,
                    observation,
                    int(mode),
                    posterior,
                )
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != int(mode):
                    raise ValueError("stationary physics mode changed")
                if arm.context_kind == "learned":
                    posterior_rows.append(posterior.copy())
                    labels.append(int(mode))
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )
                episode_return += float(reward)
                disagreements.append(disagreement)
                observation = next_observation
                if done:
                    episode_terminated = True
                    observation = env.reset()
                    if arm.context_kind == "learned":
                        estimator_state = estimator.initial_state()
            returns.append(episode_return)
            terminations.append(float(episode_terminated))
        row = {
            "arm": arm.label,
            "mode": int(mode),
            "returns": [float(value) for value in returns],
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "terminated_rate": float(np.mean(terminations)),
            "action_disagreement_mean": float(np.mean(disagreements)),
        }
        if arm.context_kind == "learned":
            row["posterior_metrics"] = protocol.final.posterior_metrics(
                np.asarray(posterior_rows),
                np.asarray(labels, dtype=np.int32),
            )
        rows.append(row)
        if hasattr(env, "close"):
            env.close()
    return rows


def _switching(
    group: str,
    config,
    arm,
    action_groups,
    estimator,
    event_seed: int,
) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config,
        seed_offset=int(event_seed) - int(config.seed),
    )
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    disagreements = []
    posterior_rows = []
    labels = []
    for episode in range(protocol.SWITCHING_EPISODES):
        sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
        _reset_eval_switch_schedule(
            env, sequence, run_config, protocol.DWELL_STEPS)
        observation = env.reset()
        estimator_state = estimator.initial_state()
        episode_return = 0.0
        episode_terminated = False
        for _ in range(protocol.MAX_EPISODE_STEPS):
            physical_mode = int(env.task_id_for_next_step())
            posterior = estimator.probabilities(estimator_state)
            action, disagreement = _action(
                group,
                arm,
                action_groups,
                observation,
                physical_mode,
                posterior,
            )
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != physical_mode:
                raise ValueError("switching mode label is not action-causal")
            if arm.context_kind == "learned":
                posterior_rows.append(posterior.copy())
                labels.append(physical_mode)
                estimator_state, _, _, _ = estimator.step(
                    estimator_state,
                    observation,
                    action,
                    reward,
                    next_observation,
                )
            episode_return += float(reward)
            disagreements.append(disagreement)
            observation = next_observation
            if done:
                episode_terminated = True
                # Preserve the physical switch clock and posterior exactly as
                # in the strict confirmation audit.
                observation = env.reset()
        returns.append(episode_return)
        terminations.append(float(episode_terminated))
    if hasattr(env, "close"):
        env.close()
    row = {
        "arm": arm.label,
        "returns": [float(value) for value in returns],
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated_rate": float(np.mean(terminations)),
        "action_disagreement_mean": float(np.mean(disagreements)),
    }
    if arm.context_kind == "learned":
        row["posterior_metrics"] = protocol.final.posterior_metrics(
            np.asarray(posterior_rows),
            np.asarray(labels, dtype=np.int32),
        )
    return row


def _load_group(group: str):
    source = protocol.source_protocol(group)
    loaded = {
        role: [
            common._load_controller(source, seed, role)
            for seed in protocol.controller_seeds(group)
        ]
        for role in source.ROLES
    }
    reference = loaded["robust"][0][0]
    for role_rows in loaded.values():
        for config, agent, _ in role_rows:
            if (config.env_name != reference.env_name
                    or config.env_type != reference.env_type
                    or config.stochastic_mode_family
                    != reference.stochastic_mode_family
                    or config.stochastic_mode_dwell_steps
                    != reference.stochastic_mode_dwell_steps
                    or config.brax_backend != reference.brax_backend
                    or agent.obs_dim != loaded["robust"][0][1].obs_dim
                    or agent.act_dim != loaded["robust"][0][1].act_dim):
                raise ValueError("controller ensemble source configs differ")
    action_groups = {
        role: _stacked_policy_action(
            [row[1] for row in role_rows],
            [row[2] for row in role_rows],
        )
        for role, role_rows in loaded.items()
    }
    reference_agent = loaded["robust"][0][1]
    return reference, reference_agent.obs_dim, reference_agent.act_dim, action_groups


def _source_records(group: str) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in protocol.source_bundle_manifests(group)
    }


def validate_audit(group: str, event_seed: int) -> dict[str, Any]:
    group = protocol.require_group(group)
    event_seed = protocol.require_event_seed(event_seed)
    destination = protocol.audit_dir(group, event_seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    if (manifest.get("schema") != protocol.AUDIT_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity") != protocol.identity(group, event_seed)
            or manifest.get("source_bundles") != _source_records(group)
            or manifest.get("result_file")
            != protocol.file_record(destination / "results.json")):
        raise ValueError(f"invalid policy-ensemble audit: {destination}")
    result = protocol.read_json(destination / "results.json")
    labels = set(protocol.arm_labels(group))
    if (result.get("schema") != protocol.EVENT_SCHEMA
            or result.get("status") != "complete"
            or result.get("identity") != protocol.identity(group, event_seed)
            or {row.get("arm") for row in result.get("switching", [])}
            != labels
            or len(result.get("stationary", []))
            != len(labels) * len(protocol.MODES)):
        raise ValueError(f"invalid policy-ensemble result: {destination}")
    for row in result["stationary"]:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        if (values.shape != (protocol.EPISODES_PER_TASK,)
                or not np.all(np.isfinite(values))):
            raise ValueError("policy-ensemble return vector is invalid")
    for row in result["switching"]:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        if (values.shape != (protocol.SWITCHING_EPISODES,)
                or not np.all(np.isfinite(values))):
            raise ValueError("policy-ensemble switching vector is invalid")
    return manifest


def run(group: str, event_seed: int) -> None:
    group = protocol.require_group(group)
    event_seed = protocol.require_event_seed(event_seed)
    destination = protocol.audit_dir(group, event_seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(group, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"POLICY ENSEMBLE AUDIT ALREADY COMPLETE: {destination}")
            return

    protocol.validate_frozen_estimator()
    config, obs_dim, act_dim, action_groups = _load_group(group)
    model, filter_config, gains, variance, _ = model_lib.load_model(
        obs_dim, act_dim)
    estimator = InverseSystemIDEstimator(
        model_lib.one_step_evidence(model),
        nnx.state(model, nnx.Param),
        gains,
        variance,
        filter_config,
    )
    stationary = []
    switching = []
    for arm in protocol.arms(group):
        stationary.extend(_stationary(
            group, config, arm, action_groups, estimator, event_seed))
        switching.append(_switching(
            group, config, arm, action_groups, estimator, event_seed))
        print(
            f"policy-ensemble group={group} event={event_seed} "
            f"arm={arm.label} complete",
            flush=True,
        )
    result = {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": protocol.identity(group, event_seed),
        "stationary": stationary,
        "switching": switching,
    }

    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        protocol.write_json_atomic(temporary / "results.json", result)
        manifest = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(group, event_seed),
            "source_bundles": _source_records(group),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.final.MODEL_PATH),
            "result_file": protocol.file_record(temporary / "results.json"),
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(group, event_seed)
    print(f"POLICY ENSEMBLE AUDIT COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=protocol.GROUPS, required=True)
    parser.add_argument("--event-seed", type=int, choices=protocol.EVENT_SEEDS,
                        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.group, args.event_seed)


if __name__ == "__main__":
    main()
