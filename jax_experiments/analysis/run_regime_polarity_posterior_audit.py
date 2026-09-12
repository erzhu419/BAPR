"""Audit one unseen policy seed with the frozen causal posterior."""
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

from jax_experiments.analysis import regime_polarity_confirmation as confirmation
from jax_experiments.analysis import regime_polarity_posterior as protocol
from jax_experiments.analysis import regime_polarity_posterior_model as model_lib
from jax_experiments.analysis import train_regime_polarity_posterior as trainer
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


ARMS = ("robust", "oracle", "learned_soft", "learned_map")
LEARNED_ARMS = ("learned_soft", "learned_map")


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": int(seed),
        "event_seeds": list(protocol.TEST_EVENT_SEEDS),
        "arms": list(ARMS),
        "benchmark_role": "unseen_frozen_posterior_screen",
    }


def _policy_action_fn(agent):
    graphdef = nnx.graphdef(agent.policy)

    @jax.jit
    def action(params, observation, context):
        policy = nnx.merge(graphdef, params)
        return policy.deterministic(
            jnp.asarray(observation)[None],
            jnp.asarray(context)[None],
        )[0]

    return action


def _arm_context(
    arm: str,
    physical_mode: int,
    posterior: np.ndarray,
) -> np.ndarray:
    if arm == "robust":
        return np.zeros((len(protocol.MODES),), dtype=np.float32)
    if arm == "oracle":
        return np.eye(
            len(protocol.MODES), dtype=np.float32)[int(physical_mode)]
    if arm == "learned_soft":
        return np.asarray(posterior, dtype=np.float32)
    if arm == "learned_map":
        return np.eye(len(protocol.MODES), dtype=np.float32)[
            int(np.argmax(posterior))]
    raise ValueError(f"unknown posterior audit arm {arm!r}")


def _posterior_step(
    emission_fn,
    model_params,
    posterior: np.ndarray,
    filter_config: protocol.FilterConfig,
    obs,
    action,
    reward,
    next_obs,
):
    log_likelihood, aleatoric, epistemic = emission_fn(
        model_params,
        jnp.asarray(obs),
        jnp.asarray(action),
        jnp.asarray(reward),
        jnp.asarray(next_obs),
    )
    evidence = np.asarray(log_likelihood, dtype=np.float64)
    next_posterior = protocol.posterior_update(
        posterior, evidence, filter_config)
    return (
        next_posterior,
        evidence,
        np.asarray(aleatoric, dtype=np.float64),
        np.asarray(epistemic, dtype=np.float64),
    )


class ForwardHMMEstimator:
    """Adapter exposing the frozen v1 posterior through the audit interface."""

    def __init__(
        self,
        emission_fn,
        model_params,
        filter_config,
    ):
        self.emission_fn = emission_fn
        self.model_params = model_params
        self.filter_config = filter_config

    def initial_state(self):
        return np.full(
            (len(protocol.MODES),),
            1.0 / len(protocol.MODES),
            dtype=np.float64,
        )

    @staticmethod
    def probabilities(state):
        return np.asarray(state, dtype=np.float64)

    def step(self, state, obs, action, reward, next_obs):
        return _posterior_step(
            self.emission_fn,
            self.model_params,
            state,
            self.filter_config,
            obs,
            action,
            reward,
            next_obs,
        )


def _strict_stationary(
    config,
    tasks,
    arm: str,
    policy_state,
    action_fn,
    estimator,
    event_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    metric_rows = []
    for mode in protocol.MODES:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            run_config,
            seed_offset=int(event_seed) - int(config.seed),
        )
        mode_tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(mode_tasks)
        env.set_task(mode_tasks[mode])
        returns = []
        terminated = []
        arm_posteriors = []
        labels = []
        for episode in range(confirmation.EPISODES_PER_TASK):
            obs = env.reset()
            estimator_state = estimator.initial_state()
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                posterior = estimator.probabilities(estimator_state)
                context = _arm_context(arm, mode, posterior)
                action = np.asarray(
                    action_fn(policy_state, obs, context))
                next_obs, reward, done, info = env.step(action)
                if int(info["mode_used"]) != int(mode):
                    raise ValueError("stationary physics mode changed")
                if arm in LEARNED_ARMS:
                    arm_posteriors.append(posterior.copy())
                    labels.append(int(mode))
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state,
                        obs,
                        action,
                        reward,
                        next_obs,
                    )
                episode_return += float(reward)
                obs = next_obs
                if done:
                    episode_terminated = True
                    obs = env.reset()
                    if arm in LEARNED_ARMS:
                        estimator_state = estimator.initial_state()
            returns.append(episode_return)
            terminated.append(float(episode_terminated))
        summaries.append({
            "arm": arm,
            "mode": int(mode),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "terminated_rate": float(np.mean(terminated)),
        })
        if arm in LEARNED_ARMS:
            metrics = protocol.posterior_metrics(
                np.asarray(arm_posteriors),
                np.asarray(labels, dtype=np.int32),
            )
            metric_rows.append({
                "arm": arm,
                "mode": int(mode),
                **metrics,
            })
        if hasattr(env, "close"):
            env.close()
    return summaries, metric_rows


def _strict_switching(
    config,
    tasks,
    arm: str,
    policy_state,
    action_fn,
    estimator,
    event_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config,
        seed_offset=int(event_seed) - int(config.seed),
    )
    switch_tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminated = []
    trace: dict[str, list[Any]] = {
        "episode": [],
        "step": [],
        "physics_mode": [],
        "posterior_before": [],
        "posterior_after": [],
        "context": [],
        "log_likelihood": [],
        "aleatoric": [],
        "epistemic": [],
        "reward": [],
        "done": [],
    }
    for episode in range(confirmation.SWITCHING_EPISODES):
        sequence, _ = _select_eval_switch_sequence(
            env, switch_tasks, episode)
        _reset_eval_switch_schedule(
            env, sequence, run_config, protocol.DWELL_STEPS)
        obs = env.reset()
        estimator_state = estimator.initial_state()
        episode_return = 0.0
        episode_terminated = False
        for step in range(protocol.MAX_EPISODE_STEPS):
            physical_mode = int(env.task_id_for_next_step())
            posterior = estimator.probabilities(estimator_state)
            context = _arm_context(arm, physical_mode, posterior)
            action = np.asarray(action_fn(policy_state, obs, context))
            next_obs, reward, done, info = env.step(action)
            if int(info["mode_used"]) != physical_mode:
                raise ValueError("switching mode label is not action-causal")
            if arm in LEARNED_ARMS:
                next_state, evidence, aleatoric, epistemic = estimator.step(
                    estimator_state,
                    obs,
                    action,
                    reward,
                    next_obs,
                )
                next_posterior = estimator.probabilities(next_state)
                trace["episode"].append(episode)
                trace["step"].append(step)
                trace["physics_mode"].append(physical_mode)
                trace["posterior_before"].append(posterior.copy())
                trace["posterior_after"].append(next_posterior.copy())
                trace["context"].append(context.copy())
                trace["log_likelihood"].append(evidence)
                trace["aleatoric"].append(aleatoric)
                trace["epistemic"].append(epistemic)
                trace["reward"].append(float(reward))
                trace["done"].append(bool(done))
                estimator_state = next_state
            episode_return += float(reward)
            obs = next_obs
            if done:
                episode_terminated = True
                # Match the strict switching audit: preserve both the physical
                # mode clock and posterior across a simulator state reset.
                obs = env.reset()
        returns.append(episode_return)
        terminated.append(float(episode_terminated))
    if hasattr(env, "close"):
        env.close()

    output = {
        key: np.asarray(value)
        for key, value in trace.items()
    }
    summary = {
        "arm": arm,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "returns": [float(value) for value in returns],
        "terminated_rate": float(np.mean(terminated)),
    }
    if arm in LEARNED_ARMS:
        summary["posterior_metrics"] = protocol.posterior_metrics(
            output["posterior_before"],
            output["physics_mode"],
        )
        selected_aleatoric = np.sum(
            output["posterior_before"] * output["aleatoric"], axis=-1)
        selected_epistemic = np.sum(
            output["posterior_before"] * output["epistemic"], axis=-1)
        summary["uncertainty"] = {
            "aleatoric_mean": float(np.mean(selected_aleatoric)),
            "aleatoric_std": float(np.std(selected_aleatoric)),
            "epistemic_mean": float(np.mean(selected_epistemic)),
            "epistemic_std": float(np.std(selected_epistemic)),
        }
    return summary, output


def _event_result(
    seed: int,
    event_seed: int,
    robust,
    oracle,
    model,
    filter_config,
    model_manifest,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    robust_config, robust_agent, robust_state = robust
    oracle_config, oracle_agent, oracle_state = oracle
    if robust_config.seed != oracle_config.seed:
        raise ValueError("paired controllers have different training seeds")
    tasks = [
        {"mode_id": mode} for mode in protocol.MODES
    ]
    action_fns = {
        "robust": _policy_action_fn(robust_agent),
        "oracle": _policy_action_fn(oracle_agent),
    }
    model_params = nnx.state(model, nnx.Param)
    emission_fn = model_lib.one_step_emission(model)
    estimator = ForwardHMMEstimator(
        emission_fn, model_params, filter_config)
    stationary = []
    stationary_metrics = []
    switching = []
    traces = {}
    for arm in ARMS:
        source = "robust" if arm == "robust" else "oracle"
        agent = robust_agent if source == "robust" else oracle_agent
        state = robust_state if source == "robust" else oracle_state
        config = robust_config if source == "robust" else oracle_config
        action_fn = action_fns[source]
        arm_stationary, arm_metrics = _strict_stationary(
            config,
            tasks,
            arm,
            state,
            action_fn,
            estimator,
            event_seed,
        )
        arm_switching, trace = _strict_switching(
            config,
            tasks,
            arm,
            state,
            action_fn,
            estimator,
            event_seed,
        )
        stationary.extend(arm_stationary)
        stationary_metrics.extend(arm_metrics)
        switching.append(arm_switching)
        if arm in LEARNED_ARMS:
            traces[arm] = trace
    return {
        "schema": "bapr.regime-polarity-posterior-event.v1",
        "status": "complete",
        "training_seed": int(seed),
        "event_seed": int(event_seed),
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "dwell_steps": protocol.DWELL_STEPS,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "episodes_per_stationary_mode": confirmation.EPISODES_PER_TASK,
        "switching_episodes": confirmation.SWITCHING_EPISODES,
        "filter_config": filter_config.to_dict(),
        "model_parameter_file": model_manifest["parameter_file"],
        "stationary": stationary,
        "stationary_posterior_metrics": stationary_metrics,
        "switching": switching,
    }, traces


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_audit(seed: int) -> dict[str, Any]:
    seed = int(seed)
    destination = protocol.audit_dir(seed)
    payload = protocol.read_json(destination / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _identity(seed)
            or payload.get("model_manifest")
            != protocol.file_record(protocol.MODEL_MANIFEST)
            or payload.get("model_parameters")
            != protocol.file_record(protocol.MODEL_PATH)):
        raise ValueError(f"invalid posterior audit: {destination}")
    expected = {
        f"event_seed_{event_seed}/results.json"
        for event_seed in protocol.TEST_EVENT_SEEDS
    } | {
        f"event_seed_{event_seed}/{arm}_trace.npz"
        for event_seed in protocol.TEST_EVENT_SEEDS
        for arm in LEARNED_ARMS
    }
    records = payload.get("files") or {}
    if set(records) != expected:
        raise ValueError(f"incomplete posterior audit: {destination}")
    for relative, record in records.items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"posterior audit file changed: {path}")
    for event_seed in protocol.TEST_EVENT_SEEDS:
        result = protocol.read_json(
            destination / f"event_seed_{event_seed}/results.json")
        if (result.get("schema")
                != "bapr.regime-polarity-posterior-event.v1"
                or result.get("status") != "complete"
                or result.get("training_seed") != seed
                or result.get("event_seed") != event_seed
                or {row["arm"] for row in result["switching"]} != set(ARMS)
                or len(result["stationary"])
                != len(ARMS) * len(protocol.MODES)):
            raise ValueError("invalid posterior event result")
        for arm in LEARNED_ARMS:
            with np.load(
                    destination
                    / f"event_seed_{event_seed}/{arm}_trace.npz",
                    allow_pickle=False) as trace:
                if (trace["posterior_before"].shape
                        != (
                            confirmation.SWITCHING_EPISODES
                            * protocol.MAX_EPISODE_STEPS,
                            len(protocol.MODES),
                        )):
                    raise ValueError("posterior trace has the wrong horizon")
                if not np.allclose(
                        np.sum(trace["posterior_before"], axis=-1),
                        1.0,
                        atol=1e-5):
                    raise ValueError("posterior trace is not normalized")
    return payload


def run(seed: int) -> None:
    seed = int(seed)
    if seed not in protocol.TEST_CONTROLLER_SEEDS:
        raise ValueError(f"unknown posterior audit seed {seed}")
    destination = protocol.audit_dir(seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"POLARITY POSTERIOR AUDIT ALREADY COMPLETE: {destination}")
            return

    robust = trainer._load_controller(confirmation, seed, "robust")
    oracle = trainer._load_controller(confirmation, seed, "oracle")
    model, filter_config, model_manifest = model_lib.load_model(
        robust[1].obs_dim, robust[1].act_dim)

    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for event_seed in protocol.TEST_EVENT_SEEDS:
            result, traces = _event_result(
                seed,
                event_seed,
                robust,
                oracle,
                model,
                filter_config,
                model_manifest,
            )
            event_dir = temporary / f"event_seed_{event_seed}"
            protocol.write_json_atomic(event_dir / "results.json", result)
            relative = f"event_seed_{event_seed}/results.json"
            records[relative] = protocol.file_record(temporary / relative)
            for arm, trace in traces.items():
                relative = f"event_seed_{event_seed}/{arm}_trace.npz"
                _write_npz(temporary / relative, trace)
                records[relative] = protocol.file_record(temporary / relative)
            print(
                f"posterior audit seed={seed} event={event_seed} complete",
                flush=True,
            )
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "model_manifest": protocol.file_record(protocol.MODEL_MANIFEST),
            "model_parameters": protocol.file_record(protocol.MODEL_PATH),
            "controller_bundles": {
                role: protocol.file_record(
                    confirmation.bundle_manifest(
                        protocol.ENV, role, seed))
                for role in protocol.ROLES
            },
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"POLARITY POSTERIOR AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        choices=protocol.TEST_CONTROLLER_SEEDS,
        type=int,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    run(args.seed)


if __name__ == "__main__":
    main()
