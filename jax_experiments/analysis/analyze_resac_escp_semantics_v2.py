"""Aggregate corrected RE-SAC/ESCP numerical-semantics diagnostics."""
from __future__ import annotations

import csv
import math
import statistics
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import resac_escp_semantics_v2 as protocol
from jax_experiments.analysis.final_task_sweep import load_config
from jax_experiments.analysis.run_escp_first_nonfinite_probe_v1 import (
    validate as validate_probe,
)
from jax_experiments.analysis.run_resac_escp_semantics_audit_v2 import (
    validate_audit,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return float(statistics.fmean(values))


def _sd(values) -> float:
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _event_metrics(env: str, role: str, seed: int, event_seed: int) -> dict:
    rows = _rows(
        protocol.audit_event_dir(env, role, seed, event_seed) / "summary.csv")
    stationary = [
        row for row in rows
        if row["metric_group"] == "stationary" and row["split"] == "test"
    ]
    switching = [row for row in rows if row["metric_group"] == "switching"]
    if len(stationary) != 1 or len(switching) != 1:
        raise ValueError("semantics audit summary has ambiguous rows")
    return {
        "stationary_ood": float(stationary[0]["return_mean"]),
        "stationary_termination": float(
            stationary[0]["terminated_rate_mean"]),
        "switching": float(switching[0]["switch_return_mean"]),
        "switching_termination_count": float(
            switching[0]["termination_count_mean"]),
    }


def _seed_metrics(env: str, role: str, seed: int) -> dict:
    events = {
        event_seed: _event_metrics(env, role, seed, event_seed)
        for event_seed in protocol.EVENT_SEEDS
    }
    return {
        "event_metrics": {str(key): value for key, value in events.items()},
        "stationary_ood": _mean(
            value["stationary_ood"] for value in events.values()),
        "stationary_termination": _mean(
            value["stationary_termination"] for value in events.values()),
        "switching": _mean(value["switching"] for value in events.values()),
        "switching_termination_count": _mean(
            value["switching_termination_count"]
            for value in events.values()),
    }


def _role_summary(seed_rows: dict[int, dict]) -> dict:
    output = {}
    for key in (
        "stationary_ood",
        "stationary_termination",
        "switching",
        "switching_termination_count",
    ):
        values = [float(seed_rows[seed][key]) for seed in protocol.TRAINING_SEEDS]
        output[key] = {"mean": _mean(values), "sd": _sd(values)}
    return output


def _load_agent_diagnostics(env: str, role: str, seed: int) -> dict:
    run_dir = protocol.bundle_dir(env, role, seed)
    config = load_config(run_dir)
    environment = make_env(config, seed_offset=0)
    agent = make_algo(config.algo, environment.obs_dim, environment.act_dim, config)
    replay = ReplayBuffer(
        environment.obs_dim, environment.act_dim, capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0))
    with tempfile.TemporaryDirectory() as temporary:
        start_iter, total_steps = load_checkpoint(
            str(run_dir / "checkpoints"), agent, replay, Logger(temporary),
            config.algo, load_replay_buffer=False)
    obs = jnp.zeros((1, environment.obs_dim), dtype=jnp.float32)
    action = jnp.zeros((1, environment.act_dim), dtype=jnp.float32)
    context = None
    critic_obs = obs
    if hasattr(agent, "context_net") and role == "escp":
        context = agent.context_net(obs)
        critic_obs = jnp.concatenate([obs, context], axis=-1)
    q_values = np.asarray(agent.critic(critic_obs, action), dtype=np.float64)
    policy = (
        agent.ema_policy
        if bool(getattr(config, "use_ema_eval", False))
        and hasattr(agent, "ema_policy") else agent.policy)
    mean, log_std = policy(obs, context)
    reg_norm = None
    if hasattr(agent.target_critic, "compute_reg_norm"):
        reg_norm = np.asarray(
            agent.target_critic.compute_reg_norm(), dtype=np.float64)
    return {
        "checkpoint_next_iter": int(start_iter),
        "checkpoint_total_steps": int(total_steps),
        "q_zero_mean": float(np.mean(q_values)),
        "q_zero_std": float(np.std(q_values)),
        "q_zero_max_abs": float(np.max(np.abs(q_values))),
        "actor_zero_mean_abs": float(np.mean(np.abs(np.asarray(mean)))),
        "actor_zero_log_std_mean": float(np.mean(np.asarray(log_std))),
        "target_raw_l1_mean": (
            None if reg_norm is None else float(np.mean(reg_norm))),
        "target_raw_l1_max": (
            None if reg_norm is None else float(np.max(reg_norm))),
    }


def _training_log_diagnostics(env: str, role: str, seed: int) -> dict | None:
    if role == "sac":
        return None
    log_dir = protocol.bundle_dir(env, role, seed) / "logs"
    output = {}
    for name in ("q_mean", "q_std_mean", "critic_loss", "policy_loss", "alpha"):
        values = np.asarray(np.load(log_dir / f"{name}.npy"), dtype=np.float64)
        output[name] = {
            "count": int(values.size),
            "all_finite": bool(np.all(np.isfinite(values))),
            "last": float(values[-1]),
            "max_abs": float(np.max(np.abs(values))),
        }
    if role == "resac":
        bonuses = np.asarray(
            np.load(log_dir / "reg_bonus_mean.npy"), dtype=np.float64)
        output["reg_bonus_mean"] = {
            "all_zero": bool(np.all(bonuses == 0.0)),
            "max_abs": float(np.max(np.abs(bonuses))),
        }
    else:
        finite = np.asarray(
            np.load(log_dir / "finite_update_rate.npy"), dtype=np.float64)
        output["finite_update_rate"] = {
            "all_one": bool(np.all(finite == 1.0)),
            "minimum": float(np.min(finite)),
        }
    return output


def _analyze_env(env: str) -> dict:
    by_seed = {}
    checkpoint_diagnostics = {}
    training_logs = {}
    for seed in protocol.TRAINING_SEEDS:
        by_seed[seed] = {}
        checkpoint_diagnostics[seed] = {}
        training_logs[seed] = {}
        for role in protocol.ROLES:
            validate_audit(env, role, seed)
            by_seed[seed][role] = _seed_metrics(env, role, seed)
            checkpoint_diagnostics[seed][role] = _load_agent_diagnostics(
                env, role, seed)
            training_logs[seed][role] = _training_log_diagnostics(
                env, role, seed)
    summaries = {
        role: _role_summary({
            seed: by_seed[seed][role] for seed in protocol.TRAINING_SEEDS
        })
        for role in protocol.ROLES
    }
    deltas = {}
    for adaptive in protocol.TRAIN_ROLES:
        role_deltas = {}
        for metric in ("stationary_ood", "switching"):
            values = [
                float(by_seed[seed][adaptive][metric])
                - float(by_seed[seed]["sac"][metric])
                for seed in protocol.TRAINING_SEEDS
            ]
            role_deltas[metric] = {
                "per_seed": dict(zip(protocol.TRAINING_SEEDS, values)),
                "mean": _mean(values),
                "wins": sum(value > 0.0 for value in values),
            }
        deltas[f"{adaptive}_minus_sac"] = role_deltas
    numerical_checks = {
        "all_corrected_training_logs_finite": all(
            all(
                metrics[metric]["all_finite"]
                for metric in (
                    "q_mean", "q_std_mean", "critic_loss",
                    "policy_loss", "alpha"))
            for seed in protocol.TRAINING_SEEDS
            for role, metrics in training_logs[seed].items()
            if role in protocol.TRAIN_ROLES),
        "resac_regularization_shift_disabled": all(
            training_logs[seed]["resac"]["reg_bonus_mean"]["all_zero"]
            for seed in protocol.TRAINING_SEEDS),
        "escp_finite_guard_passed": all(
            training_logs[seed]["escp"]["finite_update_rate"]["all_one"]
            for seed in protocol.TRAINING_SEEDS),
    }
    return {
        "training_seed_metrics": {
            str(seed): by_seed[seed] for seed in protocol.TRAINING_SEEDS},
        "summaries": summaries,
        "paired_deltas": deltas,
        "checkpoint_diagnostics": {
            str(seed): value for seed, value in checkpoint_diagnostics.items()},
        "training_log_diagnostics": {
            str(seed): value for seed, value in training_logs.items()},
        "numerical_checks": numerical_checks,
        "numerical_semantics_pass": all(numerical_checks.values()),
    }


def analyze() -> dict:
    probes = {
        protocol.env_slug(env): validate_probe(env, protocol.PROBE_SEEDS[0])
        for env in protocol.ENVS
    }
    environments = {env: _analyze_env(env) for env in protocol.ENVS}
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "legacy_escp_probes": probes,
        "environments": environments,
        "global_numerical_pass": all(
            row["numerical_semantics_pass"]
            for row in environments.values()),
        "inference_scope": (
            "two-seed implementation diagnostic; not a confirmatory paper result"),
        "provenance_findings": [
            "The earlier v1 smoke used positive weight_reg=0.01 and pure "
            "independent targets, so it was not the completed MuJoCo B0 artifact.",
            "The released MuJoCo B0 launcher disables weight_reg and beta_ood "
            "and uses an independent/min blend, EMA evaluation, and anchoring.",
            "The positive bus regularization sign remains unchanged; this "
            "MuJoCo protocol sets its coefficient to zero.",
            "The corrected ESCP arm is still a state-only JAX approximation, "
            "not the original recurrent (state,last_action) environment probe.",
        ],
    }


def _fmt(summary: dict, metric: str) -> str:
    return f"{summary[metric]['mean']:.1f} +/- {summary[metric]['sd']:.1f}"


def report(payload: dict) -> str:
    lines = [
        "# RE-SAC / ESCP numerical-semantics audit",
        "",
        "This protocol replaces the invalid v1 `paper-fidelity` label. It is "
        "a two-seed implementation diagnostic at the sealed v1 SAC budget, "
        "not an 8M-step confirmatory reproduction.",
        "",
        "## Provenance correction",
        "",
    ]
    lines.extend(f"- {finding}" for finding in payload["provenance_findings"])
    lines += ["", "## Legacy ESCP failure", ""]
    for slug, probe in payload["legacy_escp_probes"].items():
        first = probe.get("first_nonfinite")
        if first is None:
            lines.append(
                f"- {slug}: finite through {probe['max_iters']} iterations.")
        else:
            lines.append(
                f"- {slug}: first non-finite update at iteration "
                f"{first['iteration']}, global update "
                f"{first['global_update']} (scan index "
                f"{first['scan_index']}).")
    for env in protocol.ENVS:
        row = payload["environments"][env]
        lines += [
            "",
            f"## {env}",
            "",
            "| Method | Stationary OOD | Switching |",
            "|---|---:|---:|",
        ]
        for role in protocol.ROLES:
            summary = row["summaries"][role]
            lines.append(
                f"| {role.upper()} | {_fmt(summary, 'stationary_ood')} | "
                f"{_fmt(summary, 'switching')} |")
        lines += ["", "Paired differences against the sealed SAC reference:"]
        for name, metrics in row["paired_deltas"].items():
            lines.append(
                f"- `{name}`: stationary "
                f"{metrics['stationary_ood']['mean']:+.1f} "
                f"({metrics['stationary_ood']['wins']}/2 wins), switching "
                f"{metrics['switching']['mean']:+.1f} "
                f"({metrics['switching']['wins']}/2 wins).")
        lines += [
            "",
            f"Numerical semantics pass: **{row['numerical_semantics_pass']}**",
        ]
    lines += [
        "",
        "## Decision boundary",
        "",
        f"Global numerical pass: **{payload['global_numerical_pass']}**.",
        "",
        "A numerical pass only makes the corrected rows interpretable. It does "
        "not establish that ESCP or RE-SAC beats SAC, and it does not make the "
        "state-only ESCP approximation architecture-faithful. A fresh 8M-step, "
        "five-seed run is warranted only after this diagnostic is finite and "
        "its Q/actor scales are plausible.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "RE-SAC/ESCP SEMANTICS ANALYSIS COMPLETE: "
        f"numerical_pass={payload['global_numerical_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
