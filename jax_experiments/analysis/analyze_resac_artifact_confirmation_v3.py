"""Aggregate the five-seed, 8M-step RE-SAC artifact confirmation."""
from __future__ import annotations

import csv
import math
import statistics
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import resac_artifact_confirmation_v3 as protocol
from jax_experiments.analysis.final_task_sweep import load_config
from jax_experiments.analysis.run_resac_artifact_confirmation_audit_v3 import (
    validate_audit,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


HISTORICAL_ARTIFACT = {
    "HalfCheetah-v2": {
        "sac": {"mean": 4610.24, "sd": 1100.22},
        "resac": {"mean": 5327.90, "sd": 1752.92},
        "paired_delta": 717.66,
        "relative_delta_percent": 15.57,
        "paired_wins": 4,
        "paper_worst_quartile": {"sac": -266.0, "resac": 740.0},
    },
    "Ant-v2": {
        "sac": {"mean": 3912.03, "sd": 972.27},
        "resac": {"mean": 3866.37, "sd": 637.85},
        "paired_delta": -45.66,
        "relative_delta_percent": -1.17,
        "paired_wins": 2,
        "paper_worst_quartile": {"sac": 1093.0, "resac": 1208.0},
    },
}


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
    directory = protocol.audit_event_dir(env, role, seed, event_seed)
    summary = _rows(directory / "summary.csv")
    stationary = [
        row for row in summary
        if row["metric_group"] == "stationary" and row["split"] == "test"
    ]
    switching = [
        row for row in summary if row["metric_group"] == "switching"
    ]
    task_rows = _rows(directory / "task_returns.csv")
    if (len(stationary) != 1 or len(switching) != 1
            or len(task_rows) != protocol.AUDIT_TASKS):
        raise ValueError("artifact audit has ambiguous or incomplete rows")
    task_returns = sorted(float(row["return_mean"]) for row in task_rows)
    worst_count = max(1, math.ceil(len(task_returns) * 0.25))
    return {
        "stationary_ood": float(stationary[0]["return_mean"]),
        "stationary_worst_quartile": _mean(task_returns[:worst_count]),
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
    output = {
        "event_metrics": {str(key): value for key, value in events.items()},
    }
    for metric in (
        "stationary_ood",
        "stationary_worst_quartile",
        "stationary_termination",
        "switching",
        "switching_termination_count",
    ):
        output[metric] = _mean(
            value[metric] for value in events.values())
    return output


def _role_summary(seed_rows: dict[int, dict]) -> dict:
    output = {}
    for metric in (
        "stationary_ood",
        "stationary_worst_quartile",
        "stationary_termination",
        "switching",
        "switching_termination_count",
    ):
        values = [
            float(seed_rows[seed][metric])
            for seed in protocol.TRAINING_SEEDS
        ]
        output[metric] = {
            "mean": _mean(values),
            "sd": _sd(values),
            "per_seed": dict(zip(protocol.TRAINING_SEEDS, values)),
        }
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
    q_values = np.asarray(agent.critic(obs, action), dtype=np.float64)
    policy = (
        agent.ema_policy
        if bool(getattr(config, "use_ema_eval", False))
        and hasattr(agent, "ema_policy") else agent.policy)
    mean, log_std = policy(obs)
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


def _training_log_diagnostics(env: str, role: str, seed: int) -> dict:
    log_dir = protocol.bundle_dir(env, role, seed) / "logs"
    output = {}
    for name in (
        "q_mean", "q_std_mean", "critic_loss", "policy_loss", "alpha"):
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
    return output


def _analyze_env(env: str) -> dict:
    by_seed: dict[int, dict] = {}
    checkpoint_diagnostics: dict[int, dict] = {}
    training_logs: dict[int, dict] = {}
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
    paired = {}
    for metric in (
        "stationary_ood", "stationary_worst_quartile", "switching"):
        values = [
            float(by_seed[seed]["resac"][metric])
            - float(by_seed[seed]["sac"][metric])
            for seed in protocol.TRAINING_SEEDS
        ]
        paired[metric] = {
            "per_seed": dict(zip(protocol.TRAINING_SEEDS, values)),
            "mean": _mean(values),
            "sd": _sd(values),
            "wins": sum(value > 0.0 for value in values),
        }
    numerical_checks = {
        "all_training_logs_finite": all(
            metrics[metric]["all_finite"]
            for seed in protocol.TRAINING_SEEDS
            for metrics in training_logs[seed].values()
            for metric in (
                "q_mean", "q_std_mean", "critic_loss", "policy_loss", "alpha")
        ),
        "resac_regularization_shift_disabled": all(
            training_logs[seed]["resac"]["reg_bonus_mean"]["all_zero"]
            for seed in protocol.TRAINING_SEEDS),
        "all_checkpoints_exact": all(
            row["checkpoint_next_iter"] == protocol.MAX_ITERS
            and row["checkpoint_total_steps"] == protocol.FINAL_TOTAL_STEPS
            for seed_rows in checkpoint_diagnostics.values()
            for row in seed_rows.values()),
    }
    return {
        "training_seed_metrics": {
            str(seed): by_seed[seed] for seed in protocol.TRAINING_SEEDS},
        "summaries": summaries,
        "paired_resac_minus_sac": paired,
        "checkpoint_diagnostics": {
            str(seed): value for seed, value in checkpoint_diagnostics.items()},
        "training_log_diagnostics": {
            str(seed): value for seed, value in training_logs.items()},
        "numerical_checks": numerical_checks,
        "numerical_pass": all(numerical_checks.values()),
        "historical_artifact_reference": HISTORICAL_ARTIFACT[env],
    }


def analyze() -> dict:
    environments = {env: _analyze_env(env) for env in protocol.ENVS}
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "controller_steps": protocol.FINAL_TOTAL_STEPS,
        "controller_updates": protocol.FINAL_UPDATE_COUNT,
        "environments": environments,
        "global_numerical_pass": all(
            row["numerical_pass"] for row in environments.values()),
        "inference_scope": (
            "fresh five-seed confirmation under the released B0 training "
            "budget and strict held-out deterministic evaluation"),
        "historical_reference_scope": (
            "historical means are independently read from the released "
            "final_multi_task artifact; its task stream is not interchangeable "
            "with this strict held-out audit"),
    }


def _fmt(summary: dict, metric: str) -> str:
    return f"{summary[metric]['mean']:.1f} +/- {summary[metric]['sd']:.1f}"


def report(payload: dict) -> str:
    lines = [
        "# RE-SAC released-artifact B0 confirmation",
        "",
        "This is a fresh five-seed confirmation at the released controller "
        "budget: 2,000 iterations, 4,000 environment steps per iteration "
        "(8M total), and 250 updates per eligible iteration. SAC and RE-SAC "
        "share the same continuous gravity task stream.",
        "",
        "Evaluation uses three independently generated held-out 40-task "
        "streams per training seed, deterministic-mean actions, strict "
        "1,000-step horizons, and a 500-step switching sequence.",
    ]
    for env in protocol.ENVS:
        row = payload["environments"][env]
        historical = row["historical_artifact_reference"]
        lines += [
            "",
            f"## {env}",
            "",
            "| Method | Stationary OOD | Worst quartile | Switching |",
            "|---|---:|---:|---:|",
        ]
        for role in protocol.ROLES:
            summary = row["summaries"][role]
            lines.append(
                f"| {role.upper()} | {_fmt(summary, 'stationary_ood')} | "
                f"{_fmt(summary, 'stationary_worst_quartile')} | "
                f"{_fmt(summary, 'switching')} |")
        lines += ["", "Paired RE-SAC minus SAC:"]
        for metric, label in (
            ("stationary_ood", "stationary mean"),
            ("stationary_worst_quartile", "worst quartile"),
            ("switching", "switching"),
        ):
            value = row["paired_resac_minus_sac"][metric]
            lines.append(
                f"- {label}: {value['mean']:+.1f} +/- {value['sd']:.1f}; "
                f"{value['wins']}/5 seed wins.")
        lines += [
            "",
            "Released-result cross-check (different legacy evaluation stream):",
            f"- SAC {historical['sac']['mean']:.1f} +/- "
            f"{historical['sac']['sd']:.1f}; RE-SAC "
            f"{historical['resac']['mean']:.1f} +/- "
            f"{historical['resac']['sd']:.1f}.",
            f"- Paired delta {historical['paired_delta']:+.1f} "
            f"({historical['paired_wins']}/5 wins); paper worst-quartile "
            f"SAC {historical['paper_worst_quartile']['sac']:.0f}, RE-SAC "
            f"{historical['paper_worst_quartile']['resac']:.0f}.",
            "",
            f"Numerical/provenance pass: **{row['numerical_pass']}**.",
        ]
    lines += [
        "",
        "## Interpretation boundary",
        "",
        f"Global numerical pass: **{payload['global_numerical_pass']}**.",
        "",
        "The fresh strict audit is the comparison used for current claims. "
        "The released values are a provenance cross-check only; agreement or "
        "disagreement must be interpreted together with the task-stream and "
        "evaluation-protocol differences. ESCP is deliberately excluded until "
        "the original recurrent history encoder is reproduced.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "RE-SAC ARTIFACT CONFIRMATION COMPLETE: "
        f"numerical_pass={payload['global_numerical_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
