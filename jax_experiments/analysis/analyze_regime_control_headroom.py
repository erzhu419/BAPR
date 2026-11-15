"""Aggregate the preregistered equal-budget regime-control headroom test."""
from __future__ import annotations

import csv
import math
import statistics

from jax_experiments.analysis import regime_control_headroom as protocol
from jax_experiments.analysis.run_regime_control_headroom_audit import (
    validate_audit,
)


T_CRITICAL_95_DF4 = 2.7764451051977987
MIN_RELATIVE_GAIN = 0.10
MAX_TERMINATION_GAP = 0.05
MIN_MODE_WINS = 3
MIN_PASSING_ENVS = 2


def _rows(path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _sd(values) -> float:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _indicator(value) -> float:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    return float(value)


def _paired_ci(values: list[float]) -> dict[str, float | int]:
    if len(values) != len(protocol.TRAINING_SEEDS):
        raise ValueError(
            f"paired inference requires {len(protocol.TRAINING_SEEDS)} "
            f"training seeds, got {len(values)}")
    mean = _mean(values)
    sd = _sd(values)
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return {
        "n_training_seeds": len(values),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "wins": sum(value > 0.0 for value in values),
    }


def _task_identity(row: dict[str, str]):
    return tuple(sorted(
        (key, value) for key, value in row.items()
        if key.endswith(("_mean", "_min", "_max"))
        and not key.startswith(("return_", "steps_"))))


def _load_event(env: str, role: str, seed: int, event_seed: int):
    directory = protocol.audit_event_dir(env, role, seed, event_seed)
    task_rows = _rows(directory / "task_returns.csv")
    switching_rows = _rows(directory / "switching_returns.csv")
    task_rows.sort(key=lambda row: int(float(row["mode_id_mean"])))
    switching_rows.sort(key=lambda row: int(row["episode"]))
    mode_returns = {
        int(float(row["mode_id_mean"])): float(row["return_mean"])
        for row in task_rows
    }
    if set(mode_returns) != set(protocol.MODES):
        raise ValueError(
            f"missing stationary modes for {env}/{role}/seed{seed}/"
            f"event{event_seed}")
    return {
        "task_identity": tuple(_task_identity(row) for row in task_rows),
        "switch_identity": tuple(
            row["switch_sequence_source_indices"]
            for row in switching_rows),
        "mode_returns": mode_returns,
        "stationary_termination": _mean(
            float(row["terminated_rate"]) for row in task_rows),
        "switching_returns": [
            float(row["return"]) for row in switching_rows],
        "switching_termination": [
            _indicator(row["terminated"]) for row in switching_rows],
    }


def _load_all():
    data = {}
    for env in protocol.ENVS:
        data[env] = {}
        for seed in protocol.TRAINING_SEEDS:
            data[env][seed] = {}
            for role in protocol.ROLES:
                validate_audit(env, role, seed)
                data[env][seed][role] = {
                    event_seed: _load_event(
                        env, role, seed, event_seed)
                    for event_seed in protocol.AUDIT_EVENT_SEEDS
                }
            for event_seed in protocol.AUDIT_EVENT_SEEDS:
                robust = data[env][seed]["robust"][event_seed]
                oracle = data[env][seed]["oracle"][event_seed]
                if (robust["task_identity"] != oracle["task_identity"]
                        or robust["switch_identity"]
                        != oracle["switch_identity"]):
                    raise ValueError(
                        "robust/oracle audit streams are not paired: "
                        f"{env}/seed{seed}/event{event_seed}")
    return data


def _seed_metrics(events) -> dict[str, object]:
    mode_returns = {
        mode: _mean(
            event["mode_returns"][mode] for event in events.values())
        for mode in protocol.MODES
    }
    switching = [
        value
        for event in events.values()
        for value in event["switching_returns"]
    ]
    switching_termination = [
        value
        for event in events.values()
        for value in event["switching_termination"]
    ]
    return {
        "stationary_by_mode": mode_returns,
        "stationary_mean": _mean(mode_returns.values()),
        "stationary_worst": min(mode_returns.values()),
        "stationary_termination": _mean(
            event["stationary_termination"] for event in events.values()),
        "switching_mean": _mean(switching),
        "switching_termination": _mean(switching_termination),
    }


def _role_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = (
        "stationary_mean",
        "stationary_worst",
        "stationary_termination",
        "switching_mean",
        "switching_termination",
    )
    payload = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        payload[f"{key}_mean"] = _mean(values)
        payload[f"{key}_sd"] = _sd(values)
    payload["stationary_by_mode"] = {
        str(mode): _mean(
            float(row["stationary_by_mode"][mode]) for row in rows)
        for mode in protocol.MODES
    }
    return payload


def _analyze_env(env_data) -> dict[str, object]:
    by_seed = {}
    for seed in protocol.TRAINING_SEEDS:
        by_seed[seed] = {
            role: _seed_metrics(env_data[seed][role])
            for role in protocol.ROLES
        }
    summaries = {
        role: _role_summary([
            by_seed[seed][role] for seed in protocol.TRAINING_SEEDS])
        for role in protocol.ROLES
    }
    switching_delta = [
        float(by_seed[seed]["oracle"]["switching_mean"])
        - float(by_seed[seed]["robust"]["switching_mean"])
        for seed in protocol.TRAINING_SEEDS
    ]
    worst_delta = [
        float(by_seed[seed]["oracle"]["stationary_worst"])
        - float(by_seed[seed]["robust"]["stationary_worst"])
        for seed in protocol.TRAINING_SEEDS
    ]
    stationary_delta = [
        float(by_seed[seed]["oracle"]["stationary_mean"])
        - float(by_seed[seed]["robust"]["stationary_mean"])
        for seed in protocol.TRAINING_SEEDS
    ]
    mode_deltas = {
        mode: [
            float(by_seed[seed]["oracle"]["stationary_by_mode"][mode])
            - float(by_seed[seed]["robust"]["stationary_by_mode"][mode])
            for seed in protocol.TRAINING_SEEDS
        ]
        for mode in protocol.MODES
    }
    switching_ci = _paired_ci(switching_delta)
    worst_ci = _paired_ci(worst_delta)
    stationary_ci = _paired_ci(stationary_delta)
    robust_switch = float(summaries["robust"]["switching_mean_mean"])
    robust_worst = float(summaries["robust"]["stationary_worst_mean"])
    switching_relative = (
        float(switching_ci["mean"]) / max(abs(robust_switch), 100.0))
    worst_relative = (
        float(worst_ci["mean"]) / max(abs(robust_worst), 100.0))
    mode_wins = sum(_mean(values) > 0.0 for values in mode_deltas.values())
    termination_gap = (
        float(summaries["oracle"]["switching_termination_mean"])
        - float(summaries["robust"]["switching_termination_mean"]))
    relative_gain_label = int(round(100.0 * MIN_RELATIVE_GAIN))
    checks = {
        f"switching_relative_gain_at_least_{relative_gain_label}pct": (
            switching_relative >= MIN_RELATIVE_GAIN),
        "switching_paired_ci_positive": (
            float(switching_ci["ci95_low"]) > 0.0),
        f"worst_mode_relative_gain_at_least_{relative_gain_label}pct": (
            worst_relative >= MIN_RELATIVE_GAIN),
        "worst_mode_paired_ci_positive": (
            float(worst_ci["ci95_low"]) > 0.0),
        "stationary_modes_improved_at_least_3_of_4": (
            mode_wins >= MIN_MODE_WINS),
        "switching_termination_gap_at_most_5pp": (
            termination_gap <= MAX_TERMINATION_GAP),
    }
    return {
        "training_seed_metrics": {
            str(seed): by_seed[seed] for seed in protocol.TRAINING_SEEDS},
        "summaries": summaries,
        "paired_deltas": {
            "switching": switching_ci,
            "stationary_mean": stationary_ci,
            "stationary_worst": worst_ci,
            "stationary_by_mode": {
                str(mode): _paired_ci(values)
                for mode, values in mode_deltas.items()
            },
        },
        "switching_relative_gain": switching_relative,
        "worst_mode_relative_gain": worst_relative,
        "mode_wins": mode_wins,
        "switching_termination_gap": termination_gap,
        "gate_checks": checks,
        "env_gate_pass": all(checks.values()),
    }


def _fmt_score(row: dict[str, object], key: str) -> str:
    return f"{float(row[key + '_mean']):.1f} +/- {float(row[key + '_sd']):.1f}"


def _fmt_ci(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean']):+.1f} "
        f"[{float(row['ci95_low']):+.1f}, "
        f"{float(row['ci95_high']):+.1f}]")


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Equal-budget regime-control headroom result",
        "",
        "The inferential unit is the independent training seed "
        f"(`n={len(protocol.TRAINING_SEEDS)}`). "
        f"{len(protocol.AUDIT_EVENT_SEEDS)} sealed event seeds are averaged "
        "inside each training seed. "
        "The oracle receives the true current persistent mode; the robust "
        "arm receives an all-zero vector through the same architecture.",
        "",
        "| Env | Robust switch | Oracle switch | Switch delta 95% CI | "
        "Rel. | Robust worst | Oracle worst | Worst delta 95% CI | "
        "Rel. | Mode wins | Term gap | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for env in protocol.ENVS:
        row = payload["environments"][env]
        robust = row["summaries"]["robust"]
        oracle = row["summaries"]["oracle"]
        lines.append(
            f"| {protocol.env_slug(env)} | "
            f"{_fmt_score(robust, 'switching_mean')} | "
            f"{_fmt_score(oracle, 'switching_mean')} | "
            f"{_fmt_ci(row['paired_deltas']['switching'])} | "
            f"{100.0 * row['switching_relative_gain']:.1f}% | "
            f"{_fmt_score(robust, 'stationary_worst')} | "
            f"{_fmt_score(oracle, 'stationary_worst')} | "
            f"{_fmt_ci(row['paired_deltas']['stationary_worst'])} | "
            f"{100.0 * row['worst_mode_relative_gain']:.1f}% | "
            f"{row['mode_wins']}/4 | "
            f"{100.0 * row['switching_termination_gap']:+.1f} pp | "
            f"{'yes' if row['env_gate_pass'] else 'no'} |")
    gate = payload["learned_estimator_gate"]
    lines += [
        "",
        "## Preregistered decision",
        "",
        f"Passing environments: "
        f"**{gate['passing_envs']}/{len(protocol.ENVS)}**. "
        f"Learned-estimator gate: **{gate['status']}**.",
        "",
        "An environment passes only when switching and worst-mode gains are "
        f"both at least {100.0 * MIN_RELATIVE_GAIN:.0f}%, both paired 95% "
        "intervals are above zero, at "
        "least 3/4 stationary modes improve, and switching termination does "
        "not increase by more than 5 percentage points.",
        "",
        ("Proceed to a causal learned mode estimator trained against this "
         "oracle interface."
         if gate["status"] == "PASS" else
         "Do not train another estimator on this benchmark: privileged mode "
         "information did not establish sufficient controller headroom."),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = _load_all()
    environments = {
        env: _analyze_env(data[env]) for env in protocol.ENVS
    }
    passing = sum(
        bool(row["env_gate_pass"]) for row in environments.values())
    gate_status = "PASS" if passing >= MIN_PASSING_ENVS else "FAIL"
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "family": protocol.FAMILY,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "audit_event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "budget": {
            "environment_steps": protocol.FINAL_TOTAL_STEPS,
            "gradient_updates": protocol.FINAL_UPDATE_COUNT,
        },
        "environments": environments,
        "learned_estimator_gate": {
            "status": gate_status,
            "passing_envs": passing,
            "required_passing_envs": MIN_PASSING_ENVS,
        },
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(
        protocol.analysis_markdown(), _render_markdown(payload))
    print(
        f"LEARNED_ESTIMATOR_GATE={gate_status} "
        f"passing_envs={passing}/{len(protocol.ENVS)}")


if __name__ == "__main__":
    main()
