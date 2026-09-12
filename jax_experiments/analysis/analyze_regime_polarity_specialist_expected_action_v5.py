"""Analyze confirm3 routing with specialist-trajectory system ID v5."""
from __future__ import annotations

import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_expected_action_audit_v5 import (
    validate_audit,
)


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _seed_metrics(seed: int) -> dict:
    validate_audit(seed)
    events = [
        protocol.read_json(protocol.event_result(seed, event_seed))
        for event_seed in protocol.AUDIT_EVENT_SEEDS
    ]
    arms = {}
    for arm in protocol.ARMS:
        rows = [event["switching"][arm] for event in events]
        arms[arm] = {
            "mean": _mean(row["return_mean"] for row in rows),
            "event_returns": {
                str(event_seed): float(row["return_mean"])
                for event_seed, row in zip(protocol.AUDIT_EVENT_SEEDS, rows)
            },
            "terminated_rate": _mean(row["terminated_rate"] for row in rows),
            "fallback_action_fraction": _mean(
                row["fallback_action_fraction"] for row in rows),
            "adaptive_mode_accuracy": _mean(
                row["adaptive_mode_accuracy"]
                for row in rows
                if row["adaptive_mode_accuracy"] is not None
            ) if any(
                row["adaptive_mode_accuracy"] is not None for row in rows
            ) else None,
        }
        option_rows = [
            row for row in rows if "option_switch_count_mean" in row]
        if option_rows:
            arms[arm]["option_switch_count_mean"] = _mean(
                row["option_switch_count_mean"] for row in option_rows)

    robust = arms["robust_sac"]
    oracle = arms["dynamic_oracle"]
    for row in arms.values():
        row["relative_gain"] = (
            (row["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf")
        )
        row["oracle_recovery"] = (
            (row["mean"] - robust["mean"])
            / (oracle["mean"] - robust["mean"])
            if oracle["mean"] > robust["mean"] else float("-inf")
        )
        row["event_wins_vs_robust"] = sum(
            row["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.AUDIT_EVENT_SEEDS
        )
    return {"arms": arms}


def _passes(row: dict) -> bool:
    return bool(
        row["relative_gain"] >= protocol.MIN_GAIN
        and row["oracle_recovery"] >= protocol.MIN_ORACLE_RECOVERY
        and row["mean"] >= protocol.MIN_SWITCHING_RETURN
        and row["terminated_rate"] == 0.0
        and row["event_wins_vs_robust"] == len(protocol.AUDIT_EVENT_SEEDS)
    )


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed)
        for seed in protocol.AUDIT_SOURCE_SEEDS
    }
    seed_pass = {
        str(seed): _passes(
            seeds[str(seed)]["arms"][protocol.PRIMARY_ARM])
        for seed in protocol.AUDIT_SOURCE_SEEDS
    }
    primary_pass = all(seed_pass.values())
    validation_bank_pass = all(
        seed_pass[str(seed)] for seed in protocol.VALIDATION_SOURCE_SEEDS)
    if primary_pass and validation_bank_pass:
        diagnosis = (
            "specialist-trajectory system ID closes the frozen confirm3 loop"
        )
        next_step = "freeze v5 and train fresh independent source banks"
    elif validation_bank_pass:
        diagnosis = (
            "the held-out policy bank passes, but the development bank is unstable"
        )
        next_step = "repeat estimator fitting with a second training policy bank"
    else:
        diagnosis = (
            "specialist-trajectory fitting does not recover enough oracle headroom"
        )
        next_step = "stop estimator-only changes and train switch-matched control"
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "train_source_seeds": list(protocol.TRAIN_SOURCE_SEEDS),
        "validation_source_seeds": list(protocol.VALIDATION_SOURCE_SEEDS),
        "audit_source_seeds": list(protocol.AUDIT_SOURCE_SEEDS),
        "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "primary_arm": protocol.PRIMARY_ARM,
        "source_seed_metrics": seeds,
        "source_seed_pass": seed_pass,
        "validation_bank_pass": validation_bank_pass,
        "primary_pass": primary_pass,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def report(payload: dict) -> str:
    lines = [
        "# Specialist-trajectory expected-action system ID v5",
        "",
        "The confirm3 router is frozen from v3. Only the expected-action model "
        "and posterior filter are fitted on independent-specialist trajectories.",
        "",
    ]
    for seed in protocol.AUDIT_SOURCE_SEEDS:
        rows = payload["source_seed_metrics"][str(seed)]["arms"]
        split = (
            "training policy bank"
            if seed in protocol.TRAIN_SOURCE_SEEDS
            else "held-out policy bank"
        )
        lines += [
            f"## Source seed {seed} ({split})",
            "",
            "| Arm | Return | Gain | Oracle recovery | Robust actions | Mode accuracy | Event wins |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in protocol.ARMS:
            row = rows[arm]
            accuracy = row.get("adaptive_mode_accuracy")
            accuracy_text = "-" if accuracy is None else f"{100*accuracy:.2f}%"
            lines.append(
                f"| `{arm}` | {row['mean']:.1f} | "
                f"{100*row['relative_gain']:.1f}% | "
                f"{100*row['oracle_recovery']:.1f}% | "
                f"{100*row['fallback_action_fraction']:.2f}% | "
                f"{accuracy_text} | "
                f"{row['event_wins_vs_robust']}/{len(protocol.AUDIT_EVENT_SEEDS)} |"
            )
        lines += [
            "",
            f"Seed pass: **{payload['source_seed_pass'][str(seed)]}**",
            "",
        ]
    lines += [
        f"Held-out policy-bank pass: **{payload['validation_bank_pass']}**",
        "",
        f"Primary pass: **{payload['primary_pass']}**",
        "",
        f"Diagnosis: {payload['diagnosis']}.",
        "",
        f"Decision: {payload['next_step']}.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "SPECIALIST EXPECTED-ACTION ANALYSIS COMPLETE: "
        f"validation={payload['validation_bank_pass']} "
        f"primary={payload['primary_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
