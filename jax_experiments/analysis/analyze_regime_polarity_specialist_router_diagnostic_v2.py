"""Analyze specialist-router estimator, gate, and handoff failures."""
from __future__ import annotations

import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_router_diagnostic_v2 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_router_diagnostic_v2 import (
    validate_audit,
)


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _seed_metrics(seed: int) -> dict:
    validate_audit(seed)
    events = [
        protocol.read_json(protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    ]
    arms = {}
    for arm in protocol.ARMS:
        rows = [event["switching"][arm] for event in events]
        arms[arm] = {
            "mean": _mean(row["return_mean"] for row in rows),
            "event_returns": {
                str(event_seed): float(row["return_mean"])
                for event_seed, row in zip(protocol.EVENT_SEEDS, rows)
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
        posterior_rows = [row for row in rows if "posterior_metrics" in row]
        if posterior_rows:
            arms[arm]["posterior_mode_accuracy"] = _mean(
                row["posterior_metrics"]["mode_accuracy"]
                for row in posterior_rows
            )
        trigger_rows = [row for row in rows if "trigger_count_mean" in row]
        if trigger_rows:
            arms[arm]["trigger_count_mean"] = _mean(
                row["trigger_count_mean"] for row in trigger_rows)

    robust = arms["robust_sac"]["mean"]
    oracle = arms["dynamic_oracle"]["mean"]
    for arm, row in arms.items():
        row["oracle_recovery"] = (
            (row["mean"] - robust) / (oracle - robust)
            if oracle > robust else float("-inf")
        )
    return {"arms": arms}


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed) for seed in protocol.TRAINING_SEEDS
    }

    def both(arm: str, predicate) -> bool:
        return all(
            predicate(seeds[str(seed)]["arms"][arm])
            for seed in protocol.TRAINING_SEEDS
        )

    handoff_viable = both(
        "true_mode_robust10",
        lambda row: (
            row["oracle_recovery"] >= protocol.MIN_ORACLE_RECOVERY
            and row["terminated_rate"] == 0.0
        ),
    )
    debounced_pass = both(
        "posterior_debounced_option",
        lambda row: (
            row["oracle_recovery"] >= protocol.MIN_ORACLE_RECOVERY
            and row["mean"] >= protocol.MIN_SWITCHING_RETURN
            and row["terminated_rate"] == 0.0
        ),
    )
    if debounced_pass:
        diagnosis = "debounced persistent routing repairs the deployable path"
        next_step = "confirm the frozen debounced option on fresh source banks"
    elif handoff_viable:
        diagnosis = (
            "specialist handoff is viable; estimator distribution shift remains"
        )
        next_step = (
            "fit a specialist-trajectory estimator, then rerun the frozen router"
        )
    else:
        diagnosis = "independent specialists are not robust to causal handoff states"
        next_step = (
            "train switch-matched specialists from robust-state handoff curricula"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "training_seed_metrics": seeds,
        "handoff_viable": handoff_viable,
        "debounced_router_pass": debounced_pass,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def report(payload: dict) -> str:
    lines = [
        "# Independent-specialist router failure decomposition",
        "",
        "This switching-only checkpoint audit separates estimator mode errors, "
        "one-step gate thrashing, and robust-to-specialist handoff.",
        "",
    ]
    for seed in protocol.TRAINING_SEEDS:
        rows = payload["training_seed_metrics"][str(seed)]["arms"]
        lines += [
            f"## Source seed {seed}",
            "",
            "| Arm | Return | Oracle recovery | Fallback | Mode accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in protocol.ARMS:
            row = rows[arm]
            accuracy = row.get("adaptive_mode_accuracy")
            accuracy_text = "-" if accuracy is None else f"{100*accuracy:.2f}%"
            lines.append(
                f"| `{arm}` | {row['mean']:.1f} | "
                f"{100*row['oracle_recovery']:.1f}% | "
                f"{100*row['fallback_action_fraction']:.2f}% | "
                f"{accuracy_text} |"
            )
        lines.append("")
    lines += [
        f"Handoff viable: **{payload['handoff_viable']}**",
        "",
        f"Debounced router pass: **{payload['debounced_router_pass']}**",
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
        "SPECIALIST DIAGNOSTIC ANALYSIS COMPLETE: "
        f"handoff={payload['handoff_viable']} "
        f"debounced={payload['debounced_router_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
