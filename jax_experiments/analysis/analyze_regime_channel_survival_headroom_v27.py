"""Aggregate the registered V27 survival-valid oracle headroom screen."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_control_headroom as common,
)
from jax_experiments.analysis import (
    regime_channel_survival_headroom_v27 as protocol,
)
from jax_experiments.analysis import (
    run_regime_channel_survival_headroom_audit_v27 as audit,
)


def _bind() -> None:
    common.protocol = protocol
    common.validate_audit = audit.validate_audit
    common.T_CRITICAL_95_DF4 = 4.302652729696142
    common.MIN_RELATIVE_GAIN = protocol.MIN_RELATIVE_GAIN
    common.MAX_TERMINATION_GAP = protocol.MAX_TERMINATION_GAP
    common.MIN_MODE_WINS = protocol.MIN_MODE_WINS
    common.MIN_PASSING_ENVS = protocol.MIN_PASSING_ENVS


def apply_survival_gate(row: dict) -> dict:
    checks = {
        "switching_relative_gain_at_least_15pct": (
            row["switching_relative_gain"] >= protocol.MIN_RELATIVE_GAIN
        ),
        "worst_mode_relative_gain_at_least_15pct": (
            row["worst_mode_relative_gain"] >= protocol.MIN_RELATIVE_GAIN
        ),
        "switching_wins_all_training_seeds": (
            row["paired_deltas"]["switching"]["wins"]
            >= protocol.MIN_SEED_WINS
        ),
        "worst_mode_wins_all_training_seeds": (
            row["paired_deltas"]["stationary_worst"]["wins"]
            >= protocol.MIN_SEED_WINS
        ),
        "stationary_modes_improved_at_least_3_of_4": (
            row["mode_wins"] >= protocol.MIN_MODE_WINS
        ),
        "switching_termination_gap_at_most_5pp": (
            row["switching_termination_gap"]
            <= protocol.MAX_TERMINATION_GAP
        ),
    }
    for role in protocol.ROLES:
        summary = row["summaries"][role]
        checks[f"{role}_stationary_termination_at_most_10pct"] = (
            summary["stationary_termination_mean"]
            <= protocol.MAX_ABSOLUTE_TERMINATION
        )
        checks[f"{role}_switching_termination_at_most_10pct"] = (
            summary["switching_termination_mean"]
            <= protocol.MAX_ABSOLUTE_TERMINATION
        )
    row["gate_checks"] = checks
    row["env_gate_pass"] = all(checks.values())
    return row


def _render(payload: dict) -> str:
    lines = [
        "# Hopper and Walker2d structured-channel headroom result",
        "",
        "This is an environment-capacity screen, not a learned BAPR result. ",
        "Each role has the same 5.6M-step budget; the oracle receives the true ",
        "mode and the robust arm receives zero context.",
        "",
        "| Env | Robust / oracle switch | Gain | Robust / oracle worst | "
        "Gain | Mode wins | Seed wins switch/worst | Max termination | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for env in protocol.ENVS:
        row = payload["environments"][env]
        robust = row["summaries"]["robust"]
        oracle = row["summaries"]["oracle"]
        max_termination = max(
            robust["stationary_termination_mean"],
            robust["switching_termination_mean"],
            oracle["stationary_termination_mean"],
            oracle["switching_termination_mean"],
        )
        lines.append(
            f"| {protocol.env_slug(env)} | "
            f"{robust['switching_mean_mean']:.1f} / "
            f"{oracle['switching_mean_mean']:.1f} | "
            f"{100 * row['switching_relative_gain']:+.1f}% | "
            f"{robust['stationary_worst_mean']:.1f} / "
            f"{oracle['stationary_worst_mean']:.1f} | "
            f"{100 * row['worst_mode_relative_gain']:+.1f}% | "
            f"{row['mode_wins']}/4 | "
            f"{row['paired_deltas']['switching']['wins']}/"
            f"{row['paired_deltas']['stationary_worst']['wins']} | "
            f"{100 * max_termination:.1f}% | "
            f"{'PASS' if row['env_gate_pass'] else 'FAIL'} |"
        )
    gate = payload["bapr_authorization"]
    lines += [
        "",
        "## Decision",
        "",
        f"Passing environments: `{gate['passing_environments']}`.",
        "",
        ("Only those environments are authorized for a frozen V21 BAPR "
         "transfer screen."
         if gate["passing_environments"] else
         "No BAPR transfer is authorized; retain both environments as "
         "headroom or survival negatives without severity tuning."),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    protocol.validate_registration()
    _bind()
    data = common._load_all()
    environments = {
        env: apply_survival_gate(common._analyze_env(data[env]))
        for env in protocol.ENVS
    }
    passing = [
        env for env, row in environments.items() if row["env_gate_pass"]
    ]
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "family": protocol.FAMILY,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "audit_event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "budget": {
            "environment_steps_per_role": protocol.FINAL_TOTAL_STEPS,
            "gradient_updates_per_role": protocol.FINAL_UPDATE_COUNT,
        },
        "environments": environments,
        "bapr_authorization": {
            "status": "PASS" if passing else "FAIL",
            "passing_environments": passing,
            "required_passing_environments": protocol.MIN_PASSING_ENVS,
            "scope": "passing environments only",
        },
        "stopping_rule": protocol.registration_payload()["stopping_rule"],
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _render(payload))
    print(
        "V27_HEADROOM=" + payload["bapr_authorization"]["status"]
        + " passing=" + ",".join(passing),
        flush=True,
    )


if __name__ == "__main__":
    main()
