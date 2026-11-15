"""Aggregate the registered V30 Ant paired branch-risk diagnostic."""
from __future__ import annotations

from typing import Any

from jax_experiments.analysis import (
    regime_polarity_ant_branch_risk_v30 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_branch_risk_audit_v30 as audit,
)


COUNT_KEYS = (
    "n",
    "candidate_terminated",
    "fallback_terminated",
    "rescued",
    "harmed",
    "both_terminated",
    "neither_terminated",
)
SUM_KEYS = ("candidate_return_sum", "fallback_return_sum")


def _merge(rows) -> dict[str, Any]:
    counter = audit._empty_counter()
    for row in rows:
        for key in COUNT_KEYS:
            counter[key] += int(row[key])
        for key in SUM_KEYS:
            counter[key] += float(row[key])
    return audit._finalize(counter)


def _seed_decision(payload: dict[str, Any]) -> dict[str, Any]:
    horizon = str(protocol.MAX_RISK_HORIZON)
    primary = payload["overall_horizons"][horizon]
    unique = payload["unique_candidate_horizons"][horizon]
    mode_rows = {
        mode: payload["mode_horizons"][mode][horizon]
        for mode in map(str, protocol.MODES)
    }
    informative = bool(
        int(unique["terminated"])
        >= protocol.MIN_UNIQUE_CANDIDATE_FAILURES_PER_INFORMATIVE_SEED)
    mode_wins = sum(
        float(row["absolute_risk_reduction"])
        >= protocol.MIN_ABSOLUTE_RISK_REDUCTION
        for row in mode_rows.values()
    )
    gate_pass = bool(
        informative
        and float(primary["absolute_risk_reduction"])
        >= protocol.MIN_ABSOLUTE_RISK_REDUCTION
        and float(primary["rescue_fraction_given_candidate_failure"])
        >= protocol.MIN_RESCUE_FRACTION
        and float(primary["harm_fraction_given_candidate_survival"])
        <= protocol.MAX_HARM_FRACTION
        and mode_wins >= protocol.MIN_MODE_WINS
    )
    return {
        "reference_mode": payload["identity"]["reference_mode"],
        "snapshot_count": payload["snapshot_count"],
        "unique_candidate_continuations": unique["n"],
        "unique_candidate_failures": unique["terminated"],
        "informative": informative,
        "candidate_termination_risk": primary["candidate_termination_risk"],
        "fallback_termination_risk": primary["fallback_termination_risk"],
        "absolute_risk_reduction": primary["absolute_risk_reduction"],
        "rescue_fraction_given_candidate_failure": primary[
            "rescue_fraction_given_candidate_failure"],
        "harm_fraction_given_candidate_survival": primary[
            "harm_fraction_given_candidate_survival"],
        "mode_wins": mode_wins,
        "gate_pass": gate_pass,
    }


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    payloads = {}
    for seed in protocol.TRAINING_SEEDS:
        audit.validate_audit(seed)
        payloads[seed] = protocol.read_json(protocol.audit_result(seed))

    seed_rows = {
        str(seed): _seed_decision(payload)
        for seed, payload in payloads.items()
    }
    horizon_rows = {
        str(horizon): _merge(
            payload["overall_horizons"][str(horizon)]
            for payload in payloads.values())
        for horizon in protocol.RISK_HORIZONS
    }
    mode_rows = {
        str(mode): _merge(
            payload["mode_horizons"][str(mode)][
                str(protocol.MAX_RISK_HORIZON)]
            for payload in payloads.values())
        for mode in protocol.MODES
    }
    primary = horizon_rows[str(protocol.MAX_RISK_HORIZON)]
    informative = [
        row for row in seed_rows.values() if row["informative"]
    ]
    mode_wins = sum(
        float(row["absolute_risk_reduction"])
        >= protocol.MIN_ABSOLUTE_RISK_REDUCTION
        for row in mode_rows.values()
    )
    invariance_pass = all(
        payload["candidate_mode_invariance"]["pass"] is True
        for payload in payloads.values()
    )
    primary_pass = bool(
        invariance_pass
        and len(informative) >= protocol.MIN_INFORMATIVE_SEEDS
        and all(row["gate_pass"] for row in informative)
        and float(primary["absolute_risk_reduction"])
        >= protocol.MIN_ABSOLUTE_RISK_REDUCTION
        and float(primary["rescue_fraction_given_candidate_failure"])
        >= protocol.MIN_RESCUE_FRACTION
        and float(primary["harm_fraction_given_candidate_survival"])
        <= protocol.MAX_HARM_FRACTION
        and mode_wins >= protocol.MIN_MODE_WINS
    )

    if not invariance_pass:
        diagnosis = "branch_protocol_or_action_compensation_failure"
        next_step = "debug branch-state restoration; do not train a risk model"
    elif len(informative) < protocol.MIN_INFORMATIVE_SEEDS:
        diagnosis = "candidate_failure_risk_not_reproduced_on_new_states"
        next_step = (
            "close the Ant shielding branch because the preregistered screen "
            "does not provide enough independent failure signal")
    elif primary_pass:
        diagnosis = "full_robust_continuation_has_rescue_headroom"
        next_step = (
            "on new development policies and states, train separate calibrated "
            "finite-horizon candidate- and fallback-continuation risk models")
    elif float(primary["harm_fraction_given_candidate_survival"]) > (
        protocol.MAX_HARM_FRACTION
    ):
        diagnosis = "robust_fallback_introduces_excess_state_dependent_harm"
        next_step = (
            "close the fallback-shield route; immutable robust parameters do "
            "not make robust continuation safe from specialist states")
    else:
        diagnosis = "robust_fallback_lacks_consistent_rescue_headroom"
        next_step = (
            "close the Ant shielding branch and retain V29 as a reliability "
            "counterexample rather than fitting another risk critic")

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "seed_results": seed_rows,
        "horizon_results": horizon_rows,
        "primary_mode_results": mode_rows,
        "informative_seed_count": len(informative),
        "primary_mode_wins": mode_wins,
        "candidate_mode_invariance_pass": invariance_pass,
        "risk_model_training_authorized": primary_pass,
        "diagnosis": diagnosis,
        "next_step": next_step,
        "accounting": {
            "new_training_interactions": 0,
            "saved_or_synchronized_simulator_states": 0,
            "policy_seed_count": len(protocol.TRAINING_SEEDS),
            "paired_mode_interventions_are_not_independent_seeds": True,
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# V30 Ant finite-horizon paired branch-risk diagnostic",
        "",
        "V30 reuses the frozen V29 reference choice and branches the same "
        "simulator states into complete compensated-candidate and robust-fallback "
        "continuations. It performs no training.",
        "",
        "| Horizon | Candidate term. | Fallback term. | Reduction | Rescue | Harm |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in protocol.RISK_HORIZONS:
        row = payload["horizon_results"][str(horizon)]
        lines.append(
            f"| {horizon} | {row['candidate_termination_risk']:.1%} "
            f"| {row['fallback_termination_risk']:.1%} "
            f"| {row['absolute_risk_reduction']:+.1%} "
            f"| {row['rescue_fraction_given_candidate_failure']:.1%} "
            f"| {row['harm_fraction_given_candidate_survival']:.1%} |")
    lines.extend([
        "",
        "| Seed | Ref. | Snapshots | Unique failures | Candidate | Fallback | Reduction | Modes | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for seed, row in payload["seed_results"].items():
        lines.append(
            f"| {seed} | {row['reference_mode']} | {row['snapshot_count']} "
            f"| {row['unique_candidate_failures']} "
            f"| {row['candidate_termination_risk']:.1%} "
            f"| {row['fallback_termination_risk']:.1%} "
            f"| {row['absolute_risk_reduction']:+.1%} "
            f"| {row['mode_wins']}/4 "
            f"| {'PASS' if row['gate_pass'] else ('N/A' if not row['informative'] else 'FAIL')} |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"Finite-horizon risk-model authorization: **{'PASS' if payload['risk_model_training_authorized'] else 'FAIL'}**.",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Informative policy seeds: `{payload['informative_seed_count']}/3`; "
        f"actual-mode wins: `{payload['primary_mode_wins']}/4`.",
        f"Next step: {payload['next_step']}.",
        "",
        "This is a development headroom audit, not a safety guarantee. Actuator "
        "mode interventions and continuation draws are paired diagnostics, not "
        "additional independent policy seeds.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    markdown = render(payload)
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), markdown)
    protocol.write_text_atomic(protocol.REPORT, markdown)
    print(
        "V30 ANT BRANCH-RISK ANALYSIS COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"risk_model_authorized={payload['risk_model_training_authorized']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
