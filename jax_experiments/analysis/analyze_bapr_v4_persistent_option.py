"""Compare BAPR-v4 persistent options with the frozen v3 baselines."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from jax_experiments.analysis import bapr_v4_persistent_option as protocol


DECISION_VARIANT = "cs4d025c80h8"
MIN_ORACLE_HARD_GAIN = 50.0
STATIONARY_MARGIN = 50.0


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stationary_values(group, source):
    return [
        float(episode["return"])
        for mode in range(4)
        for episode in group["stationary"][source][str(mode)]["episodes"]
    ]


def _switch_values(group, kind, source):
    return [
        float(episode["return"])
        for episode in group["switching"][kind][source]["episodes"]
    ]


def _termination_count(group, kind, source):
    return int(sum(
        int(episode["termination_count"])
        for episode in group["switching"][kind][source]["episodes"]))


def _summarize_values(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("audit returns must be finite and non-empty")
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "n": len(values),
    }


def _validate_group(payload, candidate_protocol):
    if (payload.get("schema") != candidate_protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or int(payload.get("event_seed", -1))
            not in candidate_protocol.DEVELOPMENT_EVENT_SEEDS
            or set(payload.get("stationary") or {})
            != set(candidate_protocol.CONTEXT_SOURCES)
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid persistent-option audit group identity")
    stationary_episodes = int(payload.get("episodes_per_task", 0))
    switching_episodes = int(payload.get("switching_episodes", 0))
    if stationary_episodes <= 0 or switching_episodes <= 0:
        raise ValueError("invalid persistent-option audit episode counts")
    for source in candidate_protocol.CONTEXT_SOURCES:
        if set(payload["stationary"][source]) != {"0", "1", "2", "3"}:
            raise ValueError("incomplete persistent-option stationary modes")
        for record in payload["stationary"][source].values():
            episodes = record.get("episodes") or []
            if (len(episodes) != stationary_episodes
                    or not all(math.isfinite(float(row["return"]))
                               for row in episodes)):
                raise ValueError("invalid persistent-option stationary episodes")
        for kind in ("slow_pair", "full_cycle"):
            episodes = payload["switching"][kind][source].get("episodes") or []
            if (len(episodes) != switching_episodes
                    or not all(math.isfinite(float(row["return"]))
                               for row in episodes)):
                raise ValueError("invalid persistent-option switching episodes")


def _baseline_group_path(candidate_protocol, event_seed: int) -> Path:
    live = (
        candidate_protocol.ROOT
        / "jax_experiments"
        / "results_bapr_v3_structured_channel_utility_router_validation_v3"
        / DECISION_VARIANT
        / "structured_channel"
        / "HalfCheetah"
        / f"event_seed_{int(event_seed)}"
        / "group.json"
    )
    if live.is_file():
        return live
    snapshot = (
        candidate_protocol.ROOT
        / "jax_experiments"
        / "analysis"
        / "protocol_snapshots"
        / "v5_baseline_validation"
        / f"event_seed_{int(event_seed)}"
        / "group.json"
    )
    if snapshot.is_file():
        return snapshot
    raise FileNotFoundError(
        f"missing live and frozen baseline group for event seed {event_seed}")


def summarize_event(event_seed: int, candidate_protocol=protocol,
                    version: str = "v4"):
    candidate = _load(candidate_protocol.audit_group_path(event_seed))
    _validate_group(candidate, candidate_protocol)
    baseline_path = _baseline_group_path(candidate_protocol, event_seed)
    baseline = _load(baseline_path)
    rows = {}
    for kind in ("stationary", "slow_pair", "full_cycle"):
        if kind == "stationary":
            get_candidate = lambda source: _stationary_values(
                candidate, source)
            get_baseline = lambda source: _stationary_values(
                baseline, source)
        else:
            get_candidate = lambda source, kind=kind: _switch_values(
                candidate, kind, source)
            get_baseline = lambda source, kind=kind: _switch_values(
                baseline, kind, source)
        values = {
            "baseline_robust": get_baseline("robust"),
            "baseline_dynamic_oracle": get_baseline(
                "dynamic_utility_oracle"),
            "hard_cusum": get_baseline("learned_utility_router"),
            f"{version}_robust": get_candidate("robust"),
            f"{version}_oracle_persistent": get_candidate(
                "oracle_persistent"),
            f"{version}_learned_persistent": get_candidate(
                "learned_persistent"),
        }
        rows[kind] = {
            name: _summarize_values(item)
            for name, item in values.items()
        }
        hard = rows[kind]["hard_cusum"]["mean"]
        robust = rows[kind]["baseline_robust"]["mean"]
        oracle = rows[kind][f"{version}_oracle_persistent"]["mean"]
        learned = rows[kind][f"{version}_learned_persistent"]["mean"]
        rows[kind]["comparisons"] = {
            "oracle_minus_hard": oracle - hard,
            "learned_minus_hard": learned - hard,
            "oracle_minus_baseline_robust": oracle - robust,
            "learned_minus_baseline_robust": learned - robust,
            "learned_oracle_recovery": (
                (learned - robust) / (oracle - robust)
                if oracle > robust else float("nan")),
        }
    rows["full_cycle"]["termination_counts"] = {
        "baseline_robust": _termination_count(
            baseline, "full_cycle", "robust"),
        "hard_cusum": _termination_count(
            baseline, "full_cycle", "learned_utility_router"),
        f"{version}_robust": _termination_count(
            candidate, "full_cycle", "robust"),
        f"{version}_oracle_persistent": _termination_count(
            candidate, "full_cycle", "oracle_persistent"),
        f"{version}_learned_persistent": _termination_count(
            candidate, "full_cycle", "learned_persistent"),
    }
    return {
        "event_seed": int(event_seed),
        "candidate_group": candidate_protocol.file_record(
            candidate_protocol.audit_group_path(event_seed)),
        "baseline_group": candidate_protocol.file_record(baseline_path),
        "protocols": rows,
    }


def analyze(candidate_protocol=protocol, version: str = "v4",
            title: str = "BAPR-v4 persistent-option development screen"):
    events = [summarize_event(seed, candidate_protocol, version)
              for seed in candidate_protocol.DEVELOPMENT_EVENT_SEEDS]
    full = [event["protocols"]["full_cycle"] for event in events]
    stationary = [event["protocols"]["stationary"] for event in events]
    oracle_deltas = [
        row["comparisons"]["oracle_minus_hard"] for row in full]
    learned_deltas = [
        row["comparisons"]["learned_minus_hard"] for row in full]
    oracle_term_safe = all(
        row["termination_counts"][f"{version}_oracle_persistent"]
        <= row["termination_counts"]["hard_cusum"]
        for row in full)
    learned_term_safe = all(
        row["termination_counts"][f"{version}_learned_persistent"]
        <= row["termination_counts"]["hard_cusum"]
        for row in full)
    oracle_stationary_safe = all(
        row["comparisons"]["oracle_minus_hard"] >= -STATIONARY_MARGIN
        for row in stationary)
    learned_stationary_safe = all(
        row["comparisons"]["learned_minus_hard"] >= -STATIONARY_MARGIN
        for row in stationary)
    oracle_gate = (
        all(delta > 0.0 for delta in oracle_deltas)
        and statistics.fmean(oracle_deltas) >= MIN_ORACLE_HARD_GAIN
        and oracle_term_safe and oracle_stationary_safe)
    learned_gate = (
        oracle_gate
        and all(delta > 0.0 for delta in learned_deltas)
        and statistics.fmean(learned_deltas) >= MIN_ORACLE_HARD_GAIN
        and learned_term_safe and learned_stationary_safe)
    payload = {
        "schema": candidate_protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "decision_variant": DECISION_VARIANT,
        "development_event_seeds": list(
            candidate_protocol.DEVELOPMENT_EVENT_SEEDS),
        "promotion_thresholds": {
            "minimum_mean_full_cycle_gain_over_hard": MIN_ORACLE_HARD_GAIN,
            "stationary_noninferiority_margin": STATIONARY_MARGIN,
            "termination_count_margin": 0,
        },
        "events": events,
        "gate": {
            "oracle_capacity_pass": bool(oracle_gate),
            "learned_persistent_pass": bool(learned_gate),
            "oracle_full_cycle_deltas": oracle_deltas,
            "oracle_full_cycle_mean_delta": float(
                statistics.fmean(oracle_deltas)),
            "learned_full_cycle_deltas": learned_deltas,
            "learned_full_cycle_mean_delta": float(
                statistics.fmean(learned_deltas)),
            "oracle_termination_safe": bool(oracle_term_safe),
            "learned_termination_safe": bool(learned_term_safe),
            "oracle_stationary_safe": bool(oracle_stationary_safe),
            "learned_stationary_safe": bool(learned_stationary_safe),
        },
    }
    candidate_protocol.write_json_atomic(
        candidate_protocol.ANALYSIS_ROOT / "summary.json", payload)
    lines = [
        f"# {title}",
        "",
        f"Oracle capacity gate: **{'PASS' if oracle_gate else 'FAIL'}**  ",
        f"Learned persistent gate: **{'PASS' if learned_gate else 'FAIL'}**",
        "",
        "| Event | Source | Stationary | Slow pair | Full cycle | Full - hard |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    labels = (
        ("hard_cusum", "hard CUSUM"),
        (f"{version}_robust", f"{version} robust"),
        (f"{version}_oracle_persistent", f"{version} oracle option"),
        (f"{version}_learned_persistent", f"{version} learned option"),
    )
    for event in events:
        event_seed = event["event_seed"]
        rows = event["protocols"]
        hard = rows["full_cycle"]["hard_cusum"]["mean"]
        for key, label in labels:
            value = rows["full_cycle"][key]["mean"]
            lines.append(
                f"| {event_seed} | {label} | "
                f"{rows['stationary'][key]['mean']:.1f} | "
                f"{rows['slow_pair'][key]['mean']:.1f} | "
                f"{value:.1f} | {value - hard:+.1f} |")
    lines += [
        "",
        "The oracle gate is evaluated first. A failed oracle gate closes this "
        "persistent-option architecture; a passed oracle gate with a failed "
        "learned gate localizes the remaining problem to inference or option "
        "boundary selection.",
        "",
    ]
    report = candidate_protocol.ANALYSIS_ROOT / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"{version.upper()} ANALYSIS COMPLETE: "
        f"oracle={'PASS' if oracle_gate else 'FAIL'} "
        f"learned={'PASS' if learned_gate else 'FAIL'}",
        flush=True)
    return payload


if __name__ == "__main__":
    analyze()
