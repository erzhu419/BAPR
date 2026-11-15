"""Apply the preregistered BAPR-v87 promotion gate."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "jax_experiments" / "results_bapr_v87_constrained_deploy"
RESULT_ROOT = ROOT / "jax_experiments" / "results_bapr_v87_validation"
REPORT = ROOT / "reports" / "bapr_v87_validation_2026-07-11.md"
CSV_PATH = ROOT / "reports" / "bapr_v87_validation_2026-07-11.csv"
MODES = ("robust", "oracle", "learned")
ENVS = ("Ant", "HalfCheetah", "Hopper", "Walker2d")


@dataclass(frozen=True)
class Metrics:
    run_name: str
    variant: str
    env: str
    mode: str
    ood_mean: float
    ood_worst: float
    ood_terminated_rate: float
    switching_mean: float
    switching_terminations: float
    switch_span: float
    switch_auc: float
    detection_delay: float
    latent_correlation: float
    latent_mae: float
    deployment_adaptation_strength: float
    safe_target_mean: float
    safe_target_std: float
    safe_targets_calibrated: float


def finite(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def fmt(value, digits: int = 1) -> str:
    value = finite(value)
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def gain(value: float, base: float) -> float:
    if not all(math.isfinite(finite(item)) for item in (value, base)):
        return math.nan
    if abs(base) < 1e-9:
        return math.nan
    return 100.0 * (value - base) / abs(base)


def retention(value: float, base: float) -> float:
    if not all(math.isfinite(finite(item)) for item in (value, base)):
        return math.nan
    if base > 1e-9:
        return value / base
    relative_gain = gain(value, base)
    return 1.0 + relative_gain / 100.0


def split_run_name(run_name: str) -> tuple[str, str]:
    for env in ENVS:
        suffix = f"_{env}_s0"
        if run_name.endswith(suffix):
            return run_name[:-len(suffix)], env
    raise ValueError(f"unrecognized V87 run name: {run_name}")


def final_log_value(run_dir: Path, key: str) -> float:
    path = run_dir / "logs" / f"{key}.npy"
    try:
        values = np.asarray(np.load(path), dtype=float).reshape(-1)
        return finite(values[-1]) if len(values) else math.nan
    except Exception:
        return math.nan


def tail_log_mean(run_dir: Path, key: str, count: int = 200) -> float:
    path = run_dir / "logs" / f"{key}.npy"
    try:
        values = np.asarray(np.load(path), dtype=float).reshape(-1)
        if not len(values):
            return math.nan
        return finite(np.nanmean(values[-min(len(values), count):]))
    except Exception:
        return math.nan


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_metrics(train_dir: Path, result_dir: Path, mode: str) -> Metrics | None:
    summary = read_rows(result_dir / "summary.csv")
    task_rows = read_rows(result_dir / "task_returns.csv")
    stationary = next((
        row for row in summary
        if row.get("metric_group") == "stationary"
        and row.get("split") == "test"
    ), None)
    switching = next((
        row for row in summary
        if row.get("metric_group") == "switching"
    ), None)
    if stationary is None or switching is None or not task_rows:
        return None
    test_returns = [
        finite(row.get("return_mean")) for row in task_rows
        if row.get("split") == "test"
        and math.isfinite(finite(row.get("return_mean")))
    ]
    if not test_returns:
        return None
    variant, env = split_run_name(train_dir.name)
    return Metrics(
        run_name=train_dir.name,
        variant=variant,
        env=env,
        mode=mode,
        ood_mean=finite(stationary.get("return_mean")),
        ood_worst=min(test_returns),
        ood_terminated_rate=finite(stationary.get("terminated_rate_mean")),
        switching_mean=finite(switching.get("switch_return_mean")),
        switching_terminations=finite(
            switching.get("termination_count_mean")),
        switch_span=finite(switching.get("switch_latent_span_mean")),
        switch_auc=finite(switching.get("switch_auc")),
        detection_delay=finite(switching.get("median_detection_delay")),
        latent_correlation=finite(
            switching.get("latent_correlation_after")),
        latent_mae=finite(switching.get("latent_mae_after")),
        deployment_adaptation_strength=tail_log_mean(
            train_dir, "v2_policy_adaptation_strength"),
        safe_target_mean=final_log_value(
            train_dir, "v2_safe_target_mean"),
        safe_target_std=final_log_value(
            train_dir, "v2_safe_target_std"),
        safe_targets_calibrated=final_log_value(
            train_dir, "v2_safe_targets_calibrated"),
    )


def expected_training_runs(train_root: Path) -> list[Path]:
    return sorted(train_root.glob("v87*_s0"))


def load_all(train_root: Path, result_root: Path) -> tuple[list[Metrics], list[str]]:
    rows: list[Metrics] = []
    missing: list[str] = []
    for train_dir in expected_training_runs(train_root):
        for mode in MODES:
            result_dir = result_root / f"{train_dir.name}__{mode}"
            metrics = load_metrics(train_dir, result_dir, mode)
            if metrics is None:
                missing.append(result_dir.name)
            else:
                rows.append(metrics)
    return rows, missing


def by_mode(rows: list[Metrics], variant: str, env: str) -> dict[str, Metrics]:
    return {
        row.mode: row for row in rows
        if row.variant == variant and row.env == env
    }


def variant_gate(rows: list[Metrics], variant: str) -> dict[str, object]:
    ladders = {env: by_mode(rows, variant, env) for env in ENVS}
    complete = all(set(ladder) == set(MODES) for ladder in ladders.values())
    if not complete:
        return {"complete": False, "passes": False}

    oracle_headroom = 0
    recovery_values: list[float] = []
    protocol_valid = True
    detector_ok = True
    safe_targets_ok = True
    for env, ladder in ladders.items():
        robust, oracle, learned = (
            ladder["robust"], ladder["oracle"], ladder["learned"])
        oracle_gains = (
            gain(oracle.ood_mean, robust.ood_mean),
            gain(oracle.switching_mean, robust.switching_mean),
        )
        learned_gains = (
            gain(learned.ood_mean, robust.ood_mean),
            gain(learned.switching_mean, robust.switching_mean),
        )
        oracle_headroom += int(min(oracle_gains) >= 10.0)
        for learned_gain, oracle_gain in zip(learned_gains, oracle_gains):
            if oracle_gain > 0.0:
                recovery_values.append(learned_gain / oracle_gain)
        protocol_valid &= all(
            math.isfinite(row.switch_span) and row.switch_span >= 0.5
            for row in ladder.values())
        detector_ok &= (
            learned.switch_auc >= 0.8
            and math.isfinite(learned.detection_delay)
            and learned.detection_delay < 50.0)
        safe_targets_ok &= (
            learned.safe_targets_calibrated >= 0.5
            and learned.safe_target_std > 1e-3)

    ant_hc_ok = all(
        min(
            gain(ladders[env]["learned"].ood_mean,
                 ladders[env]["robust"].ood_mean),
            gain(ladders[env]["learned"].switching_mean,
                 ladders[env]["robust"].switching_mean),
        ) >= 10.0
        for env in ("Ant", "HalfCheetah"))
    retention_ok = all(
        min(
            retention(ladders[env]["learned"].ood_mean,
                      ladders[env]["robust"].ood_mean),
            retention(ladders[env]["learned"].switching_mean,
                      ladders[env]["robust"].switching_mean),
        ) >= 0.95
        for env in ("Hopper", "Walker2d"))
    ant_base = ladders["Ant"]["robust"]
    ant_learned = ladders["Ant"]["learned"]
    ant_risk_ok = (
        ant_learned.ood_terminated_rate
        <= ant_base.ood_terminated_rate + 1e-9
        and ant_learned.switching_terminations
        <= ant_base.switching_terminations + 1e-9)
    recovery_ok = bool(recovery_values) and min(recovery_values) >= 0.70
    checks = {
        "complete": True,
        "oracle_headroom": oracle_headroom >= 3,
        "ant_hc_gain": ant_hc_ok,
        "oracle_recovery": recovery_ok,
        "hopper_walker_retention": retention_ok,
        "detector": detector_ok,
        "safe_targets": safe_targets_ok,
        "ant_risk": ant_risk_ok,
        "protocol": protocol_valid,
        "oracle_headroom_count": oracle_headroom,
        "min_recovery": min(recovery_values) if recovery_values else math.nan,
    }
    checks["passes"] = all(
        checks[key] for key in (
            "oracle_headroom", "ant_hc_gain", "oracle_recovery",
            "hopper_walker_retention", "detector", "safe_targets",
            "ant_risk", "protocol"))
    return checks


def build_report(rows: list[Metrics], missing: list[str]) -> str:
    variants = sorted({row.variant for row in rows})
    lines = [
        "# BAPR-v87 Validation and Promotion Gate",
        "",
        "## Audit",
        "",
        f"- Complete evaluation rows: **{len(rows)}/36**.",
        "- All stationary results use the untouched validation task stream; the reserved stream remains unopened.",
        "- Robust, oracle, and learned policies are evaluated from the same final checkpoint and deterministic RNG schedule.",
        "- Promotion is decided per predeclared variant; metrics are not mixed across variants.",
    ]
    if missing:
        lines.extend(["", "### Missing", ""])
        lines.extend(f"- `{name}`" for name in missing)

    lines.extend([
        "", "## Same-checkpoint ladder", "",
        "| variant | env | robust OOD / switch | oracle gain OOD / switch | learned gain OOD / switch | learned corr / MAE | AUC / delay | adapt strength | Ant/test term robust->learned | safe target std |",
        "|---|---|---|---|---|---|---|---:|---|---:|",
    ])
    for variant in variants:
        for env in ENVS:
            ladder = by_mode(rows, variant, env)
            if set(ladder) != set(MODES):
                lines.append(
                    f"| {variant} | {env} | incomplete | | | | | | | |")
                continue
            robust, oracle, learned = (
                ladder["robust"], ladder["oracle"], ladder["learned"])
            termination = (
                f"{fmt(robust.ood_terminated_rate, 3)}->"
                f"{fmt(learned.ood_terminated_rate, 3)}"
                if env == "Ant" else "n/a")
            lines.append(
                f"| {variant} | {env} | {fmt(robust.ood_mean)} / "
                f"{fmt(robust.switching_mean)} | "
                f"{fmt(gain(oracle.ood_mean, robust.ood_mean))}% / "
                f"{fmt(gain(oracle.switching_mean, robust.switching_mean))}% | "
                f"{fmt(gain(learned.ood_mean, robust.ood_mean))}% / "
                f"{fmt(gain(learned.switching_mean, robust.switching_mean))}% | "
                f"{fmt(learned.latent_correlation, 3)} / "
                f"{fmt(learned.latent_mae, 3)} | "
                f"{fmt(learned.switch_auc, 3)} / "
                f"{fmt(learned.detection_delay)} | "
                f"{fmt(learned.deployment_adaptation_strength, 3)} | "
                f"{termination} | "
                f"{fmt(learned.safe_target_std, 3)} |")

    lines.extend([
        "", "## Fixed gate", "",
        "| variant | oracle 3/4 | Ant+HC +10% | recover 70% | Hopper+Walker 95% | AUC/delay | safe targets | Ant risk | protocol | promote |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    winners = []
    for variant in variants:
        gate = variant_gate(rows, variant)
        if gate.get("passes"):
            winners.append(variant)
        def mark(key: str) -> str:
            return "PASS" if gate.get(key) else "FAIL"
        lines.append(
            f"| {variant} | {mark('oracle_headroom')} "
            f"({gate.get('oracle_headroom_count', 0)}/4) | "
            f"{mark('ant_hc_gain')} | {mark('oracle_recovery')} "
            f"({fmt(100.0 * finite(gate.get('min_recovery')), 1)}%) | "
            f"{mark('hopper_walker_retention')} | {mark('detector')} | "
            f"{mark('safe_targets')} | {mark('ant_risk')} | "
            f"{mark('protocol')} | {mark('passes')} |")

    lines.extend(["", "## Verdict", ""])
    if missing:
        lines.append(
            "**Verdict withheld:** the preregistered 36-row validation ladder is incomplete.")
    elif len(winners) == 1:
        lines.append(
            f"**Promote `{winners[0]}`:** freeze this configuration, expand to seeds 0-4, and open the reserved task stream only after the multi-seed model is fixed.")
    elif len(winners) > 1:
        lines.append(
            "**Multiple variants passed:** apply the predeclared simplicity rule and promote the least constrained passing variant; do not combine metrics or tune on the reserved stream.")
    else:
        lines.append(
            "**No V87 variant passed:** do not run five seeds. Freeze the "
            "algorithm-search result and pivot to the systematic analysis "
            "paper route, using bus and V87b Ant as validated positive cases, "
            "historical HalfCheetah only after protocol-matched revalidation, "
            "and robust-policy or termination failures as negative cases.")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[Metrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(Metrics.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()
    rows, missing = load_all(args.train_root, args.result_root)
    write_csv(args.csv, rows)
    report = build_report(rows, missing)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
