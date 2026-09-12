"""Aggregate one conservative frozen-residual development variant."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    analyze_regime_polarity_anchored_residual as base,
)
from jax_experiments.analysis import (
    audit_regime_polarity_anchored_residual as base_audit,
)
from jax_experiments.analysis import (
    calibrate_regime_polarity_anchored_residual as base_calibration,
)
from jax_experiments.analysis import (
    regime_polarity_conservative_residual as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_conservative_residual_eval as common,
)


def bind_variant(variant: str):
    view = common.bind_variant(variant)
    base_calibration.protocol = view
    base_calibration.common = common
    base_audit.protocol = view
    base_audit.common = common
    base_audit.calibration = base_calibration
    base.protocol = view
    base.common = common
    base.audit = base_audit
    return view


def analyze(variant: str):
    view = bind_variant(variant)
    result = base.analyze()
    result["conservative_training"] = [
        {
            "training_seed": seed,
            **(
                view.read_json(
                    protocol.branch_manifest(variant, seed)
                ).get("conservative_training") or {}
            ),
        }
        for seed in view.TRAINING_SEEDS
    ]
    return result


def _markdown(result: dict) -> str:
    text = base._markdown(result).rstrip()
    lines = [
        text,
        "",
        "## Conservative training diagnostics",
        "",
        "| seed | actor changed | accept rate | min LCB | mean shortfall |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in result["conservative_training"]:
        lines.append(
            f"| {row['training_seed']} | "
            f"{row.get('adaptive_policy_changed', 'n/a')} | "
            f"{row.get('update_accept_rate_mean', float('nan')):.3f} | "
            f"{row.get('advantage_lcb_min', float('nan')):.4f} | "
            f"{row.get('advantage_shortfall_mean', float('nan')):.4f} |")
    lines.append("")
    return "\n".join(lines)


def run(variant: str) -> None:
    view = bind_variant(variant)
    result = analyze(variant)
    view.write_json_atomic(view.analysis_json(), result)
    view.write_text_atomic(
        view.analysis_markdown(), _markdown(result))
    print(
        "CONSERVATIVE RESIDUAL ANALYSIS COMPLETE: "
        f"variant={variant} "
        f"promotion={result['promotion_to_untouched_five_seed_confirmation']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=protocol.VARIANTS, required=True)
    args = parser.parse_args()
    run(args.variant)


if __name__ == "__main__":
    main()
