"""Aggregate one frozen-anchor v2 development variant."""
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
    regime_polarity_frozen_anchor as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_anchor_eval as common,
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
    bind_variant(variant)
    return base.analyze()


def run(variant: str) -> None:
    view = bind_variant(variant)
    result = base.analyze()
    view.write_json_atomic(view.analysis_json(), result)
    view.write_text_atomic(
        view.analysis_markdown(), base._markdown(result))
    print(
        "FROZEN ANCHOR ANALYSIS COMPLETE: "
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
