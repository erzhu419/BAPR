"""Validate one specialist family and publish its checkpoint staging marker."""
from __future__ import annotations

import argparse
from pathlib import Path

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("legacy", "stochastic_headroom", "structured_channel"),
        default="legacy",
    )
    parser.add_argument("--env")
    parser.add_argument("--family", required=True)
    parser.add_argument("--status-dir", type=Path)
    args = parser.parse_args()
    if args.profile == "stochastic_headroom":
        if not args.env:
            parser.error("--env is required for stochastic_headroom")
        protocol.configure_stochastic_headroom(args.env)
    elif args.profile == "structured_channel":
        if not args.env:
            parser.error("--env is required for structured_channel")
        protocol.configure_structured_channel_headroom(args.env)
    elif args.env and args.env != protocol.ENV:
        parser.error(f"legacy profile only supports {protocol.ENV}")
    protocol._require_family(args.family)

    marker = protocol.publish_family_audit_ready(args.family)
    if args.status_dir:
        status_dir = args.status_dir.resolve()
        protocol.write_json_atomic(status_dir / "audit_ready_summary.json", {
            "schema": "bapr.v3-independent-specialist-audit-ready-summary.v1",
            "status": "complete",
            "profile": args.profile,
            "family": args.family,
            "env": protocol.ENV,
            "seed": protocol.SEED,
            "bundle_count": len(marker["bundle_manifests"]),
            "source_sha256": marker["source_sha256"],
            "audit_ready_path": str(
                protocol.family_bundle_root(args.family)
                / protocol.AUDIT_READY_MARKER
            ),
        })
    print(
        "INDEPENDENT SPECIALIST AUDIT INPUT READY: "
        f"family={args.family} env={protocol.ENV} "
        f"bundles={len(marker['bundle_manifests'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
