"""Run one strict BAPR-v5 audit with the shared option-audit engine."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import bapr_v5_hard_option as protocol
from jax_experiments.analysis import run_bapr_v4_persistent_option_audit as audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    audit.run(args.event_seed, candidate_protocol=protocol)


if __name__ == "__main__":
    main()
