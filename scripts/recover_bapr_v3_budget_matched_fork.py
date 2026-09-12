#!/usr/bin/env python3
"""CLI wrapper for the archived-source BAPR-v3 pair finalizer."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_experiments.analysis.recover_bapr_v3_budget_matched_fork import main


if __name__ == "__main__":
    main()
