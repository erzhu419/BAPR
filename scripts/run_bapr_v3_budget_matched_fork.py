#!/usr/bin/env python3
"""CLI wrapper for the distributable BAPR-v3 budget-matched fork runner."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_experiments.analysis.run_bapr_v3_budget_matched_fork import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
