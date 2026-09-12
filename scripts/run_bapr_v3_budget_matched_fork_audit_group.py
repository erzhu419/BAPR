#!/usr/bin/env python3
"""Compatibility wrapper for the packaged grouped v2 audit runner."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_experiments.analysis.run_bapr_v3_budget_matched_fork_audit_group import main


if __name__ == "__main__":
    main()
