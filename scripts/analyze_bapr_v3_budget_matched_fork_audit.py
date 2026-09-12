#!/usr/bin/env python3
"""Compatibility wrapper for the packaged causal v2 audit analyzer."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_experiments.analysis.analyze_bapr_v3_budget_matched_fork_audit import main


if __name__ == "__main__":
    main()
