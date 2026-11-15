"""Frozen protocol for fresh-seed CUSUM return confirmation."""
from __future__ import annotations

from pathlib import Path

from jax_experiments.analysis import bapr_v3_utility_aware_router as utility


ROOT = utility.ROOT
FAMILY = utility.FAMILY
ENV = utility.ENV
DECISION_VARIANT = "cs4d025c80h8"
EVENT_SEEDS = (10100, 10200, 10300, 10400, 10500)
STATIONARY_NONINFERIORITY_MARGIN = 100.0
TERMINATION_RATE_MARGIN = 0.05
MIN_ORACLE_RECOVERY = 0.70
MIN_FULL_CYCLE_WINS = 4
MIN_COMMITTED_ROUTE_ACCURACY = 0.90

RESULT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_cusum_confirmation_v1"
)
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
ANALYSIS_JSON = ANALYSIS_ROOT / "summary.json"
ANALYSIS_REPORT = ANALYSIS_ROOT / "report.md"

GROUP_SCHEMA = "bapr.v3-structured-channel-cusum-confirmation-group.v1"
ANALYSIS_SCHEMA = "bapr.v3-structured-channel-cusum-confirmation-analysis.v1"


def configure() -> None:
    utility.configure()


def group_path(event_seed: int) -> Path:
    if int(event_seed) not in EVENT_SEEDS:
        raise ValueError(f"unregistered confirmation seed {event_seed}")
    return RESULT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def file_record(path: Path):
    return utility.estimator.file_record(path)


def write_json_atomic(path: Path, payload) -> None:
    utility.estimator.write_json_atomic(path, payload)

