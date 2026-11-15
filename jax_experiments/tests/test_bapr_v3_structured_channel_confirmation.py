"""Protocol tests for fresh-seed CUSUM confirmation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import (
    bapr_v3_sequence_router as sequence,
)
from jax_experiments.analysis import (
    bapr_v3_structured_channel_confirmation as protocol,
)
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility


def load_submit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_structured_channel_confirmation.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_structured_channel_confirmation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_confirmation_seeds_are_fresh_and_disjoint():
    prior = {
        *estimator.TRAIN_EVENT_SEEDS,
        *estimator.VALIDATION_EVENT_SEEDS,
        *utility.VALIDATION_EVENT_SEEDS,
        *utility.HOLDOUT_EVENT_SEEDS,
        *sequence.TRAIN_EVENT_SEEDS,
        *sequence.VALIDATION_EVENT_SEEDS,
        *sequence.HOLDOUT_EVENT_SEEDS,
    }
    assert len(protocol.EVENT_SEEDS) == 5
    assert not prior.intersection(protocol.EVENT_SEEDS)


def test_confirmation_submission_is_scheduler_only():
    module = load_submit_module()
    signature, task, output = module.evaluation_spec(10100, "high")
    assert signature.endswith("/event-seed-10100")
    assert output == protocol.group_path(10100)
    assert task["require_node"] == "jtl311linux"
    assert task["ckpt_glob"] == "router_manifest.json"
    assert "--resume" in task["cmd"]
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()

