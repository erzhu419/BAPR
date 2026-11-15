"""Tests for the conservative frozen-residual v3 protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jax_experiments.algos.regime_sac import RegimeSAC
from jax_experiments.analysis import (
    regime_polarity_conservative_residual as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_anchor as v2,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_anchor_branch as base_branch,
)
from jax_experiments.configs.default import Config
from jax_experiments.train import make_algo


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import submit_regime_polarity_conservative_residual as submit
import submit_regime_polarity_conservative_diagnostic as diagnostic_submit


def test_protocol_reuses_equal_budget_robust_without_retraining():
    assert protocol.BRANCH_TOTAL_STEPS == 11_200_000
    assert protocol.BRANCH_UPDATE_COUNT == 700_000
    assert protocol.BRANCH_ROLES == (
        "robust_long",
        "strict_small",
        "trust_small",
        "trust_tight",
    )
    for seed in protocol.TRAINING_SEEDS:
        assert (
            protocol.branch_bundle_dir("robust_long", seed)
            == v2.branch_bundle_dir("robust_long", seed)
        )


def test_variant_config_and_remote_command_enable_constraints():
    previous = base_branch.protocol
    base_branch.protocol = protocol
    try:
        source = Config()
        config = base_branch._adaptive_config(
            source,
            "strict_small",
            protocol.branch_run_dir("strict_small", 1103),
        )
        command = base_branch._training_command(
            1103,
            "strict_small",
            protocol.branch_run_dir("strict_small", 1103),
        )
    finally:
        base_branch.protocol = previous

    assert config.bapr_v2_train_advantage_constraint is True
    assert config.bapr_v2_train_update_filter is True
    assert config.bapr_v2_train_update_tolerance == 0.0
    assert config.bapr_v2_train_update_floor == 0.0
    assert "--bapr_v2_train_advantage_constraint" in command
    assert "--bapr_v2_train_update_filter" in command
    floor = command.index("--bapr_v2_train_update_floor")
    tolerance = command.index("--bapr_v2_train_update_tolerance")
    assert command[floor + 1] == "0.0"
    assert command[tolerance + 1] == "0.0"


def test_new_variant_names_copy_into_shared_residual_network():
    previous = base_branch.protocol
    base_branch.protocol = protocol
    try:
        source_config = Config()
        source_config.algo = "regime_sac"
        source_config.env_type = "stochastic_mode"
        source_config.regime_context_source = "robust"
        source_config.task_num = 4
        source_config.test_task_num = 4
        source_config.hidden_dim = 16
        source_config.ensemble_size = 2
        source = RegimeSAC(5, 2, source_config, seed=7)
        tasks = [{"mode_id": mode} for mode in protocol.MODES]
        source.set_task_metadata(tasks)
        target_config = base_branch._adaptive_config(
            source_config,
            "strict_small",
            protocol.branch_run_dir("strict_small", 1103),
        )
        target_config.hidden_dim = 16
        target_config.ensemble_size = 2
        target_config.bapr_v2_context_hidden_dim = 8
        target_config.bapr_v3_context_ensemble_size = 2
        target = make_algo(target_config.algo, 5, 2, target_config)
        target.set_task_metadata(tasks)
        base_branch._copy_source_controller(
            source, target, "strict_small")
        equivalence = base_branch._equivalence(source, target)
    finally:
        base_branch.protocol = previous

    assert equivalence["pass"], equivalence


def test_scheduler_graph_is_30_tasks_and_excludes_unavailable_gpu_nodes():
    rows = submit.candidates("all", "high")
    assert len(rows) == 30
    specs = [spec for _, spec, _ in rows]
    json.dumps(specs)
    gpu = [spec for spec in specs if spec["vram"] > 0]
    cpu = [spec for spec in specs if spec["vram"] == 0]
    assert len(gpu) == 9
    assert len(cpu) == 21
    assert all(set(spec["allowed_nodes"]) == set(submit.GPU_NODES)
               for spec in gpu)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in cpu)
    assert all("jtl110gpu2" not in spec["allowed_nodes"] for spec in specs)
    assert all("jtl311linux" not in spec["allowed_nodes"] for spec in specs)
    assert all(spec["vram"] == 2800 for spec in gpu)
    assert all(
        "run_regime_polarity_conservative_residual_branch" in spec["cmd"]
        for spec in gpu
    )

    calibrations = [
        spec for spec in cpu if "/calibration/" in spec["signature"]
    ]
    audits = [
        spec for spec in cpu if "/audit/" in spec["signature"]
    ]
    analyses = [
        spec for spec in cpu if spec["signature"].endswith("/analysis")
    ]
    assert len(calibrations) == 9
    assert len(audits) == 9
    assert len(analyses) == 3
    assert all(len(spec["wait_for_files"]) == 10
               for spec in calibrations)
    assert all(len(spec["wait_for_files"]) == 13 for spec in audits)
    assert all(len(spec["wait_for_files"]) == 3 for spec in analyses)


def test_rejection_diagnostic_is_one_iteration_and_three_gpu_tasks():
    rows = diagnostic_submit.candidates("high")
    assert len(rows) == 3
    for task_signature, spec, output in rows:
        assert "/diagnostic-v2/" in task_signature
        assert spec["allowed_nodes"] == ["local", "jtl110gpu", "node007"]
        assert spec["vram"] == 2800
        assert spec["cpu"] == 2
        assert "jtl110gpu2" not in spec["allowed_nodes"]
        assert "jtl311linux" not in spec["allowed_nodes"]
        assert (
            "diagnose_regime_polarity_conservative_residual"
            in spec["cmd"])
        assert "SCHEDULEURM_ETA_TOTAL_UNITS=2101" in spec["cmd"]
        assert output.name == "diagnostic.json"
