"""Structural tests for the preregistered 2026-08-03 next-step protocols."""
from __future__ import annotations

import math

from jax_experiments.analysis import analyze_resac_paper_fidelity_smoke_v1
from jax_experiments.analysis import regime_polarity_source_headroom_v1 as headroom
from jax_experiments.analysis import resac_paper_fidelity_smoke_v1 as fidelity
from jax_experiments.analysis.run_regime_polarity_source_controller_v1 import (
    training_command as source_training_command,
)
from jax_experiments.analysis.run_resac_paper_fidelity_smoke_v1 import (
    training_command as fidelity_training_command,
)
from scripts import submit_bapr_next_protocols_20260803 as submit


FINAL_CONFIRMATION_SEEDS = {2009, 2113, 2213, 2311, 2417}


def _value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_fidelity_budget_and_reference_resac_flags_are_frozen():
    command = fidelity_training_command("HalfCheetah-v2", "resac", 4001)
    assert fidelity.FINAL_TOTAL_STEPS == 1_010_000
    assert fidelity.FINAL_UPDATE_COUNT == 1_000_000
    assert _value(command, "--initial_random_steps") == "10000"
    assert _value(command, "--start_train_steps") == "10000"
    assert _value(command, "--samples_per_iter") == "1000"
    assert _value(command, "--updates_per_iter") == "1000"
    assert _value(command, "--weight_reg") == "0.01"
    assert float(_value(command, "--weight_reg")) > 0.0
    assert _value(command, "--beta_bc") == "0.001"
    assert _value(command, "--critic_actor_ratio") == "2"
    assert _value(command, "--clip_norm") == "1.0"
    assert _value(command, "--task_scale_distribution") == "pow1p5"


def test_source_roles_use_switching_or_valid_fixed_modes():
    for role in ("robust_sac", "escp"):
        command = source_training_command(role, 4021)
        assert "--stochastic_mode_fixed_id" not in command
    for mode in headroom.MODES:
        command = source_training_command(f"specialist_{mode}", 4021)
        assert _value(command, "--stochastic_mode_fixed_id") == str(mode)


def test_task_graph_is_file_gated_and_respects_node_policy():
    rows = submit.candidates("all", "all", "high")
    assert len(rows) == 40
    assert len({signature for signature, _, _ in rows}) == 40
    gpu_specs = [spec for _, spec, _ in rows if int(spec["vram"]) > 0]
    cpu_specs = [spec for _, spec, _ in rows if int(spec["vram"]) == 0]
    assert len(gpu_specs) == 24
    assert len(cpu_specs) == 16
    assert all(spec["allowed_nodes"] == ["jtl311linux"] for spec in gpu_specs)
    assert all(spec["allowed_nodes"] == submit.CPU_NODES for spec in cpu_specs)
    assert all("wait_for_files" in spec for spec in cpu_specs)
    assert all(int(spec["vram"]) in {2300, 3200} for spec in gpu_specs)
    assert all(spec.get("allow_gpu_over_one_third") is True
               for spec in gpu_specs)
    assert all(spec.get("resume_managed_by_cmd") is True for spec in gpu_specs)
    assert all(spec.get("resume_flag") == "" for spec in gpu_specs)


def test_development_seeds_are_fresh_and_no_estimator_is_scheduled():
    seeds = set(fidelity.TRAINING_SEEDS) | set(headroom.TRAINING_SEEDS)
    assert not seeds & FINAL_CONFIRMATION_SEEDS
    signatures = [
        signature for signature, _, _ in submit.candidates(
            "all", "all", "high")
    ]
    assert not any("estimator" in signature for signature in signatures)


def test_fidelity_aggregate_propagates_nonfinite_failures():
    aggregate = analyze_resac_paper_fidelity_smoke_v1
    assert math.isnan(aggregate._mean([1.0, math.nan]))
    assert math.isnan(aggregate._sd([1.0, math.nan]))
