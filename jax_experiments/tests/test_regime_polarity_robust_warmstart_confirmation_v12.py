from __future__ import annotations

import sys
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v12 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v12 as specialist_runner,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submit_regime_polarity_robust_warmstart_confirmation_v12 as submitter


def test_v12_seeds_schedules_and_budgets_are_frozen():
    protocol.assert_protocol_integrity()
    assert len(protocol.TRAINING_SEEDS) == 5
    assert len(set(protocol.TRAINING_SEEDS)) == 5
    assert protocol.expected_source_checkpoint() == {
        "iteration": 1399,
        "next_iteration": 1400,
        "total_steps": 5_600_000,
        "update_count": 350_000,
        "algo": "sac",
    }
    assert protocol.expected_checkpoint() == {
        "iteration": 2099,
        "next_iteration": 2100,
        "total_steps": 8_400_000,
        "update_count": 525_000,
        "algo": "sac",
    }
    traces = {
        tuple(protocol.switching_sequence(seed, episode)
              for episode in range(protocol.SWITCHING_EPISODES))
        for seed in protocol.SWITCHING_EVENT_SEEDS
    }
    assert len(traces) == len(protocol.SWITCHING_EVENT_SEEDS)


def test_v12_source_and_specialist_commands_match_the_frozen_recipe():
    source = source_runner.training_command(71003)
    assert source[source.index("--max_iters") + 1] == "1400"
    assert source[source.index("--start_train_steps") + 1] == "4000"
    assert "--stochastic_mode_fixed_id" not in source
    specialist = specialist_runner.training_command(71003, 2)
    assert specialist[specialist.index("--stochastic_mode_fixed_id") + 1] == "2"
    assert specialist[specialist.index("--max_iters") + 1] == "2100"
    assert specialist[specialist.index("--min_resume_iteration") + 1] == "1400"
    assert "--resume" in specialist


def test_v12_syncs_only_policy_and_json_bundles():
    source_names = {
        path.name for path in protocol.source_required_paths(71003)
    }
    specialist_names = {
        path.name
        for path in protocol.bundle_required_paths("actor_only", 71003, 0)
    }
    for names in (source_names, specialist_names):
        assert protocol.POLICY_NAME in names
        assert "params.pkl" not in names
        assert "train_state.pkl" not in names
        assert "replay_buffer.npz" not in names


def test_v12_scheduler_graph_and_vram_are_explicit():
    rows = submitter.candidates("high")
    assert len(rows) == 31
    assert len({signature for signature, _, _ in rows}) == 31
    assert len({str(output) for _, _, output in rows}) == 31
    gpu = [spec for _, spec, _ in rows if int(spec["vram"]) > 0]
    cpu = [spec for _, spec, _ in rows if int(spec["vram"]) == 0]
    assert len(gpu) == 25
    assert len(cpu) == 6
    assert {int(spec["vram"]) for spec in gpu} == {1200}
    assert all("checkpoints" not in spec["result_dir"] for spec in gpu)


def test_v12_scheduler_stages_registration_data_sources():
    data_files = {
        path
        for path in protocol.registration_source_paths()
        if path.suffix != ".py"
    }
    for _, spec, _ in submitter.candidates("high"):
        wait_files = {Path(path) for path in spec["wait_for_files"]}
        stage_roots = [Path(path) for path in spec["stage_input_paths"]]
        assert data_files <= wait_files
        assert all(
            any(path.parent == root or path.parent.is_relative_to(root)
                for root in stage_roots)
            for path in data_files
        )
