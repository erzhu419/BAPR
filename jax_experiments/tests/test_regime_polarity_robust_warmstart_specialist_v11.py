from __future__ import annotations

import sys
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_specialist_v11 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v11 as runner,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submit_regime_polarity_robust_warmstart_specialist_v11 as submitter


def test_v11_splits_and_budget_are_frozen():
    protocol.assert_split_integrity()
    assert protocol.FINAL_NEXT_ITERATION == 2100
    assert protocol.expected_checkpoint() == {
        "iteration": 2099,
        "next_iteration": 2100,
        "total_steps": 8_400_000,
        "update_count": 525_000,
        "algo": "sac",
    }


def test_v11_training_command_is_a_fixed_mode_resume():
    command = runner.training_command("full_state", 61039, 2)
    assert command[command.index("--stochastic_mode_fixed_id") + 1] == "2"
    assert command[command.index("--max_iters") + 1] == "2100"
    assert command[command.index("--min_resume_iteration") + 1] == "1400"
    assert "--resume" in command


def test_v11_syncs_policy_only_bundles():
    paths = protocol.bundle_required_paths("actor_only", 61057, 3)
    names = {path.name for path in paths}
    assert protocol.POLICY_NAME in names
    assert "params.pkl" not in names
    assert "train_state.pkl" not in names
    assert "replay_buffer.npz" not in names


def test_v11_scheduler_graph_and_first_run_vram_are_explicit():
    rows = submitter.candidates("high")
    assert len(rows) == 21
    assert len({signature for signature, _, _ in rows}) == 21
    gpu = [spec for _, spec, _ in rows if int(spec["vram"]) > 0]
    cpu = [spec for _, spec, _ in rows if int(spec["vram"]) == 0]
    assert len(gpu) == 16
    assert len(cpu) == 5
    assert {int(spec["vram"]) for spec in gpu} == {1200}
    assert all("checkpoints" not in spec["result_dir"] for spec in gpu)
