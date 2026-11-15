import importlib.util
import sys
from pathlib import Path

from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as fork
from jax_experiments.analysis import (
    analyze_bapr_v3_stochastic_headroom_audit as analysis,
)
from jax_experiments.analysis import run_bapr_v3_stochastic_headroom as protocol


def _load_submitter():
    scripts = protocol.ROOT / "scripts"
    path = scripts / "submit_bapr_v3_stochastic_headroom.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_stochastic_headroom", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    return module


def test_stochastic_headroom_protocol_isolated_from_old_screen():
    assert protocol.FAMILIES == ("packet_loss", "burst_torque")
    assert protocol.SAVE_ROOT.name == "results_bapr_v3_stochastic_headroom_fork_v1"
    assert protocol.pair_dir("packet_loss", "Ant-v2") == fork.pair_dir(
        "packet_loss", "Ant-v2", 0, protocol.SAVE_ROOT)
    assert set(protocol.FAMILIES) <= set(fork.SUPPORTED_FAMILIES)


def test_training_command_preserves_persistent_stochastic_family(tmp_path):
    pair = fork.pair_dir("packet_loss", "Ant-v2", 0, tmp_path)
    command = fork.training_command(
        "packet_loss", "Ant-v2", 0, pair, "shared_base",
        max_iters=700, base_iters=700, teacher_iters=0)
    joined = " ".join(command)
    assert "--env_type stochastic_mode" in joined
    assert "--stochastic_mode_family packet_loss" in joined
    assert "--stochastic_mode_dwell_steps 500" in joined
    assert "--stochastic_mode_dwell_distribution fixed" in joined


def test_submit_spec_is_unpinned_and_keeps_large_checkpoints_remote():
    module = _load_submitter()
    task = module.task_spec("burst_torque", "HalfCheetah-v2", "high")
    assert task["signature"].startswith(
        "BAPR/v3-stochastic-headroom/v1/burst_torque/HalfCheetah-v2/")
    assert "run_bapr_v3_stochastic_headroom" in task["cmd"]
    assert "--family burst_torque" in task["cmd"]
    assert "--env HalfCheetah-v2" in task["cmd"]
    assert task["allow_initial_resume_scan_error"] is False
    assert task["reroute_on_node_down"] is False
    assert "require_node" not in task
    assert "preferred_node" not in task
    assert "allowed_nodes" not in task
    assert "result_dir" not in task
    assert "local_result_dir" not in task
    assert task["ckpt_dir"].startswith(str(protocol.SAVE_ROOT))


def test_finalize_spec_is_pinned_to_original_runtime_and_cannot_train():
    module = _load_submitter()
    task = module.task_spec(
        "packet_loss", "HalfCheetah-v2", "high", finalize_existing=True)
    assert "--finalize-existing" in task["cmd"]
    assert task["require_node"] == "jtl311linux"
    assert task["require_gpu_idx"] == 1
    assert task["reroute_on_node_down"] is False
    assert "finalize-only" in task["description"]


def test_stochastic_audit_uses_separate_result_tree():
    assert analysis.RESULTS_ROOT.name == (
        "results_bapr_v3_stochastic_headroom_audit_v1")
    assert analysis.RESULTS_ROOT != protocol.SAVE_ROOT
