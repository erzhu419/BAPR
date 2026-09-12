"""Protocol locks for the fresh RE-SAC artifact confirmation."""
from __future__ import annotations

from jax_experiments.analysis import resac_artifact_confirmation_v3 as protocol
from jax_experiments.analysis.run_resac_artifact_confirmation_audit_v3 import (
    audit_command,
)
from jax_experiments.analysis.run_resac_artifact_confirmation_v3 import (
    expected_config,
    training_command,
)
from scripts import submit_resac_artifact_confirmation_v3_20260808 as submit


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_released_budget_and_environment_are_explicit():
    command = training_command("HalfCheetah-v2", "sac", 4109)
    assert _value(command, "--max_iters") == "2000"
    assert _value(command, "--samples_per_iter") == "4000"
    assert _value(command, "--updates_per_iter") == "250"
    assert _value(command, "--start_train_steps") == "10000"
    assert _value(command, "--initial_random_steps") == "0"
    assert _value(command, "--task_scale_distribution") == "exp"
    assert _value(command, "--log_scale_limit") == "3.0"
    assert _value(command, "--changing_period") == "20000"
    assert _value(command, "--changing_interval") == "4000"
    assert protocol.FINAL_TOTAL_STEPS == 8_000_000
    assert protocol.FINAL_UPDATE_COUNT == 499_500


def test_b0_configuration_is_environment_specific():
    halfcheetah = expected_config("HalfCheetah-v2", "resac", 4109)
    ant = expected_config("Ant-v2", "resac", 4109)
    assert halfcheetah["ensemble_size"] == 5
    assert halfcheetah["resac_beta_end"] == 0.0
    assert halfcheetah["resac_anchor_lambda"] == 0.001
    assert ant["ensemble_size"] == 10
    assert ant["resac_beta_end"] == -2.0
    assert ant["resac_anchor_lambda"] == 0.01
    for config in (halfcheetah, ant):
        assert config["weight_reg"] == 0.0
        assert config["beta_ood"] == 0.0
        assert config["resac_independent_ratio"] == 0.75
        assert config["use_ema_eval"] is True
        assert config["use_ema_rollout"] is False


def test_audit_is_strict_heldout_and_requires_final_checkpoint(tmp_path):
    command = audit_command(
        "Ant-v2", "resac", 4109, protocol.EVENT_SEEDS[0], tmp_path)
    assert "--stationary-test-only" in command
    assert _value(command, "--heldout-task-stream") == "validation"
    assert _value(command, "--max-tasks") == "40"
    assert _value(command, "--min-checkpoint-next-iter") == "2000"
    assert _value(command, "--switching-period-steps") == "500"


def test_scheduler_graph_is_unique_and_file_gated():
    rows = submit.candidates("all", "high")
    assert len(rows) == 41
    signatures = [signature for signature, _, _ in rows]
    assert len(set(signatures)) == len(signatures)

    specs = [spec for _, spec, _ in rows]
    gpu = [spec for spec in specs if spec["vram"] > 0]
    audits = [
        spec for spec in specs if "/audit/" in str(spec["signature"])]
    analyses = [
        spec for spec in specs if str(spec["signature"]).endswith("/analysis")]
    assert len(gpu) == 20
    assert len(audits) == 20
    assert len(analyses) == 1
    assert len({str(spec["ckpt_dir"]) for spec in gpu}) == 20
    assert {spec["vram"] for spec in gpu} == {1500, 1750, 1900}
    assert all(spec["allowed_nodes"] == submit.GPU_NODES for spec in gpu)
    assert all(len(spec["wait_for_files"]) == 4 for spec in audits)
    assert len(analyses[0]["wait_for_files"]) == 20
