"""Tests for the late-base min-target fail-fast diagnostic."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_experiments.algos.bapr_regime import BAPRRegime
from jax_experiments.algos.bapr_v2 import _reduce_critic_target
from jax_experiments.analysis import regime_adapter_latebase_min as protocol
from jax_experiments.analysis import (
    run_regime_adapter_latebase_min_branch as runner,
)
from jax_experiments.configs.default import Config
from scripts import submit_regime_adapter_latebase_min as submitter


def _regime_config(**overrides) -> Config:
    config = Config()
    config.algo = "bapr_regime"
    config.task_num = 2
    config.test_task_num = 2
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.bapr_v2_mode = "supervised"
    config.bapr_v2_latent_dim = 2
    config.bapr_v2_policy_mode = "residual"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_base_pretrain_iters = 1
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_variance_model = "mode_empirical"
    config.bapr_regime_inference_iters = 1
    config.bapr_regime_adaptation_source = "oracle"
    config.bapr_v2_critic_target_mode = "min"
    config.bapr_v2_freeze_alpha = True
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_protocol_uses_mature_base_and_minimal_extension() -> None:
    assert protocol.TRAINING_SEEDS == (24, 32)
    assert protocol.SOURCE_NEXT_ITERATION == 2100
    assert protocol.SOURCE_TOTAL_STEPS == 8_400_000
    assert protocol.SOURCE_UPDATE_COUNT == 525_000
    assert protocol.ADDITIONAL_ITERS == 175
    assert protocol.FINAL_NEXT_ITERATION == 2275
    assert protocol.FINAL_TOTAL_STEPS == 9_100_000
    assert protocol.FINAL_UPDATE_COUNT == 568_750
    assert protocol.PER_CONTROLLER_POST_FORK_STEPS == 700_000
    assert protocol.BANK_AGGREGATE_TOTAL_STEPS == 11_200_000


def test_target_reduction_matches_robust_sac_lower_bound() -> None:
    target = jnp.asarray([
        [3.0, -1.0, 5.0],
        [2.0, 4.0, 1.0],
        [7.0, 0.0, 6.0],
    ])
    np.testing.assert_array_equal(
        np.asarray(_reduce_critic_target(target, "independent")),
        np.asarray(target))
    np.testing.assert_array_equal(
        np.asarray(_reduce_critic_target(target, "min")),
        np.asarray([2.0, -1.0, 1.0]))


def test_freeze_alpha_blocks_temperature_update() -> None:
    agent = BAPRRegime(3, 1, _regime_config(), seed=9)
    agent.set_task_metadata([{"mode_id": 0}, {"mode_id": 1}])
    context = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
    context = jnp.broadcast_to(context, (1, 4, 3))
    batch = {
        "obs": jnp.ones((1, 4, 3), dtype=jnp.float32),
        "act": jnp.zeros((1, 4, 1), dtype=jnp.float32),
        "rew": jnp.ones((1, 4, 1), dtype=jnp.float32),
        "next_obs": jnp.full((1, 4, 3), 0.5, dtype=jnp.float32),
        "done": jnp.zeros((1, 4, 1), dtype=jnp.float32),
        "belief": context,
        "next_belief": context,
        "task_id": jnp.zeros((1, 4), dtype=jnp.int32),
    }
    before = np.asarray(agent.log_alpha).copy()
    agent.multi_update(batch, current_iter=2)
    np.testing.assert_array_equal(np.asarray(agent.log_alpha), before)


def test_training_command_seals_target_alpha_and_resume_boundary() -> None:
    command = runner._training_command(
        24, 0, protocol.run_dir(24, 0))
    assert command[command.index("--max_iters") + 1] == "2275"
    assert command[command.index("--min_resume_iteration") + 1] == "2100"
    assert command[
        command.index("--bapr_v2_critic_target_mode") + 1] == "min"
    assert "--bapr_v2_freeze_alpha" in command
    assert runner._expected_steps_at_resume(2100) == 8_400_000
    assert runner._expected_steps_at_resume(2275) == 9_100_000


def test_scheduler_graph_is_dependency_gated_and_excludes_jtl311() -> None:
    preparation = submitter.candidates("prepare", "high")
    training = submitter.candidates("train", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(preparation) == 2
    assert len(training) == 8
    assert len(audits) == 10
    assert len(analysis) == 1
    signatures = [
        row[0] for row in preparation + training + audits + analysis
    ]
    assert len(signatures) == len(set(signatures))

    for _, spec, _ in preparation:
        assert spec["vram"] == 0
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert spec["allow_cpu_training"] is True
    for _, spec, _ in training:
        assert spec["vram"] == submitter.MEASURED_VRAM_MB == 2300
        assert spec["allowed_nodes"] == submitter.GPU_NODES
        assert "node007" in spec["allowed_nodes"]
        assert "jtl311linux" not in spec["allowed_nodes"]
        assert "preferred_node" not in spec
        assert "require_node" not in spec
        assert spec["resume_managed_by_cmd"] is True
        assert len(spec["wait_for_files"]) == 5
        assert "JAX_PLATFORMS=cuda" in spec["cmd"]
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 25
        assert len(spec["stage_input_paths"]) == 5
    assert len(analysis[0][1]["wait_for_files"]) == 10
