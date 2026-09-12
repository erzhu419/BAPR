"""Protocol tests for the fresh robust-anchored development screen."""
from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.algos.anchored_regime_sac import AnchoredRegimeSAC
from jax_experiments.algos.regime_sac import RegimeSAC
from jax_experiments.analysis import (
    calibrate_regime_polarity_anchored_residual as calibration,
)
from jax_experiments.analysis import (
    regime_polarity_anchored_eval as eval_common,
)
from jax_experiments.analysis import (
    regime_polarity_anchored_residual as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_anchored_branch as branch,
)
from jax_experiments.configs.default import Config


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import submit_regime_polarity_anchored_residual as submit


def _source_config():
    config = Config()
    config.algo = "regime_sac"
    config.env_type = "stochastic_mode"
    config.task_num = 4
    config.test_task_num = 4
    config.regime_context_source = "robust"
    config.hidden_dim = 16
    config.ensemble_size = 2
    return config


def _anchored_config():
    config = deepcopy(_source_config())
    config.algo = "anchored_regime_sac"
    config.bapr_v2_mode = "supervised"
    config.bapr_v2_latent_dim = 4
    config.bapr_v2_policy_context_source = "stored"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_policy_mode = "residual"
    config.bapr_v2_critic_target_mode = "min"
    config.bapr_v2_residual_delta = protocol.RESIDUAL_DELTA
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v4_training_source_period = 2
    config.bapr_v4_training_robust_slots = 1
    return config


def test_protocol_uses_fresh_development_split_and_equal_budget():
    historical = {
        8, 16, 24, 32, 40,
        101, 211, 307, 419, 523,
        607, 719, 823, 929, 1031,
    }
    assert not historical.intersection(protocol.TRAINING_SEEDS)
    assert protocol.SOURCE_TOTAL_STEPS == 5_600_000
    assert protocol.BRANCH_TOTAL_STEPS == 8_400_000
    assert protocol.BRANCH_UPDATE_COUNT == 525_000
    assert len(protocol.TRAINING_SEEDS) == 3


def test_source_copy_is_exact_for_every_residual_context():
    source = RegimeSAC(5, 2, _source_config(), seed=17)
    target = AnchoredRegimeSAC(5, 2, _anchored_config(), seed=23)
    tasks = [{"mode_id": mode} for mode in protocol.MODES]
    source.set_task_metadata(tasks)
    target.set_task_metadata(tasks)
    branch.copy_source_controller(source, target)
    equivalence = branch._equivalence(source, target)
    assert equivalence["pass"], equivalence
    assert target.log_alpha.shape == (5,)

    obs = jax.random.normal(jax.random.PRNGKey(31), (16, 5))
    robust = np.asarray(target.policy.base_deterministic(obs))
    for mode in protocol.MODES:
        context = jnp.broadcast_to(
            jnp.concatenate([
                jax.nn.one_hot(mode, 4),
                jnp.ones((1,), dtype=jnp.float32),
            ])[None],
            (len(obs), 5),
        )
        np.testing.assert_array_equal(
            np.asarray(target.policy.deterministic(obs, context)),
            robust,
        )


def test_safe_context_is_exact_fallback_until_calibrated():
    posterior = np.asarray([0.01, 0.02, 0.96, 0.01], dtype=np.float32)
    disabled = np.zeros((4,), dtype=bool)
    context = eval_common.arm_context(
        "learned_safe", 2, posterior, disabled)
    np.testing.assert_array_equal(context, np.zeros((5,), np.float32))

    enabled = disabled.copy()
    enabled[2] = True
    context = eval_common.arm_context(
        "learned_safe", 2, posterior, enabled)
    np.testing.assert_allclose(context[:4], posterior)
    assert context[-1] == 1.0

    uncertain = np.full((4,), 0.25, dtype=np.float32)
    context = eval_common.arm_context(
        "learned_safe", 2, uncertain, np.ones((4,), dtype=bool))
    np.testing.assert_array_equal(context, np.zeros((5,), np.float32))


def test_branch_config_is_one_to_one_and_min_target():
    source = _source_config()
    config = branch._anchored_config(
        source, protocol.branch_run_dir("anchored", 1103))
    assert config.bapr_v4_training_source_period == 2
    assert config.bapr_v4_training_robust_slots == 1
    assert config.bapr_v2_critic_target_mode == "min"
    assert config.bapr_v2_policy_mode == "residual"
    assert config.bapr_v2_residual_delta == 0.5


def test_scheduler_graph_is_file_gated_and_excludes_311linux():
    rows = submit.candidates("all", "high")
    assert len(rows) == 16
    specs = [spec for _, spec, _ in rows]
    json.dumps(specs)
    gpu = [spec for spec in specs if spec["vram"] > 0]
    cpu = [spec for spec in specs if spec["vram"] == 0]
    assert len(gpu) == 9
    assert len(cpu) == 7
    assert all("jtl311linux" not in spec["allowed_nodes"] for spec in specs)
    assert all(
        set(spec["allowed_nodes"]) == set(submit.GPU_NODES)
        for spec in gpu)
    assert all(
        set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
        for spec in cpu)

    branch_specs = [
        spec for spec in gpu
        if "/branch/" in spec["signature"]
    ]
    assert len(branch_specs) == 6
    assert all(len(spec["wait_for_files"]) == 4 for spec in branch_specs)
    calibration_specs = [
        spec for spec in cpu
        if "/calibration/" in spec["signature"]
    ]
    assert all(len(spec["wait_for_files"]) == 10
               for spec in calibration_specs)
    audit_specs = [
        spec for spec in cpu
        if "/audit/" in spec["signature"]
    ]
    assert all(len(spec["wait_for_files"]) == 13 for spec in audit_specs)


def test_calibration_enables_only_positive_safe_modes():
    rows = []
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        for mode in protocol.MODES:
            for arm, value, terminated in (
                    ("robust_continue", 100.0, 0.0),
                    ("anchored_base", 102.0, 0.0),
                    (
                        "oracle_residual",
                        112.0 if mode != 3 else 101.0,
                        0.0,
                    )):
                rows.append({
                    "event_seed": event_seed,
                    "arm": arm,
                    "mode": mode,
                    "returns": [value, value],
                    "terminated_rate": terminated,
                })
    decisions = calibration._mode_decisions(rows)
    assert [row["enabled"] for row in decisions] == [
        True, True, True, False]
