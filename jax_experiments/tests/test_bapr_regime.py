import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.bapr_regime import BAPRRegime
from jax_experiments.analysis import analyze_bapr_regime_screen as analysis
from jax_experiments.analysis import bapr_regime_screen as screen
from jax_experiments.analysis.run_bapr_regime_screen_audit import (
    _context_args,
    _expected_rows,
)
from jax_experiments.analysis.run_bapr_regime_screen_controller import (
    training_command,
)
from jax_experiments.configs.default import Config
from jax_experiments.train import make_algo


def regime_config(**overrides):
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
    config.bapr_v2_base_pretrain_iters = 2
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_variance_model = "mode_empirical"
    config.bapr_regime_inference_iters = 3
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_regime_schedule_trains_inference_before_shared_adaptation():
    agent = BAPRRegime(3, 1, regime_config(), seed=3)

    assert agent.training_stage(0) == "robust"
    assert agent.training_stage(1) == "robust"
    assert agent.training_stage(2) == "inference"
    assert agent.training_stage(4) == "inference"
    assert agent.training_stage(5) == "adaptation"
    assert agent.controller_update_flags(2) == (True, False, True)
    assert agent.controller_update_flags(5) == (False, True, True)
    assert agent.rollout_context_source(4) == agent.CONTEXT_ROBUST
    assert agent.rollout_context_source(5) == agent.CONTEXT_LEARNED
    assert not agent.advantage_gate_active(4)
    assert agent.advantage_gate_active(5)


def test_oracle_screen_changes_only_adaptation_context_source():
    config = regime_config(bapr_regime_adaptation_source="oracle")
    agent = BAPRRegime(3, 1, config, seed=4)

    assert agent.rollout_context_source(4) == agent.CONTEXT_ROBUST
    assert agent.rollout_context_source(5) == agent.CONTEXT_ORACLE


def test_adaptation_boundary_is_exact_robust_policy_and_resets_replay():
    agent = BAPRRegime(3, 1, regime_config(), seed=5)
    agent.policy.residual_mean.kernel.value = jnp.ones_like(
        agent.policy.residual_mean.kernel.value)
    agent.policy.residual_mean.bias.value = jnp.ones_like(
        agent.policy.residual_mean.bias.value)

    agent.set_training_iteration(5)

    assert agent.consume_replay_reset_request()
    assert not agent.consume_replay_reset_request()
    assert agent._regime_residual_warmstarted
    np.testing.assert_allclose(
        np.asarray(agent.policy.residual_mean.kernel.value), 0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(agent.policy.residual_mean.bias.value), 0.0, atol=0.0)
    obs = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float32)
    adaptive_context = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(agent.policy.deterministic(obs, adaptive_context)),
        np.asarray(agent.policy.base_deterministic(obs)),
        atol=1e-7,
    )


def test_regime_checkpoint_records_boundary_initialization():
    config = regime_config()
    agent = BAPRRegime(3, 1, config, seed=6)
    agent.set_training_iteration(5)
    state = agent.checkpoint_state()

    resumed = BAPRRegime(3, 1, config, seed=7)
    resumed.load_checkpoint_state(state)

    assert resumed._regime_residual_warmstarted
    assert resumed.training_stage() == "adaptation"
    assert resumed.context_checkpoint_signature()["kind"] == (
        "bapr_regime_v1")


def test_make_algo_exposes_regime_path_without_legacy_dispatch():
    config = regime_config()
    agent = make_algo("bapr_regime", 3, 1, config)
    assert isinstance(agent, BAPRRegime)
    assert isinstance(agent.context_net, nnx.Module)


def test_regime_rejects_policy_banks_and_unmodelled_variance():
    with np.testing.assert_raises_regex(ValueError, "shared bounded residual"):
        BAPRRegime(
            3, 1, regime_config(bapr_v2_policy_mode="categorical_expert"),
            seed=8)
    with np.testing.assert_raises_regex(
            ValueError, "empirical heteroscedastic"):
        BAPRRegime(
            3, 1, regime_config(bapr_v3_variance_model="legacy_state"),
            seed=9)


def test_headroom_screen_stops_before_learned_policy_training():
    assert screen.ROLES == (
        "sac", "escp", "resac", "regime_robust", "regime_oracle")
    assert screen.FINAL_TOTAL_STEPS == 5_600_000
    assert screen.FINAL_UPDATE_COUNT == 350_000

    robust = training_command("regime_robust")
    oracle = training_command("regime_oracle")
    resac = training_command("resac")
    assert robust[robust.index("--bapr_v2_base_pretrain_iters") + 1] == (
        "1401")
    assert oracle[oracle.index("--bapr_v2_base_pretrain_iters") + 1] == (
        "500")
    assert oracle[oracle.index("--bapr_regime_inference_iters") + 1] == (
        "200")
    assert "--no_bapr_regime_advantage_fallback" in oracle
    assert resac[resac.index("--weight_reg") + 1] == "0.01"
    assert resac[resac.index("--beta_ood") + 1] == "0.01"


def test_regime_analysis_parses_csv_boolean_indicators():
    assert analysis._indicator("True") == 1.0
    assert analysis._indicator("False") == 0.0
    assert analysis._indicator(True) == 1.0
    assert analysis._indicator(False) == 0.0
    assert analysis._indicator("0.25") == 0.25


def test_headroom_audit_is_paired_and_keeps_learned_training_gated():
    assert screen.AUDIT_EVENT_SEEDS == (32100, 32200, 32300, 32400, 32500)
    assert _expected_rows("sac") == {
        "summary.csv": 3,
        "task_returns.csv": 4,
        "switching_returns.csv": 5,
        "switching_trace.csv": 5000,
    }
    assert _expected_rows("regime_oracle") == {
        "summary.csv": 15,
        "task_returns.csv": 20,
        "switching_returns.csv": 25,
        "switching_trace.csv": 25000,
    }
    assert "--oracle-context-ladder" in _context_args("regime_oracle")
    assert "--oracle-context-ladder" not in _context_args("regime_robust")

    events = []
    for event_seed in screen.AUDIT_EVENT_SEEDS:
        conditions = {}
        for condition in analysis.CONDITIONS:
            is_dynamic = condition == "oracle_dynamic"
            conditions[condition] = {
                "stationary": 110.0 if is_dynamic else 100.0,
                "switching": 100.0 if is_dynamic else 90.0,
                "stationary_termination_rate": 0.0,
                "switching_termination_rate": 0.0,
                "stationary_by_mode": {
                    str(mode): float(
                        100 if condition == f"fixed_context_{mode}" else 90)
                    for mode in range(4)
                },
            }
        events.append({"event_seed": event_seed, "conditions": conditions})

    comparisons = {
        name: analysis._comparison(events, name)
        for name in analysis.COMPARATORS
    }
    matrix = analysis._fixed_context_matrix(events)
    assert all(
        row["stationary"]["mean"] > 0.0
        and row["switching"]["mean"] > 0.0
        for row in comparisons.values())
    assert matrix["diagonal_wins"] == 4
