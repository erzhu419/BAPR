from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as v32,
)
from jax_experiments.analysis import (
    regime_polarity_canonical_training_ablation_v33 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_canonical_training_direct_v33 as direct,
)
from jax_experiments.analysis import (
    run_regime_polarity_canonical_training_warmstart_v33 as warmstart,
)


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _optional_int_flag(command: list[str], name: str, default: int = -1) -> int:
    return int(_flag(command, name)) if name in command else default


def test_v33_uses_new_equal_budget_development_seeds():
    assert len(protocol.TRAINING_SEEDS) == 5
    assert set(protocol.TRAINING_SEEDS).isdisjoint(v32.TRAINING_SEEDS)
    assert protocol.FINAL_TOTAL_STEPS == 8_400_000
    assert protocol.FINAL_UPDATE_COUNT == 525_000
    assert protocol.REQUIRED_RECIPE_WINS == 4


def test_v33_direct_recipe_commands_isolate_training_coordinates():
    seed = protocol.TRAINING_SEEDS[0]
    expected = {
        protocol.ROBUST_RECIPE: (-1, -1),
        protocol.DIRECT_RECIPE: (0, -1),
        protocol.CANONICAL_RECIPE: (-1, 0),
    }
    for recipe, (fixed_mode, canonical_reference) in expected.items():
        command = direct.training_command(recipe, seed)
        assert _optional_int_flag(
            command, "--stochastic_mode_fixed_id") == fixed_mode
        assert _optional_int_flag(
            command,
            "--stochastic_mode_canonical_reference_id",
        ) == canonical_reference
        assert int(_flag(command, "--max_iters")) == 2100
        assert int(_flag(command, "--updates_per_iter")) == 250


def test_v33_warmstart_keeps_historical_two_stage_budget():
    seed = protocol.TRAINING_SEEDS[0]
    command = warmstart.training_command(seed)
    assert int(_flag(command, "--stochastic_mode_fixed_id")) == 0
    assert int(_flag(command, "--max_iters")) == 2100
    assert int(_flag(command, "--min_resume_iteration")) == 1400
    assert int(_flag(command, "--start_train_steps")) == 0


def test_v33_protocol_disables_canonical_interface_at_audit_boundary():
    from jax_experiments.analysis import (
        run_regime_polarity_canonical_training_audit_v33 as audit,
    )

    source = audit._load_controller.__code__.co_names
    assert "stochastic_mode_fixed_id" in source
    assert "stochastic_mode_canonical_reference_id" in source


def test_v33_scheduler_dag_has_expected_shape():
    from scripts import submit_regime_polarity_canonical_training_v33 as submit

    rows = submit.candidates("high")
    assert len(rows) == 31
    specs = [spec for _, spec, _ in rows]
    assert sum(int(spec["vram"]) > 0 for spec in specs) == 25
    assert sum(int(spec["vram"]) == 0 for spec in specs) == 6
    assert all(
        int(spec["vram"]) == 1200
        for spec in specs if int(spec["vram"]) > 0
    )
