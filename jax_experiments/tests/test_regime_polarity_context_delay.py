"""Protocol tests for the checkpoint-only context-delay audit."""
from __future__ import annotations

from jax_experiments.analysis import regime_polarity_context_delay as protocol
from jax_experiments.analysis import regime_polarity_headroom as base
from jax_experiments.analysis.run_regime_polarity_context_delay_audit import (
    _expected_context_rows,
)
from scripts import submit_regime_polarity_context_delay as submitter


def test_protocol_has_only_checkpoint_audits_on_two_candidate_envs():
    assert protocol.ENVS == ("HalfCheetah-v2", "Ant-v2")
    assert len(protocol.FULL_CASES) == 9
    assert len(protocol.DELAY_CASES) == 5
    assert len(protocol.EVALUATION_CASES) == 14
    assert protocol.DELAY_STEPS == (1, 5, 10, 25, 50)
    assert all(case.source_role == "oracle"
               for case in protocol.DELAY_CASES)
    assert all(not case.stationary for case in protocol.DELAY_CASES)


def test_shuffled_maps_are_reproducible_derangements():
    for event_seed in base.AUDIT_EVENT_SEEDS:
        left = protocol.shuffled_mode_map(event_seed)
        right = protocol.shuffled_mode_map(event_seed)
        assert left == right
        assert sorted(left) == list(base.MODES)
        assert all(source != target
                   for source, target in zip(base.MODES, left))


def test_delayed_schedule_validator_uses_exact_action_delay():
    case = protocol.EvaluationCase(
        "delayed_2", "oracle", "delayed",
        delay_steps=2, stationary=False)
    physics = (0, 0, 1, 1, 1, 1)
    contexts = (0, 0, 0, 0, 1, 1)
    remaining = (0, 0, 2, 1, 0, 0)
    rows = [
        {
            "episode": "0",
            "step": str(index + 1),
            "physics_action_task_id": str(physics[index]),
            "action_task_id": str(contexts[index]),
            "eval_context_delay_remaining": str(remaining[index]),
        }
        for index in range(len(physics))
    ]
    _expected_context_rows(rows, case, base.AUDIT_EVENT_SEEDS[0])


def test_scheduler_graph_is_cpu_only_and_file_gated():
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(audits) == 18
    assert len(analysis) == 1
    signatures = [row[0] for row in audits + analysis]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 8
        assert len(spec["stage_input_paths"]) == 2
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
        assert "auto-adopt" not in spec["cmd"]
    assert len(analysis[0][1]["wait_for_files"]) == 18
