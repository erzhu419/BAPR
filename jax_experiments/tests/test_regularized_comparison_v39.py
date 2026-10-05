"""Frozen recipe, resource and dependency checks for the new comparison."""
import copy

import pytest

from jax_experiments.analysis import regularized_comparison_protocol_v39 as p
from jax_experiments.analysis import run_regularized_comparison_v39 as run
from scripts import submit_regularized_comparison_v39 as submit


def option(command, name):
    return command[command.index("--" + name) + 1]


def test_full_resac_enables_both_terms_without_retuning():
    cmd = run.training_command("resac_full", p.SEEDS[0])
    assert option(cmd, "seed") == str(p.SEEDS[0])
    assert option(cmd, "weight_reg") == option(cmd, "beta_ood") == "0.01"
    assert option(cmd, "beta") == "-2.0"
    assert option(cmd, "resac_independent_ratio") == "1.0"
    assert option(cmd, "resac_beta_bc") == "0.001"
    assert "--use_ema_eval" not in cmd and "--resac_adaptive_beta" not in cmd
    assert option(cmd, "max_iters") == "2100" and "--resume" in cmd


def test_zero_and_reference_keep_their_frozen_recipes():
    cmd = run.training_command("resac_zero", p.SEEDS[0])
    assert option(cmd, "weight_reg") == option(cmd, "beta_ood") == "0"
    assert "--use_ema_eval" in cmd and "--resac_adaptive_beta" in cmd
    ref = run.training_command("reference", p.SEEDS[0])
    assert option(ref, "stochastic_mode_fixed_id") == "0"
    assert option(ref, "min_resume_iteration") == "1400"
    assert option(ref, "seed") == str(p.SEEDS[0])


def test_batch_is_split_and_gated_without_gpu_whitelist_or_large_sync():
    rows = submit.candidates()
    assert len(rows) == 37
    assert sum(row["vram"] > 0 for row, unused in rows) == 31
    assert len({row["signature"] for row, unused in rows}) == len(rows)
    for row, output in rows:
        if row["vram"] > 0:
            assert not row["allowed_nodes"]
        else:
            assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"]
        if row["signature"].split("/")[-1].startswith("train-reference"):
            assert len(row["stage_input_paths"]) == 2
            assert any(path.endswith("source/seed_" + row["signature"].split("-")[-1] + "/complete.json")
                       for path in row["wait_for_files"])
        if "/audit-" in row["signature"]:
            assert len(row["wait_for_files"]) == 8
    assert len(rows[-1][0]["wait_for_files"]) == 6


def test_budget_rejects_stale_checkpoint_and_runtime():
    metadata = {"identity": p.identity("sac", p.SEEDS[0]), "runtime_versions": p.RUNTIME,
                "budget": p.budget("sac")}
    run.check_budget(metadata, "sac", p.SEEDS[0])
    stale = copy.deepcopy(metadata)
    stale["budget"]["total_steps"] = 5600000
    with pytest.raises(ValueError, match="incomplete budget"):
        run.check_budget(stale, "sac", p.SEEDS[0])
    stale = copy.deepcopy(metadata)
    stale["runtime_versions"]["brax"] = "0.12.1"
    with pytest.raises(ValueError, match="different identity/runtime"):
        run.check_budget(stale, "sac", p.SEEDS[0])


def test_reference_fork_restores_exact_iteration_and_controller(tmp_path, monkeypatch):
    from jax_experiments.configs.default import Config

    config = Config(seed=p.SEEDS[0], algo="sac", hidden_dim=16, ensemble_size=2, replay_size=8)
    source = run.make_algo("sac", 3, 2, config)
    source.update_count = 350000
    monkeypatch.setattr(p, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(run, "load_agent", lambda arm, seed: (config, source))
    run.bootstrap_reference(p.SEEDS[0])
    directory = p.run_dir("reference", p.SEEDS[0])
    target = run.SACPolicyStability(3, 2, config, seed=p.SEEDS[0])
    replay = run.ReplayBuffer(3, 2, capacity=8)
    iteration, steps = run.load_checkpoint(str(directory / "checkpoints"), target,
                                           replay, run.Logger(str(tmp_path / "restored")), "sac")
    assert (iteration, steps, target.update_count, replay.size) == (1400, 5600000, 350000, 0)
    assert run.fork._controller_equivalence(source, target)["pass"]
    run.bootstrap_reference(p.SEEDS[0])
