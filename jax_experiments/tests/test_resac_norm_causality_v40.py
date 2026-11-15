"""Matched treatments, actual update semantics and scheduler dependency checks."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_experiments.algos.resac import RESAC, regularization_bonus, reference_reg_norm
from jax_experiments.configs.default import Config
from jax_experiments.analysis import resac_norm_causality_protocol_v40 as p
from jax_experiments.analysis import run_resac_norm_causality_v40 as run
from jax_experiments.tests.test_resac_reference_semantics import _batch, _config
from scripts import submit_resac_norm_causality_v40 as submit


def test_shared_bonus_keeps_mean_and_positive_sign():
    agent = RESAC(3, 2, _config(ratio=1), seed=37)
    agent.target_critic.layers[0].kernel.value *= jnp.array([1., 2.])[:, None, None]
    per_head = np.asarray(regularization_bonus(agent.target_critic, 0.01, "per_head"))
    shared = np.asarray(regularization_bonus(agent.target_critic, 0.01, "shared"))
    np.testing.assert_allclose(per_head, 0.01 * reference_reg_norm(agent.target_critic))
    np.testing.assert_allclose(shared, np.repeat(per_head.mean(), 2))
    assert shared.min() > 0 and np.ptp(shared) == 0 and np.ptp(per_head) > 0


def test_interventions_start_with_identical_parameters_and_shared_update_is_finite():
    agents = []
    for arm in p.ARMS:
        config = _config(ratio=1)
        config.resac_reg_norm_mode = p.settings(arm)["resac_reg_norm_mode"]
        config.weight_reg = p.settings(arm)["weight_reg"]
        agents.append(RESAC(3, 2, config, seed=41))
    states = [nnx.state(agent.policy, nnx.Param) for agent in agents]
    critics = [nnx.state(agent.critic, nnx.Param) for agent in agents]
    for trees in (states, critics):
        for candidate in trees[1:]:
            for left, right in zip(jax.tree.leaves(trees[0]), jax.tree.leaves(candidate), strict=True):
                np.testing.assert_array_equal(left, right)
    for arm, agent in zip(p.ARMS, agents, strict=True):
        metrics = agent.multi_update(_batch(2))
        assert all(np.isfinite(value) for value in metrics.values())
        if arm == "shared":
            assert metrics["reg_bonus_mean"] > 0
            assert np.isclose(metrics["reg_bonus_std"], 0, atol=1e-6)
        elif arm == "no_shift":
            assert metrics["reg_bonus_mean"] == metrics["reg_bonus_std"] == 0


def test_recipes_only_change_norm_treatment_and_use_fresh_paths():
    commands = [run.training_command(arm, p.SEEDS[0]) for arm in p.ARMS]
    ignored = {"--weight_reg", "--resac_reg_norm_mode", "--save_root", "--run_name"}

    def common(command):
        return [value for index, value in enumerate(command)
                if value not in ignored and not (index and command[index - 1] in ignored)]

    assert common(commands[0]) == common(commands[1]) == common(commands[2])
    for arm, command in zip(p.ARMS, commands, strict=True):
        assert command[command.index("--beta_ood") + 1] == "0.01"
        assert command[command.index("--resac_beta_bc") + 1] == "0.001"
        assert command[command.index("--max_iters") + 1] == "2100"
        assert str(p.RUN_ROOT) in command[command.index("--save_root") + 1]
        assert p.run_dir(arm, p.SEEDS[0]) != p.previous.run_dir("resac_full", p.previous.SEEDS[0])


def test_wrong_treatment_or_budget_is_rejected():
    metadata = dict(identity=p.identity("shared", p.SEEDS[0]), budget=p.budget(),
                    runtime_versions=p.RUNTIME, norm_definition=p.NORM_DEFINITION,
                    config={**vars(Config()), **p.settings("shared"), "seed": p.SEEDS[0]})
    p.validate(metadata, "shared", p.SEEDS[0])
    changed = copy.deepcopy(metadata)
    changed["config"]["resac_reg_norm_mode"] = "per_head"
    with pytest.raises(ValueError, match="treatment"):
        p.validate(changed, "shared", p.SEEDS[0])
    changed = copy.deepcopy(metadata)
    changed["budget"]["total_steps"] -= 4000
    with pytest.raises(ValueError, match="budget"):
        p.validate(changed, "shared", p.SEEDS[0])


def test_scheduler_splits_seeds_and_never_syncs_full_checkpoints():
    rows = submit.candidates()
    assert len(rows) == 14
    assert sum(row["vram"] > 0 for row, output in rows) == 10
    assert len({row["signature"] for row, output in rows}) == 14
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"]
        if "train-" in row["signature"]:
            assert row["wait_for_files"] == [str(p.PREREG), str(p.SMOKE)]
            assert row["resume_managed_by_cmd"]
        if "audit-" in row["signature"]:
            assert row["vram"] == 0 and len(row["wait_for_files"]) == 5
            assert len(row["stage_input_paths"]) == 4
    assert len(rows[-1][0]["wait_for_files"]) == 4
