"""Registered scale controls and V45 baseline pairing."""
import json
import sys

import numpy as np
import pytest

from jax_experiments import train
from jax_experiments.analysis import resac_jax_reg_scale_protocol_v46 as p
from jax_experiments.analysis import run_resac_jax_reg_scale_v46 as run
from scripts import submit_resac_jax_reg_scale_v46 as submit


@pytest.mark.parametrize("arm,weight", [("bus_reward_reg001", 0.001), ("bus_reward_reg003", 0.003)])
def test_actual_cli_registers_only_positive_scale(arm, weight, monkeypatch):
    command = run.training_command(arm, p.SEEDS[0])
    parsed = []
    monkeypatch.setattr(sys, "argv", ["train.py", *command[command.index("jax_experiments.train") + 1:]])
    monkeypatch.setattr(train, "train", parsed.append)
    train.main()
    config = parsed[0]
    assert config.weight_reg == weight
    assert config.resac_reward_transform == "bus_standardized"
    assert config.seed == p.SEEDS[0]
    assert p.budget(arm) == {"next_iteration": 250, "total_steps": 1010000, "update_count": 1000000}
    metadata = {"identity": p.identity(arm, config.seed), "runtime_versions": p.RUNTIME,
                "norm_definition": p.NORM_DEFINITION, "budget": p.budget(arm),
                "actor_update_count": p.actor_updates(arm), "config": vars(config)}
    p.validate(metadata, arm, config.seed)


def test_scale_profiles_share_all_other_controls():
    profiles = [{key: value for key, value in p.settings(arm).items() if key != "weight_reg"}
                for arm in p.ARMS]
    assert profiles[0] == profiles[1]
    assert [p.settings(arm)["weight_reg"] for arm in p.ARMS] == [0.001, 0.003]


def test_scheduler_has_six_trainers_three_audits_and_fixed_prereg_gate():
    rows = submit.candidates()
    assert len(rows) == 11 and sum(row["vram"] > 0 for row, _ in rows) == 7
    for row, output in rows:
        assert row["signature"].startswith(submit.PREFIX)
        assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"]
        if "train-" in row["signature"]:
            assert row["wait_for_files"] == [str(p.PREREG), str(p.SMOKE)]
            assert row["resume_managed_by_cmd"]
        if "audit-" in row["signature"]:
            assert row["vram"] == 0 and len(row["wait_for_files"]) == 4
    assert len(rows[-1][0]["wait_for_files"]) == 4
