"""Reward-only intervention, checkpoint isolation and fresh-seed factorial DAG."""
import json
from types import SimpleNamespace

import pytest
import torch

from jax_experiments.analysis import resac_reward_transfer_protocol_v44 as p
from jax_experiments.analysis import resac_torch_native_protocol_v43 as old
from jax_experiments.analysis import run_resac_reward_transfer_v44 as run
from jax_experiments.analysis import run_resac_torch_native_v43 as native
from scripts import submit_resac_reward_transfer_v44 as submit


def test_four_arms_change_only_reward_processing_and_weight():
    submit.native.prepare_source()
    assert not set(p.SEEDS) & set(old.SEEDS)
    for arm in p.ARMS:
        config = p.variant(arm)
        transform = config.pop("reward_transform")
        assert transform == ("bus_batch_scale10" if arm.startswith("bus_reward_") else "raw")
        assert config == old.variant("full" if arm.endswith("_full") else "no_shift")
    assert p.budget() == old.budget()


def test_reward_matches_bus_sample_std_and_constant_fallback():
    rewards = torch.tensor([[1.], [2.], [4.], [9.]])
    expected = 10 * (rewards - rewards.mean(dim=0)) / rewards.std(dim=0, unbiased=True)
    got = run.bus_reward(torch, rewards)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(got.std(dim=0), torch.tensor([10.]))
    torch.testing.assert_close(run.bus_reward(torch, torch.ones(256, 1)), torch.zeros(256, 1))
    tiny = torch.tensor([[0.], [1e-7], [2e-7]])
    torch.testing.assert_close(run.bus_reward(torch, tiny), 10 * (tiny - tiny.mean(dim=0)))


@pytest.mark.parametrize("mode", ["raw", "bus_batch_scale10"])
def test_trainer_changes_only_training_reward_without_replay_or_rng_mutation(mode):
    class Source:
        def __init__(self):
            self._need_to_update_eval_statistics = True
            self.eval_statistics = {}

        def train_from_torch(self, batch):
            self.batch = batch
            self._need_to_update_eval_statistics = False

    trainer = run.reward_trainer({"RESACTrainer": Source, "torch": torch}, mode)()
    raw = torch.arange(256, dtype=torch.float32)[:, None]
    before = raw.clone()
    rng = torch.get_rng_state().clone()
    batch = {"rewards": raw, "actions": object(), "observations": object(),
             "next_observations": object(), "terminals": object()}
    trainer.train_from_torch(batch)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key in batch.keys() - {"rewards"}:
        assert trainer.batch[key] is batch[key]
    if mode == "raw":
        assert trainer.batch["rewards"] is raw
    else:
        torch.testing.assert_close(trainer.batch["rewards"], run.bus_reward(torch, raw), rtol=0, atol=0)
        assert trainer.eval_statistics["Training Reward Std"] == pytest.approx(10.)


def test_resume_rejects_old_protocol_or_another_reward_arm():
    runner = object.__new__(native.NativeRun)
    runner.api = dict.fromkeys(("torch", "np", "gym", "ptu"), SimpleNamespace())
    runner.protocol, runner.arm, runner.seed = p, "raw_full", p.SEEDS[0]
    runner.config = p.variant(runner.arm)
    state = {"protocol": old.PROTOCOL, "arm": runner.arm, "seed": runner.seed,
             "variant": runner.config, "runtime": p.RUNTIME}
    with pytest.raises(ValueError, match="another native arm/runtime"):
        runner.restore(state)
    state.update(protocol=p.PROTOCOL, arm="bus_reward_full", variant=p.variant("bus_reward_full"))
    with pytest.raises(ValueError, match="another native arm/runtime"):
        runner.restore(state)


def result(arm, seed, value):
    return {"protocol": p.PROTOCOL, "arm": arm, "seed": seed, "runtime": p.RUNTIME,
        "variant": p.variant(arm), "budget": p.budget(),
        "final_evaluation": {"returns": [value] * 8, "lengths": [1000] * 8, "terminated": [False] * 8}}


def test_factorial_analysis_pairs_policy_seeds_not_evaluation_episodes(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "COMPACT", tmp_path)
    values = {"raw_full": -10., "raw_no_shift": 5., "bus_reward_full": 20., "bus_reward_no_shift": 10.}
    for index, seed in enumerate(p.SEEDS):
        for arm in p.ARMS:
            p.write(p.result(arm, seed), result(arm, seed, values[arm] + index))
    run.analyze()
    summary = json.loads((p.COMPACT / "analysis/analysis.json").read_text())
    expected = {"raw_full_minus_zero": -15., "bus_full_minus_zero": 10.,
        "reward_gain_full": 30., "reward_gain_zero": 5., "reward_by_shift_interaction": 25.}
    for key, value in expected.items():
        assert summary["contrasts"][key]["per_seed"] == [value] * 3
        assert summary["contrasts"][key]["ci95"] == [value, value]


def test_new_scheduler_dag_has_twelve_independent_trainers_and_json_only_sync():
    rows = submit.candidates()
    assert len(rows) == 14
    for task, output in rows:
        assert "local" not in task["allowed_nodes"]
        assert submit.MODULE in task["cmd"]
        assert task["signature"].startswith(submit.PREFIX)
        assert "checkpoints" not in task["result_dir"]
        assert "v43/" not in task["result_dir"]
        if "train-" in task["signature"]:
            assert task["wait_for_files"] == [str(p.PREREG), str(p.SMOKE)]
            assert "results_resac_reward_transfer_v44" in task["ckpt_dir"]
            assert task["resume_managed_by_cmd"] and task["vram"] == 600
    aggregate = rows[-1][0]
    assert aggregate["vram"] == 0 and len(aggregate["wait_for_files"]) == 13
    assert set(aggregate["wait_for_files"][1:]) == {str(p.result(arm, seed)) for seed in p.SEEDS for arm in p.ARMS}
