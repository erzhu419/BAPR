"""Native source controls, interleaved prefetch and compact scheduler DAG."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from jax_experiments.analysis import resac_torch_native_protocol_v43 as p
from jax_experiments.analysis import run_resac_torch_native_v43 as run
from scripts import submit_resac_torch_native_v43 as submit


@pytest.fixture(autouse=True)
def source():
    submit.prepare_source()


def test_original_controls_and_budget():
    full, zero = p.variant("full"), p.variant("no_shift")
    assert full["trainer_kwargs"]["weight_reg"] == 0.01
    full["trainer_kwargs"]["weight_reg"] = 0
    assert full == zero
    assert full["env"] == "HalfCheetah-v2" and full["expl_env_num"] == 4
    assert p.budget() == {"next_epoch": 1000, "total_steps": 1010000,
                          "critic_calls": 1000000, "actor_opportunities": 500000}


def test_interleave_consumes_prefetched_batch_in_source_order():
    calls = []
    runner = object.__new__(run.NativeRun)
    runner.total_steps, runner.train_data = 10000, "initial"
    runner.trainer = SimpleNamespace(networks=[], train=lambda batch: calls.append(("update", batch)))
    batches = iter(("next1", "next2", "next3", "next4"))
    runner.replay = SimpleNamespace(add_paths=lambda paths: calls.append(("store", paths)),
                                   next_batch=lambda size: next(batches))
    runner.collector = SimpleNamespace(collect_new_steps=lambda *args: calls.append(("collect", args)) or "four transitions")
    runner.vector_step()
    assert calls == [("collect", (1000, 1, False)), ("store", "four transitions"),
                     ("update", "initial"), ("update", "next1"), ("update", "next2"), ("update", "next3")]
    assert runner.train_data == "next4" and runner.total_steps == 10004


def row(arm, seed):
    return {"protocol": p.PROTOCOL, "arm": arm, "seed": seed, "runtime": p.RUNTIME,
            "variant": p.variant(arm), "budget": p.budget(),
            "final_evaluation": {"returns": [1.] * 8, "lengths": [1000] * 8, "terminated": [False] * 8}}


def test_validation_rejects_incomplete_native_evaluation():
    value = row("full", p.SEEDS[0])
    p.validate(value, "full", p.SEEDS[0])
    value["final_evaluation"]["lengths"][0] = 999
    with pytest.raises(ValueError, match="incomplete"):
        p.validate(value, "full", p.SEEDS[0])


def test_scheduler_gates_sync_json_and_preserves_remote_checkpoint():
    rows = submit.candidates()
    assert len(rows) == 8
    for task, output in rows:
        assert "local" not in task["allowed_nodes"]
        assert "checkpoints" not in task["result_dir"]
        if "train-" in task["signature"]:
            assert task["wait_for_files"] == [str(p.PREREG), str(p.SMOKE)]
            assert task["ckpt_glob"] == "train_state.pt" and task["resume_managed_by_cmd"]
    assert rows[-1][0]["vram"] == 0 and len(rows[-1][0]["wait_for_files"]) == 7


def test_native_three_seed_analysis_uses_policy_seeds(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "COMPACT", tmp_path)
    for index, seed in enumerate(p.SEEDS):
        for arm in p.ARMS:
            value = row(arm, seed)
            value["final_evaluation"]["returns"] = [float(index + (4 if arm == "full" else 1))] * 8
            p.write(p.result(arm, seed), value)
    run.analyze()
    summary = json.loads((p.COMPACT / "analysis/analysis.json").read_text())
    assert summary["full_minus_zero"]["mean"] == 3
    assert summary["full_minus_zero"]["ci95"] == [3, 3]
    assert summary["full_minus_zero"]["wins"] == 3
