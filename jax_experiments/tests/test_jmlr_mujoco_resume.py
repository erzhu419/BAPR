"""Check the checkpoint used by the scheduler against uninterrupted training."""

import io
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from jax_experiments.analysis import run_jmlr_mujoco_suite_v51 as suite


@pytest.mark.parametrize("arm", ["bapr", "escp256", "rma"])
def test_checkpoint_only_transfer_reproduces_training(tmp_path, monkeypatch, arm):
    torch.set_num_threads(1)
    monkeypatch.setattr(suite, "BUNDLE_ROOT", tmp_path / "bundles")
    monkeypatch.setattr(suite, "RUN_ROOT", tmp_path / "uninterrupted")
    seed = 51003
    end = 121
    expected = suite.train_arm("hopper", arm, seed, smoke=True, stop_after_iters=end)
    original = suite.run_dir("hopper", arm, seed) / "checkpoints" / suite.CHECKPOINT_NAME

    monkeypatch.setattr(suite, "RUN_ROOT", tmp_path / "interrupted")
    suite.train_arm("hopper", arm, seed, smoke=True, stop_after_iters=108)
    source = suite.run_dir("hopper", arm, seed) / "checkpoints" / suite.CHECKPOINT_NAME

    monkeypatch.setattr(suite, "RUN_ROOT", tmp_path / "destination")
    migrated = suite.run_dir("hopper", arm, seed) / "checkpoints" / suite.CHECKPOINT_NAME
    migrated.parent.mkdir(parents=True)
    shutil.copyfile(source, migrated)
    actual = suite.train_arm("hopper", arm, seed, smoke=True, stop_after_iters=end)
    assert actual == expected

    with np.load(original) as left, np.load(migrated) as right:
        assert left.files == right.files
        for name in left.files:
            if name != "learner_state":
                np.testing.assert_array_equal(left[name], right[name])
        a, b = [
            torch.load(io.BytesIO(data["learner_state"].tobytes()), map_location="cpu", weights_only=False)
            for data in (left, right)
        ]
    assert a["episodes_terminated"] == b["episodes_terminated"] > 0
    assert a["update_count"] == b["update_count"] > 0
    for name in ("actor", "critic", "target_critic", "hist_enc", "ext_enc"):
        if a[name] is not None:
            for key in a[name]:
                torch.testing.assert_close(a[name][key], b[name][key], atol=0, rtol=0)
    torch.testing.assert_close(a["torch_rng"], b["torch_rng"])
    for left_env, right_env in zip(a["vector_env"], b["vector_env"], strict=True):
        np.testing.assert_array_equal(left_env[0], right_env[0])
        assert left_env[1:] == right_env[1:]
    assert a["rng_state"] == b["rng_state"]
    assert a["sample_rng_state"] == b["sample_rng_state"]
