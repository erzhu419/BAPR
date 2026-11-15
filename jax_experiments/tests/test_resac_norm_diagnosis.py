"""Constant offsets must be distinguished from state-dependent Q disagreement."""
import numpy as np

from jax_experiments.analysis.diagnose_resac_norm_lcb import disagreement


def test_shared_reward_response_with_head_offsets_has_zero_centered_disagreement():
    values = np.array([10., 20., 30.])[None] + np.array([100., 200., 300.])[:, None]
    result = disagreement(values)
    assert result["raw_std_mean"] > 80
    assert result["centered_std_mean"] == 0
    assert result["offset_variance_fraction"] == 1
    np.testing.assert_allclose(result["head_q_mean"], [120., 220., 320.])


def test_centering_retains_state_dependent_disagreement():
    values = np.array([[1., 2., 3.], [3., 2., 1.]])
    result = disagreement(values)
    assert result["offset_std"] == 0
    assert result["offset_variance_fraction"] == 0
    assert np.isclose(result["raw_std_mean"], 2 / 3)
    assert result["centered_std_mean"] == result["raw_std_mean"]
