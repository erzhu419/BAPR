from jax_experiments.analysis import bapr_v7_data_equivalent_option as protocol


def test_bapr_v7_matches_per_option_unique_data_and_replay_support():
    config = protocol.configure("formal")

    assert config.algo == "bapr_v6"
    assert config.max_iters == 5600
    assert config.samples_per_iter == 4000
    assert config.updates_per_iter == 250
    assert config.replay_size == 4_000_000
    assert protocol.FINAL_TOTAL_STEPS == 22_400_000


def test_bapr_v7_smoke_uses_formal_shape_for_vram_calibration():
    smoke = protocol.configure("smoke")
    formal = protocol.configure("formal")

    assert smoke.max_iters == 4
    assert smoke.samples_per_iter == formal.samples_per_iter
    assert smoke.batch_size == formal.batch_size == 256
    assert smoke.hidden_dim == formal.hidden_dim == 256
    assert smoke.ensemble_size == formal.ensemble_size == 10
    assert smoke.replay_size == formal.replay_size == 4_000_000
