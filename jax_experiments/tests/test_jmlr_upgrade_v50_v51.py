"""Unit tests for JMLR upgrade mathematical invariants, SAME_STEP reset/regret/replay logic, and scheduler DAG (v50/v51)."""
from __future__ import annotations

import ast
import itertools
from pathlib import Path
import numpy as np

from scripts import submit_jmlr_upgrade_v50_v51 as submit
from jax_experiments.analysis import run_jmlr_brax_mechanism_v50 as v50_mod

ROOT = Path(__file__).resolve().parents[2]
V51_SOURCE = ROOT / "jax_experiments/analysis/run_jmlr_mujoco_suite_v51.py"


def test_lossy_projection_optimality_and_residual_identity():
    rng = np.random.default_rng(42)
    for _ in range(25):
        d = 6
        gains = rng.choice([-1.0, -0.5, 0.5, 1.0], size=d)
        target = rng.uniform(-1.0, 1.0, size=d)
        # Include zero weights to verify optimality even when uniqueness does not hold
        weights = rng.uniform(0.0, 2.0, size=d)
        weights[0] = 0.0

        proj_cmd = np.clip(target / gains, -1.0, 1.0)
        proj_obj = np.sum(weights * (gains * proj_cmd - target) ** 2)

        candidates = rng.uniform(-1.0, 1.0, size=(200, d))
        cand_objs = np.sum(weights[None, :] * (gains[None, :] * candidates - target[None, :]) ** 2, axis=1)
        assert np.all(proj_obj <= cand_objs + 1e-12)

        lhs = np.sum((gains * proj_cmd - target) ** 2)
        rhs = np.sum(np.maximum(0.0, np.abs(target) - np.abs(gains)) ** 2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_channel_factored_map_matches_separable_product_map():
    rng = np.random.default_rng(7)
    d = 6
    all_signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=d)))  # [64, 6]
    for _ in range(30):
        phi_neg = rng.normal(size=d)
        phi_pos = rng.normal(size=d) + 1e-4

        factored_sign = np.where(phi_pos >= phi_neg, 1.0, -1.0)
        joint_scores = np.sum(
            np.where(all_signs > 0, phi_pos[None, :], phi_neg[None, :]),
            axis=1,
        )
        best_joint_sign = all_signs[int(np.argmax(joint_scores))]
        np.testing.assert_array_equal(factored_sign, best_joint_sign)


def test_tempered_log_odds_cap_and_sticky_min_envelope():
    delta = 0.98
    s_max = 2.5
    cap = s_max / (1.0 - delta)
    l_tempered = 0.0
    l_untempered = 0.0
    for _ in range(500):
        l_tempered = delta * l_tempered + s_max
        l_untempered = l_untempered + s_max
        assert abs(l_tempered) <= cap + 1e-9
    assert l_untempered == 500 * s_max
    assert l_tempered < 125.0 + 1e-6

    # Verify Lean's min(c*m, filterMarginEnvelope) lower bound on sticky filter log-odds
    h = 0.002
    m_modes = 4
    a = 1.0 - h - h / (m_modes - 1)
    k = h / (m_modes - 1)
    b = np.array([1e-6, 1.0 - 3e-6, 1e-6, 1e-6], dtype=np.float64)
    B_floor = np.log((1.0 - h + k) / k)
    c_scale = 1.0
    margin = 4.0
    env_val = -B_floor
    for _ in range(1, 15):
        prior = a * b + k
        ev = np.array([margin, 0.0, 0.0, 0.0])
        logits = delta * np.log(prior) + c_scale * ev
        logits -= np.max(logits)
        b = np.exp(logits) / np.sum(np.exp(logits))
        env_val = delta * env_val + c_scale * margin
        actual_log_odds = np.log(b[0] / b[1])
        # Without min(c*m, env_val), env_val eventually exceeds actual_log_odds; with min(c*m, env_val) it ALWAYS holds!
        assert actual_log_odds >= min(c_scale * margin, env_val) - 1e-9


def test_v50_hypercube_distinct_mode_coverage():
    distinct = {int(m) for _, seq in v50_mod.schedules_64(smoke=False) for m in seq}
    assert len(distinct) == 43


def test_v51_source_enforces_same_step_rmdm_and_recipe_constants():
    txt = V51_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(txt)
    assigns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assigns[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    assert assigns["PROTOCOL"] == "jmlr-mujoco-v5-multimorphology-v51-samestep"
    assert assigns["AUTORESET_MODE"] == "SAME_STEP"
    assert assigns["TOTAL_STEPS"] == 500_000
    assert assigns["BURST_LEN"] == 4
    assert assigns["BURST_PROB"] == 0.0125
    assert assigns["INVERSE_UPDATES"] == 4500
    assert assigns["ARMS"] == ("bapr", "bapr_noburst", "sac", "escp256", "rma", "belief_sac")
    assert "AutoresetMode.SAME_STEP" in txt
    assert "torch.linalg.cholesky" in txt
    assert "F.cross_entropy" not in txt
    assert assigns["CHECKPOINT_NAME"] == submit.V51_CHECKPOINT_NAME


def test_v50_v51_scheduler_dag_invariants():
    rows = submit.build_candidates()
    # 10 v50 seeds + 1 v50 agg + 4*5*(6 train + 1 audit) + 1 v51 agg = 11 + 140 + 1 = 152 tasks
    assert len(rows) == 152
    sigs = [r["signature"] for r, _ in rows]
    assert len(set(sigs)) == len(sigs)

    for r, _ in rows:
        assert "local" not in r["allowed_nodes"]
        assert "checkpoints" not in r["result_dir"]
        assert str(submit.PREREG) in r["wait_for_files"]
        if "-train-" in r["signature"]:
            assert r["ckpt_glob"] == submit.V51_CHECKPOINT_NAME
        if "train-belief_sac-" in r["signature"]:
            assert any(w.endswith("/bapr/seed_" + r["signature"].split("-")[-1] + "/bundle.json")
                       for w in r["wait_for_files"])
        if "-audit-" in r["signature"]:
            assert len(r["wait_for_files"]) == 7  # PREREG + 6 arm bundles


def test_posterior_mean_polarity_command_minimizes_single_step_error():
    tree = ast.parse(V51_SOURCE.read_text())
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "polarity_mean_command")
    namespace = {}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(V51_SOURCE), "exec"), namespace)
    command_rule = namespace["polarity_mean_command"]

    gains = np.array([[1., -1.], [-1., 1.]])
    target = np.array([.75, -.5])
    for belief in (np.array([.5, .5]), np.array([.85, .15]), np.array([0., 1.])):
        command = command_rule(belief, gains, target)
        expected = (2 * belief[0] - 1) * target * gains[0]
        np.testing.assert_allclose(command, expected, atol=1e-12)
        def risk(a):
            return belief @ np.sum((gains * a - target) ** 2, axis=1)

        assert risk(command) < min(risk(command + .01), risk(command - .01))
