"""Focused checks for the published compact-result recomputation path."""

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_manuscript_results as check


def test_released_statistics_and_budgets(capsys):
    check.main()
    output = capsys.readouterr().out
    assert "frozen composite outcome: False" in output
    assert "975 switching/700 stationary episodes" in output
    assert "30 training records agree" in output
    assert "24 policy audits; four separate studies" in output
    assert check.hc.ANALYSIS_JSON == ROOT / "reproducibility/v32/analysis.json"


@pytest.mark.parametrize("change", ["episode", "interval", "identity"])
def test_halfcheetah_rejects_inconsistent_records(change):
    aggregate, seeds, audits = check.hc.load_payloads()
    if change == "episode":
        first = next(iter(audits[seeds[0]]["switching_holdout"].values()))
        first["canonical_v5_map_compensation"]["returns"][0] += 1
    elif change == "interval":
        aggregate["comparisons"]["causal_vs_equal_budget_sac"]["ci95_low"] += 1
    else:
        audits[seeds[0]]["identity"]["training_seed"] = seeds[1]
    with pytest.raises(ValueError):
        check.hc.validate_aggregate(aggregate, seeds, audits)


@pytest.mark.parametrize("change", ["episode", "stationary", "interval"])
def test_reacher_rejects_inconsistent_records(change):
    root = check.reacher.PACKAGE
    aggregate = json.loads((root / "analysis.json").read_text())
    audits = [json.loads((root / "audits" / f"seed_{seed}" / "audit.json").read_text())
              for seed in aggregate["policy_seeds"]]
    if change == "episode":
        audits[0]["events"]["causal_95103"][0]["return"] += 1
    elif change == "stationary":
        audits[0]["stationary"]["causal_95103"][0]["return"] += 1
    else:
        aggregate["comparisons"]["causal_95103-minus-conditioned"]["ci95"][0] += 1
    with pytest.raises(ValueError):
        check.reacher.validate_results(aggregate, audits)


def test_resac_label_requires_zero_regularization(monkeypatch):
    original_read = check.read

    def changed_read(path):
        value = original_read(path)
        if path.name == "training_metadata.json":
            value = copy.deepcopy(value)
            value["resac_b0"]["88003"]["config"]["weight_reg"] = 0.01
        return value

    monkeypatch.setattr(check, "read", changed_read)
    with pytest.raises(ValueError, match="RE-SAC-0 label"):
        check.check_halfcheetah()


@pytest.mark.parametrize("change", ["episode", "stationary", "interval", "runtime", "budget", "schedule"])
def test_runtime_comparison_rejects_inconsistent_records(change):
    aggregate = check.read(check.RUNTIME_PACKAGE / "analysis.json")
    audits = [check.read(check.RUNTIME_PACKAGE / "audits" / f"seed_{seed}" / "audit.json")
              for seed in check.RUNTIME_SEEDS]
    if change == "episode":
        audits[0]["switching"]["197201"]["bapr"]["returns"][0] += 1
    elif change == "stationary":
        audits[0]["stationary"]["197101"]["bapr"]["0"]["returns"][0] += 1
    elif change == "interval":
        aggregate["comparisons"]["sac"]["ci95"][0] += 1
    elif change == "runtime":
        audits[0]["runtime_versions"]["brax"] = "0.12.1"
    elif change == "budget":
        audits[0]["budgets"]["reference"]["total_steps"] = 5600000
    else:
        audits[0]["switching"]["197201"]["sac"]["episode_sequences"][0] = [0, 1, 2, 3]
    with pytest.raises(ValueError):
        check.validate_runtime_comparison(aggregate, audits)


def test_runtime_comparison_checks_training_versions(monkeypatch):
    original_read = check.read

    def changed_read(path):
        value = original_read(path)
        if path.name == "complete.json":
            value["runtime_versions"]["jax"] = "0.5.3"
        return value

    monkeypatch.setattr(check, "read", changed_read)
    with pytest.raises(ValueError, match="training identity/runtime/budget"):
        check.check_runtime_comparison()


@pytest.mark.parametrize("change", ["episode", "interval", "model", "sign", "budget", "actor"])
def test_extension_package_rejects_inconsistent_records(change):
    summary = check.read(check.EXTENSION_PACKAGE)
    if change == "episode":
        summary["suites"]["permutation"]["audits"][0]["routes"]["causal_118003"]["switching_returns"][0] += 1
    elif change == "interval":
        contrast = summary["suites"]["estimator"]["comparisons"]["causal_95121-minus-conditioned_95121"]
        contrast["ci95"][0] += 1
    elif change == "model":
        summary["suites"]["permutation"]["estimator_seeds"].pop()
    elif change == "sign":
        metadata = summary["suites"]["source"]["audits"][0]["source_metadata"]["full"]
        metadata["config"]["trainer_kwargs"]["weight_reg"] = -.01
    elif change == "budget":
        summary["suites"]["source"]["audits"][0]["source_metadata"]["full"]["budget"]["total_steps"] -= 4000
    else:
        summary["suites"]["source"]["audits"][0]["source_metadata"]["full"]["optimizer_counts"]["actor"] -= 1
    with pytest.raises(ValueError):
        check.validate_extension_package(summary)


@pytest.mark.parametrize("change", ["fwd_mean", "fwd_ci", "cls_mean", "cls_ci"])
def test_identification_baselines_rejects_inconsistent_records(change):
    summary = check.read(check.ID_BASELINES_PACKAGE)
    if change == "fwd_mean":
        summary["forward_vs_inverse_same_split"]["forward_dynamics"]["switching_means"]["learned_soft"] += 1.0
    elif change == "fwd_ci":
        summary["forward_vs_inverse_same_split"]["paired_inverse_minus_forward"]["soft_routing"]["ci95"][0] += 1.0
    elif change == "cls_mean":
        summary["direct_classifier_vs_inverse_same_data"]["means"]["classifier_94103"] += 1.0
    else:
        summary["direct_classifier_vs_inverse_same_data"]["paired_comparisons"]["inverse_minus_classifier_94103"]["ci95"][0] += 1.0
    with pytest.raises(ValueError):
        check.validate_identification_baselines(summary)


def test_jmlr_v50_and_v51_reject_tampering_when_present():
    v50_path = check.V50_PACKAGE / "analysis.json"
    if v50_path.is_file():
        agg = check.read(v50_path)
        seeds = [check.read(check.V50_PACKAGE / "seeds" / f"seed_{s}.json") for s in check.V50_SEEDS]
        bad = copy.deepcopy(agg)
        bad["part1_paired_comparisons"]["bapr_d098_minus_untempered_lockin_h0_d100"]["ci95"][0] += 1.0
        with pytest.raises(ValueError):
            check.validate_jmlr_v50(bad, seeds)

    v51_path = check.V51_PACKAGE / "analysis.json"
    if v51_path.is_file():
        agg51 = check.read(v51_path)
        auds = {
            e: [check.read(check.V51_PACKAGE / "audits" / e / f"seed_{s}.json") for s in check.V51_SEEDS]
            for e in check.V51_ENVS
        }
        bad51 = copy.deepcopy(agg51)
        bad51["environments"]["halfcheetah"]["comparisons"]["bapr_selfcal_minus_sac"]["ci95"][0] += 1.0
        with pytest.raises(ValueError):
            check.validate_jmlr_v51(bad51, auds)
        bad_auds = copy.deepcopy(auds)
        bad_auds["halfcheetah"][0]["external_interactions_bapr"] = 100
        with pytest.raises(ValueError):
            check.validate_jmlr_v51(agg51, bad_auds)
