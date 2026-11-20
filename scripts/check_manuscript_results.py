#!/usr/bin/env python3
"""Recompute manuscript statistics and check released training budgets from JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
import plot_v32_figures as hc
import plot_reacher_confirmation as reacher
import summarize_submission_extensions_v47 as extensions

RUNTIME_PACKAGE = ROOT / "reproducibility/halfcheetah_runtime_comparison"
RUNTIME_PROTOCOL = "v39-matched-runtime-regularized-comparison"
RUNTIME_SEEDS = [109003, 109023, 109043, 109063, 109083]
RUNTIME_VERSIONS = {"jax": "0.4.38", "brax": "0.10.5", "flax": "0.10.4", "mujoco": "3.1.6"}
TRAIN_ARMS = ("source", "reference", "sac", "escp", "resac_zero", "resac_full")
EVAL_ARMS = ("source", "sac", "escp", "resac_zero", "resac_full", "no_compensation", "oracle", "bapr")
SWITCH_SCHEDULES = {"197201": [2, 0, 3, 1], "197217": [1, 3, 0, 2], "197233": [3, 2, 1, 0]}
EXTENSION_PACKAGE = ROOT / "reproducibility/submission_extensions_v47/summary.json"
ID_BASELINES_PACKAGE = ROOT / "reproducibility/identification_baselines/summary.json"
V50_PACKAGE = ROOT / "reproducibility/jmlr_brax_mechanism_v50"
V50_SEEDS = [88003, 88021, 88039, 88057, 88079, 88103, 88121, 88139, 88157, 88179]
V51_PACKAGE = ROOT / "reproducibility/jmlr_mujoco_v51"
V51_SEEDS = [51003, 51021, 51039, 51057, 51079]
V51_ENVS = ("halfcheetah", "ant", "walker2d", "hopper")
V51_RUNTIME = {"torch": "2.5.1+cu121", "gymnasium": "1.2.0", "mujoco": "3.2.7"}


def read(path):
    return json.loads(path.read_text())


def check_halfcheetah():
    aggregate, seeds, audits = hc.load_payloads()
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("HalfCheetah confirmation must have ten distinct policy seeds")
    hc.validate_aggregate(aggregate, seeds, audits)
    metadata = read(hc.COMPACT / "training_metadata.json")
    for seed in seeds:
        directory = hc.COMPACT / "reference_policies" / f"seed_{seed}" / "mode_0"
        checkpoint = read(directory / "bundle_manifest.json")["checkpoint"]
        if (checkpoint["next_iteration"], checkpoint["total_steps"], checkpoint["update_count"]) != (
                2100, 8400000, 525000):
            raise ValueError(f"canonical policy budget differs for {seed}")
        for arm in ("robust_long", "escp_recurrent", "resac_b0"):
            run = metadata[arm][str(seed)]
            if run["training_seed"] != seed or (run["next_iteration"], run["total_steps"], run["update_count"]) != (
                    2100, 8400000, 525000):
                raise ValueError(f"baseline budget differs for {arm}/{seed}")
            if arm == "resac_b0" and (run["config"]["weight_reg"], run["config"]["beta_ood"]) != (0, 0):
                raise ValueError(f"RE-SAC-0 label differs from training settings for {seed}")
    print("HalfCheetah: 10 policy seeds; all four final controller budgets 8.4M/525000")
    for name, row in aggregate["comparisons"].items():
        print(f"  {name}: {row['paired_mean']:.3f} [{row['ci95_low']:.3f},{row['ci95_high']:.3f}], "
              f"wins={row['seed_wins']}/10, events={row['event_wins']}/30")
    print(f"  frozen composite outcome: {aggregate['prospective_power_confirmation_pass']}")


def check_reacher():
    aggregate = read(reacher.PACKAGE / "analysis.json")
    seeds = aggregate["policy_seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Reacher confirmation must have five distinct policy seeds")
    audits = [read(reacher.PACKAGE / "audits" / f"seed_{seed}" / "audit.json") for seed in seeds]
    seeds, values = reacher.validate_results(aggregate, audits)
    frozen = read(reacher.PACKAGE / "frozen_estimator_metadata.json")
    switching = stationary = 0
    for audit in audits:
        if audit["runtime_versions"] != frozen["model"]["runtime_versions"]:
            raise ValueError("Reacher audit runtime differs from the frozen estimator")
        for arm in ("reference", "robust", "conditioned", "conditioned_no_filter"):
            budget = audit["budgets"][arm]
            if (budget["next_iteration"], budget["total_steps"], budget["update_count"]) != (
                    400, 1600000, 100000):
                raise ValueError(f"Reacher controller budget differs for {arm}/{audit['seed']}")
        for episodes in audit["events"].values():
            if len(episodes) != 15 or any(e["steps"] != 200 or e["switches_observed"] != 3
                                        or e["termination"] for e in episodes):
                raise ValueError("Reacher switching evaluation lost its registered windows")
            switching += len(episodes)
        stationary += sum(len(episodes) for episodes in audit["stationary"].values())
    if (switching, stationary) != (975, 700):
        raise ValueError("Reacher evaluation coverage differs from the manuscript")
    cost = aggregate["shared_estimator_cost"]
    if (cost["provider_interactions"], cost["provider_updates"], cost["calibration_interactions"],
            cost["estimator_updates"]) != (frozen["provider"]["total_steps"],
                                           frozen["provider"]["update_count"],
                                           frozen["model"]["calibration_interactions"],
                                           frozen["model"]["updates"]):
        raise ValueError("Reacher estimator accounting differs from frozen metadata")
    print(f"Reacher: 5 policy seeds; all 20 controller budgets 1.6M/100000; "
          f"{switching} switching/{stationary} stationary episodes")
    for left, right, label in reacher.PAIRS:
        mean, radius = reacher.interval(values[left] - values[right])
        print(f"  {label.replace(chr(10), ' ')}: {mean:.3f} [{mean-radius:.3f},{mean+radius:.3f}]")
    for right in ("no_filter", "single_head"):
        mean, radius = reacher.interval(values["causal_95103"] - values[right])
        print(f"  filtered compensation minus {right}: {mean:.3f} [{mean-radius:.3f},{mean+radius:.3f}]")


def validate_runtime_comparison(aggregate, audits):
    if (aggregate["protocol"] != RUNTIME_PROTOCOL or aggregate["policy_seeds"] != RUNTIME_SEEDS
            or aggregate["runtime_versions"] != RUNTIME_VERSIONS):
        raise ValueError("runtime comparison aggregate identity differs")
    if any(aggregate[key] is not False for key in
           ("historical_cohorts_pooled", "estimator_refitted", "total_data_matched")):
        raise ValueError("runtime comparison cohort or cost definition differs")
    budgets = {arm: dict(next_iteration=1400, total_steps=5600000, update_count=350000)
               if arm == "source" else dict(next_iteration=2100, total_steps=8400000, update_count=525000)
               for arm in TRAIN_ARMS}
    if [audit["seed"] for audit in audits] != RUNTIME_SEEDS:
        raise ValueError("runtime comparison audit seeds differ")
    for audit in audits:
        if (audit["protocol"] != RUNTIME_PROTOCOL or audit["runtime_versions"] != RUNTIME_VERSIONS
                or audit["budgets"] != budgets):
            raise ValueError("runtime comparison audit identity/runtime/budget differs")
        if (set(audit["switching"]) != set(SWITCH_SCHEDULES)
                or set(audit["stationary"]) != {"197101", "197117", "197133"}):
            raise ValueError("runtime comparison event coverage differs")
        for event, rows in audit["switching"].items():
            sequence = SWITCH_SCHEDULES[event]
            rotations = [sequence[i % 4:] + sequence[:i % 4] for i in range(5)]
            if set(rows) != set(EVAL_ARMS):
                raise ValueError("runtime comparison route coverage differs")
            for row in rows.values():
                if (row["base_schedule"] != sequence or row["episode_sequences"] != rotations
                        or row["mode_counts"] != {str(mode): 1250 for mode in range(4)}):
                    raise ValueError("runtime comparison switching schedules differ")
        for rows in audit["stationary"].values():
            if set(rows) != set(EVAL_ARMS) or any(set(row) != {"0", "1", "2", "3"}
                                                 for row in rows.values()):
                raise ValueError("runtime comparison stationary coverage differs")
        episode_rows = [row for event in audit["switching"].values() for row in event.values()]
        episode_rows += [row for event in audit["stationary"].values()
                         for modes in event.values() for row in modes.values()]
        for row in episode_rows:
            returns = np.asarray(row["returns"])
            if (returns.shape != (5,) or not np.isfinite(returns).all()
                    or row["total_actions"] != 5000 or row["terminated_rate"] != 0
                    or not np.isclose(returns.mean(), row["return_mean"], atol=1e-9, rtol=0)):
                raise ValueError("runtime comparison episode mean or horizon differs")
    values = {arm: np.asarray([np.mean([event[arm]["returns"] for event in audit["switching"].values()])
                               for audit in audits]) for arm in EVAL_ARMS}
    for arm, returns in values.items():
        if not np.isclose(returns.mean(), aggregate["switching_means"][arm], atol=1e-9, rtol=0):
            raise ValueError(f"runtime comparison aggregate mean differs for {arm}")
    for arm in ("resac_full", "sac", "escp", "resac_zero"):
        delta = values["bapr"] - values[arm]
        mean, radius = reacher.interval(delta)
        recorded = aggregate["comparisons"][arm]
        if (not np.allclose([mean, mean-radius, mean+radius],
                            [recorded["paired_mean"], *recorded["ci95"]], atol=1e-9, rtol=0)
                or recorded["seed_wins"] != int(np.sum(delta > 0))
                or recorded["primary"] != (arm == "resac_full")):
            raise ValueError(f"runtime comparison paired interval or identity differs for {arm}")
    return budgets


def check_runtime_comparison():
    aggregate = read(RUNTIME_PACKAGE / "analysis.json")
    audits = [read(RUNTIME_PACKAGE / "audits" / f"seed_{seed}" / "audit.json") for seed in RUNTIME_SEEDS]
    budgets = validate_runtime_comparison(aggregate, audits)
    for arm in TRAIN_ARMS:
        for seed in RUNTIME_SEEDS:
            record = read(RUNTIME_PACKAGE / "training" / arm / f"seed_{seed}" / "complete.json")
            if (record["identity"] != {"protocol": RUNTIME_PROTOCOL, "arm": arm, "seed": seed}
                    or record["runtime_versions"] != RUNTIME_VERSIONS or record["budget"] != budgets[arm]
                    or record["config"]["seed"] != seed):
                raise ValueError(f"runtime comparison training identity/runtime/budget differs for {arm}/{seed}")
            if arm in ("resac_zero", "resac_full"):
                expected = 0 if arm == "resac_zero" else 0.01
                if (record["config"]["weight_reg"], record["config"]["beta_ood"]) != (expected, expected):
                    raise ValueError(f"runtime comparison regularization label differs for {arm}/{seed}")
    print("HalfCheetah runtime-matched follow-up: 5 separate policy seeds; 30 training records agree")
    for arm, row in aggregate["comparisons"].items():
        print(f"  BAPR minus {arm}: {row['paired_mean']:.3f} "
              f"[{row['ci95'][0]:.3f},{row['ci95'][1]:.3f}], wins={row['seed_wins']}/5")


def validate_extension_package(summary):
    protocol = extensions.p
    if (summary["protocol"] != protocol.PROTOCOL or summary["runtime_versions"] != protocol.RUNTIME
            or summary["historical_cohorts_pooled"] is not False
            or summary["event_seeds"] != list(protocol.EVENT_SEEDS)
            or summary["stationary_event_seed"] != protocol.STATIONARY_EVENT_SEED
            or set(summary["suites"]) != set(extensions.SUITES)):
        raise ValueError("extension protocol or cohort definition differs")
    models = {"estimator": list(protocol.REACHER_MODELS), "hc-estimator": list(protocol.HC_MODELS),
              "permutation": list(protocol.PERMUTATION_MODELS), "source": None}
    for suite, (seeds, horizon) in extensions.SUITES.items():
        study = summary["suites"][suite]
        if (study["policy_seeds"] != list(seeds) or study["estimator_seeds"] != models[suite]
                or [row["seed"] for row in study["audits"]] != list(seeds)):
            raise ValueError("extension policy/model roster differs")
        values = {arm: [] for arm in study["means"]}
        for audit in study["audits"]:
            for arm, route in audit["routes"].items():
                switching = np.asarray(route["switching_returns"])
                stationary = np.asarray(route["stationary_returns"])
                if (switching.shape != (15,) or stationary.shape != (20,)
                        or not np.isfinite(switching).all() or not np.isfinite(stationary).all()
                        or route["switching_horizon"] != horizon or route["switches_per_episode"] != 3
                        or route["termination_count"] != 0
                        or not np.allclose([switching.mean(), stationary.mean()],
                            [route["switching_mean"], route["stationary_mean"]], atol=1e-8, rtol=0)):
                    raise ValueError("extension episode coverage or mean differs")
                values[arm].append(float(switching.mean()))
            if "native_reference" in audit["routes"] and audit["routes"]["oracle"]["switching_returns"] != audit["routes"]["native_reference"]["switching_returns"]:
                raise ValueError("extension oracle/native equivalence differs")
            if suite == "source":
                extensions.validate_source(audit["source_metadata"])
                if audit["reference_budget"] != dict(next_iteration=2100, total_steps=8400000, update_count=525000):
                    raise ValueError("extension reference budget differs")
                for metadata in audit["source_metadata"].values():
                    if metadata["optimizer_counts"]["actor"] != 262500:
                        raise ValueError("extension source actor updates incomplete")
            elif suite != "hc-estimator":
                expected = dict(next_iteration=400, total_steps=1600000, update_count=100000)
                if any(budget != expected for budget in audit["controller_budgets"].values()):
                    raise ValueError("extension controller budget differs")
        for arm, returns in values.items():
            if not np.isclose(np.mean(returns), study["means"][arm], atol=1e-8, rtol=0):
                raise ValueError("extension aggregate mean differs")
        for name, saved in study["comparisons"].items():
            left, right = name.split("-minus-")
            computed = extensions.paired(values[left], values[right])
            if (saved["n_policy_seeds"] != len(seeds) or saved["wins"] != computed["wins"]
                    or not np.allclose([saved["mean"], *saved["ci95"]],
                                       [computed["mean"], *computed["ci95"]], atol=1e-8, rtol=0)):
                raise ValueError("extension paired interval or wins differ")


def check_submission_extensions():
    summary = read(EXTENSION_PACKAGE)
    validate_extension_package(summary)
    print("Publication extensions: 24 policy audits; four separate studies; no pooled cohorts")
    for suite, study in summary["suites"].items():
        for name, row in study["comparisons"].items():
            if "conditioned" in name or "source_" in name:
                print(f"  {suite}/{name}: {row['mean']:.3f} "
                      f"[{row['ci95'][0]:.3f},{row['ci95'][1]:.3f}], "
                      f"wins={row['wins']}/{row['n_policy_seeds']}")


def validate_identification_baselines(summary):
    if summary["protocol"] != "bapr-identification-baselines-summary" or summary["historical_cohorts_pooled"] is not False:
        raise ValueError("identification baselines protocol or pooling flag differs")
    fi = summary["forward_vs_inverse_same_split"]
    seeds_5 = [101, 211, 307, 419, 523]
    for side in ("forward_dynamics", "inverse_dynamics"):
        block = fi[side]
        if block["test_controller_seeds"] != seeds_5:
            raise ValueError(f"{side} test seeds differ")
        for arm in ("robust", "oracle", "learned_map", "learned_soft"):
            vals = np.asarray([block["per_seed"][str(s)]["switching"][arm] for s in seeds_5])
            if not np.isclose(vals.mean(), block["switching_means"][arm], atol=1e-9, rtol=0):
                raise ValueError(f"{side} switching mean differs for {arm}")
    for route, key in (("learned_map", "map_routing"), ("learned_soft", "soft_routing")):
        inv_v = np.asarray([fi["inverse_dynamics"]["per_seed"][str(s)]["switching"][route] for s in seeds_5])
        fwd_v = np.asarray([fi["forward_dynamics"]["per_seed"][str(s)]["switching"][route] for s in seeds_5])
        mean, radius = reacher.interval(inv_v - fwd_v)
        rec = fi["paired_inverse_minus_forward"][key]
        if (rec["n_seeds"] != 5 or rec["wins"] != int(np.sum(inv_v > fwd_v))
                or not np.allclose([mean, mean - radius, mean + radius], [rec["mean"], *rec["ci95"]], atol=1e-8, rtol=0)):
            raise ValueError(f"forward vs inverse paired CI differs for {key}")

    dc = summary["direct_classifier_vs_inverse_same_data"]
    if dc["policy_seeds"] != V50_SEEDS[1:] or dc["estimator_seeds"] != [94103, 94121, 94139]:
        raise ValueError("direct classifier seed roster differs")
    for arm, vals in dc["seed_means"].items():
        arr = np.asarray(vals)
        if arr.shape != (9,) or not np.isclose(arr.mean(), dc["means"][arm], atol=1e-8, rtol=0):
            raise ValueError(f"direct classifier mean differs for {arm}")
    for est in (94103, 94121, 94139):
        delta = np.asarray(dc["seed_means"][f"inverse_{est}"]) - np.asarray(dc["seed_means"][f"classifier_{est}"])
        mean, radius = reacher.interval(delta)
        rec = dc["paired_comparisons"][f"inverse_minus_classifier_{est}"]
        if (rec["n_policy_seeds"] != 9 or rec["wins"] != int(np.sum(delta > 0))
                or not np.allclose([mean, mean - radius, mean + radius], [rec["mean"], *rec["ci95"]], atol=1e-8, rtol=0)):
            raise ValueError(f"inverse minus classifier CI differs for {est}")


def check_identification_baselines():
    summary = read(ID_BASELINES_PACKAGE)
    validate_identification_baselines(summary)
    fi = summary["forward_vs_inverse_same_split"]
    soft = fi["paired_inverse_minus_forward"]["soft_routing"]
    print(f"Identification baselines: inverse minus forward soft={soft['mean']:.3f} "
          f"[{soft['ci95'][0]:.3f},{soft['ci95'][1]:.3f}], wins={soft['wins']}/5; "
          f"3 paired classifier contrasts verified (n=9)")


def _check_paired_dict(left_vals, right_vals, recorded, label):
    delta = np.asarray(left_vals, dtype=np.float64) - np.asarray(right_vals, dtype=np.float64)
    mean, radius = reacher.interval(delta)
    if (recorded["n_seeds"] != len(delta) or recorded["wins"] != int(np.sum(delta > 0))
            or not np.allclose([mean, mean - radius, mean + radius], [recorded["mean"], *recorded["ci95"]], atol=1e-8, rtol=0)):
        raise ValueError(f"paired comparison mismatch for {label}")


def validate_jmlr_v50(aggregate, seed_rows):
    if (aggregate["protocol"] != "jmlr-brax-mechanism-v50"
            or aggregate["policy_seeds"] != V50_SEEDS
            or aggregate["runtime_versions"] != RUNTIME_VERSIONS):
        raise ValueError("v50 aggregate metadata mismatch")
    if [r["seed"] for r in seed_rows] != V50_SEEDS or any(r["runtime_versions"] != RUNTIME_VERSIONS for r in seed_rows):
        raise ValueError("v50 per-seed identity or runtime mismatch")

    def verify_group(agg_dict, getter, group_name):
        for arm, saved in agg_dict.items():
            vals = [getter(r)[arm]["switching_mean"] for r in seed_rows]
            if (not np.allclose(vals, saved["switching_seed_means"], atol=1e-9, rtol=0)
                    or not np.isclose(np.mean(vals), saved["switching_mean"], atol=1e-9, rtol=0)):
                raise ValueError(f"v50 switching mean mismatch in {group_name}/{arm}")

    p1 = aggregate["part1_primary_ablations"]
    verify_group(p1, lambda r: r["part1_primary_ablations"], "part1")
    for cname, left, right in (
        ("bapr_d098_minus_instantaneous", "bapr_d098", "instantaneous"),
        ("bapr_d098_minus_untempered_lockin_h0_d100", "bapr_d098", "untempered_lockin_h0_d100"),
        ("tempered_h0_d098_minus_untempered_lockin_h0_d100", "tempered_h0_d098", "untempered_lockin_h0_d100"),
        ("bapr_d098_minus_sticky_hmm_d100", "bapr_d098", "sticky_hmm_d100"),
        ("bapr_d098_minus_single_head_d098", "bapr_d098", "single_head_d098"),
        ("bapr_d098_minus_factored_d098", "bapr_d098", "factored_d098"),
    ):
        _check_paired_dict(
            p1[left]["switching_seed_means"],
            p1[right]["switching_seed_means"],
            aggregate["part1_paired_comparisons"][cname],
            f"v50/{cname}",
        )

    p2_meta = aggregate["part2_hypercube_64"]
    if p2_meta.get("candidate_modes_scored_online") != 64 or p2_meta.get("distinct_modes_sampled") != 43:
        raise ValueError("v50 hypercube mode coverage metadata mismatch")
    p2_all = p2_meta["all_15_episodes"]
    p2_1j = p2_meta["single_joint_flip_episodes"]
    verify_group(p2_all, lambda r: r["part2_hypercube_64"]["all_15_episodes"], "part2_all")
    verify_group(p2_1j, lambda r: r["part2_hypercube_64"]["single_joint_flip_episodes"], "part2_1j")
    for cname, src_d, left, right in (
        ("factored_64_minus_uncompensated_64", p2_all, "factored_64", "uncompensated_64"),
        ("factored_64_minus_joint_64", p2_all, "factored_64", "joint_64"),
        ("single_joint_factored_64_minus_uncompensated_64", p2_1j, "factored_64", "uncompensated_64"),
        ("single_joint_factored_64_minus_joint_64", p2_1j, "factored_64", "joint_64"),
    ):
        _check_paired_dict(
            src_d[left]["switching_seed_means"],
            src_d[right]["switching_seed_means"],
            p2_meta["paired_comparisons"][cname],
            f"v50/{cname}",
        )

    p3 = aggregate["part3_lossy_projection"]["arms"]
    verify_group(p3, lambda r: r["part3_lossy_projection"], "part3")
    decomp = aggregate["part3_lossy_projection"]["two_term_decomposition"]
    _check_paired_dict(p3["projected_oracle"]["switching_seed_means"], p3["native_reference"]["switching_seed_means"],
                       decomp["infeasibility_gap_projected_minus_native"], "v50/infeasibility_gap")
    _check_paired_dict(p3["causal_projected_bapr"]["switching_seed_means"], p3["projected_oracle"]["switching_seed_means"],
                       decomp["inference_loss_causal_minus_projected"], "v50/inference_loss")


def check_jmlr_brax_mechanism_v50():
    aggregate = read(V50_PACKAGE / "analysis.json")
    seed_rows = [read(V50_PACKAGE / "seeds" / f"seed_{s}.json") for s in V50_SEEDS]
    validate_jmlr_v50(aggregate, seed_rows)
    c_lock = aggregate["part1_paired_comparisons"]["bapr_d098_minus_untempered_lockin_h0_d100"]
    c_64 = aggregate["part2_hypercube_64"]["paired_comparisons"]["factored_64_minus_uncompensated_64"]
    c_fj = aggregate["part2_hypercube_64"]["paired_comparisons"]["factored_64_minus_joint_64"]
    print(f"JMLR Brax mechanism v50: 10 policy seeds; BAPR minus h=0 lock-in={c_lock['mean']:.3f} "
          f"(wins={c_lock['wins']}/10); M=64 factored minus uncomp={c_64['mean']:.3f} "
          f"(factored minus joint={c_fj['mean']:.3f}, 43/64 modes sampled)")


def validate_jmlr_v51(aggregate, audits_by_env):
    expected_recipe = {
        "total_steps": 500000,
        "num_envs": 8,
        "warmup_steps": 4800,
        "burst_length": 4,
        "burst_trigger_prob": 0.0125,
        "inverse_updates": 4500,
        "escp_objective": "luo2022_rmdm_consistency_plus_rbf_logdet",
    }
    if (aggregate["protocol"] != "jmlr-mujoco-v5-multimorphology-v51-samestep"
            or aggregate.get("autoreset_mode") != "SAME_STEP"
            or aggregate["seeds"] != V51_SEEDS
            or aggregate.get("recipe") != expected_recipe):
        raise ValueError("v51 protocol, SAME_STEP autoreset, seed list, or recipe mismatch")
    rt_agg = {k: v for k, v in aggregate["runtime_versions"].items() if k in V51_RUNTIME}
    if rt_agg != V51_RUNTIME:
        raise ValueError("v51 runtime mismatch")
    expected_arms = ("bapr", "bapr_noburst", "sac", "escp256", "rma", "belief_sac")
    for env_key in V51_ENVS:
        rows = audits_by_env[env_key]
        if [r["seed"] for r in rows] != V51_SEEDS:
            raise ValueError(f"v51 seed order mismatch in {env_key}")
        for r in rows:
            rt_r = {k: v for k, v in r["runtime_versions"].items() if k in V51_RUNTIME}
            step_map = r.get("real_physics_steps_by_arm", {})
            if (r["protocol"] != "jmlr-mujoco-v5-multimorphology-v51-samestep"
                    or r.get("autoreset_mode") != "SAME_STEP"
                    or r["env"] != env_key
                    or rt_r != V51_RUNTIME
                    or r["online_budget_per_arm"] != 500000
                    or r["external_interactions_bapr"] != 0
                    or any(step_map.get(a) != 500000 for a in expected_arms)):
                raise ValueError(f"v51 SAME_STEP/budget/runtime/zero-external check failed for {env_key}/{r['seed']}")
        env_agg = aggregate["environments"][env_key]
        prim = env_agg["primary_4mode"]
        if prim["bapr_selfcal"]["switches_scheduled_total"] != 225:
            raise ValueError(f"v51 scheduled switches != 225 in {env_key}")
        for route, saved in prim.items():
            vals = [r["primary_4mode"][route]["switching_mean"] for r in rows]
            if (not np.allclose(vals, saved["switching_seed_means"], atol=1e-8, rtol=0)
                    or not np.isclose(np.mean(vals), saved["switching_mean"], atol=1e-8, rtol=0)):
                raise ValueError(f"v51 primary mean mismatch for {env_key}/{route}")
        for comp in (
            "sac", "escp256", "rma", "belief_sac", "uncompensated",
            "bapr_instantaneous", "bapr_d100_lockin", "bapr_forward_shared", "bapr_noburst_selfcal",
        ):
            _check_paired_dict(
                prim["bapr_selfcal"]["switching_seed_means"],
                prim[comp]["switching_seed_means"],
                env_agg["comparisons"][f"bapr_selfcal_minus_{comp}"],
                f"v51/{env_key}/bapr_minus_{comp}",
            )
        _check_paired_dict(
            prim["oracle"]["switching_seed_means"],
            prim["bapr_noburst_oracle"]["switching_seed_means"],
            env_agg["comparisons"]["oracle_burst_minus_oracle_noburst"],
            f"v51/{env_key}/oracle_burst_minus_noburst",
        )


def check_jmlr_mujoco_v51():
    aggregate = read(V51_PACKAGE / "analysis.json")
    audits_by_env = {
        env_key: [read(V51_PACKAGE / "audits" / env_key / f"seed_{s}.json") for s in V51_SEEDS]
        for env_key in V51_ENVS
    }
    validate_jmlr_v51(aggregate, audits_by_env)
    print("JMLR MuJoCo v5 v51 (SAME_STEP): 4 morphologies x 5 seeds; 500k direct steps per arm. "
          "BAPR uses its own replay for calibration; Belief-SAC reuses the paired BAPR estimator.")
    for env_key in V51_ENVS:
        prim = aggregate["environments"][env_key]["primary_4mode"]["bapr_selfcal"]
        csac = aggregate["environments"][env_key]["comparisons"]["bapr_selfcal_minus_sac"]
        cbs = aggregate["environments"][env_key]["comparisons"]["bapr_selfcal_minus_belief_sac"]
        print(f"  {env_key}: BAPR minus SAC={csac['mean']:.2f} (wins={csac['wins']}/5); "
              f"minus Belief-SAC={cbs['mean']:.2f} (wins={cbs['wins']}/5); "
              f"switches={prim['switches_observed_total']}/{prim['switches_scheduled_total']} "
              f"(seeds_w_sw={prim['seeds_with_observed_switches']}/5)")


def main():
    check_halfcheetah()
    check_reacher()
    check_runtime_comparison()
    check_submission_extensions()
    check_identification_baselines()
    check_jmlr_brax_mechanism_v50()
    check_jmlr_mujoco_v51()
    print("Released statistics and budget records agree; this is not a new performance decision.")


if __name__ == "__main__":
    main()
