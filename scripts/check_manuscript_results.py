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

RUNTIME_PACKAGE = ROOT / "reproducibility/halfcheetah_runtime_comparison"
RUNTIME_PROTOCOL = "v39-matched-runtime-regularized-comparison"
RUNTIME_SEEDS = [109003, 109023, 109043, 109063, 109083]
RUNTIME_VERSIONS = {"jax": "0.4.38", "brax": "0.10.5", "flax": "0.10.4", "mujoco": "3.1.6"}
TRAIN_ARMS = ("source", "reference", "sac", "escp", "resac_zero", "resac_full")
EVAL_ARMS = ("source", "sac", "escp", "resac_zero", "resac_full", "no_compensation", "oracle", "bapr")
SWITCH_SCHEDULES = {"197201": [2, 0, 3, 1], "197217": [1, 3, 0, 2], "197233": [3, 2, 1, 0]}


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


def main():
    check_halfcheetah()
    check_reacher()
    check_runtime_comparison()
    print("Released statistics and budget records agree; this is not a new performance decision.")


if __name__ == "__main__":
    main()
