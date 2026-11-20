"""Paired frozen-policy interface and actuator-gain mismatch diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jax_experiments.analysis import run_jmlr_mujoco_suite_v51 as suite

ENVS = ("halfcheetah", "walker2d")
ROUTES = ("bapr_selfcal", "bapr_soft", "oracle")
TRUE_ATTENUATIONS = (0.35, 0.50, 0.65)


def audit(env_key, seed, out, smoke=False):
    runtime = suite.runtime_info(strict=True)
    device = suite.torch.device("cpu")
    suite.torch.set_num_threads(1)
    actor = suite.load_trained_bundle(env_key, "bapr", seed, device)
    models = dict(bapr=actor, bapr_noburst=actor)
    world = suite.make_single_env(env_key)
    dim = world.action_space.shape[0]
    polarity = suite.canonical_polarity_gains(dim)
    lossy = suite.lossy_degradation_gains(dim)
    attenuation_mask = np.abs(lossy) < 1.0
    h, dwell = (64, 16) if smoke else (suite.EPISODE_HORIZON, suite.DWELL_STEPS)
    groups = {"polarity": {r: [] for r in ROUTES}}
    if env_key == "halfcheetah":
        groups.update({f"attenuation_{a:.2f}": {r: [] for r in (
            "bapr_selfcal", "nominal_oracle", "matched_oracle", "uncompensated"
        )} for a in TRUE_ATTENUATIONS})
    try:
        for event in suite.SWITCH_EVENT_SEEDS[:1] if smoke else suite.SWITCH_EVENT_SEEDS:
            for i in range(1 if smoke else 5):
                schedule = np.roll(suite.SWITCH_BASE_SCHEDULES[event], i % 4)
                for group, routes in groups.items():
                    if group == "polarity":
                        actual = assumed = polarity
                        ep_seed = event * 100 + i
                    else:
                        actual = lossy.copy()
                        strength = float(group.rsplit("_", 1)[1])
                        actual[attenuation_mask] = np.sign(lossy[attenuation_mask]) * strength
                        assumed = lossy
                        ep_seed = (event + 70000) * 100 + i
                    for name, rows in routes.items():
                        route = "oracle" if name in ("nominal_oracle", "matched_oracle") else name
                        rows.append(suite.rollout_episode(
                            world, schedule, actual if name == "matched_oracle" else assumed,
                            ep_seed, route, models, device, h, dwell, physical_gains=actual,
                        ))
    finally:
        world.close()
    payload = {"env": env_key, "seed": seed, "runtime": runtime,
               "episodes": {key: suite.compact_episodes(rows) for key, rows in groups.items()}}
    if not smoke:
        suite.write_json(out / env_key / f"seed_{seed}.json", payload)
    return payload


def route_stats(rows):
    switches = [s for r in rows for s in r["switch_records"] if s["reached"]]
    sustained = [s["sustained_delay"] for s in switches if s["sustained_delay"] is not None]
    first = [s["first_delay"] for s in switches if s["first_delay"] is not None]
    n32 = sum(s["steps_32"] for s in switches)
    return {
        "return": float(np.mean([r["return"] for r in rows])),
        "falls": sum(r["terminated_early"] for r in rows),
        "episodes": len(rows),
        "scheduled": sum(r["switches_scheduled"] for r in rows),
        "reached": len(switches),
        "observed_without_sustained": len(switches) - len(sustained) if rows[0]["exact_steps"] is not None else None,
        "first_median": float(np.median(first)) if first else None,
        "sustained_median": float(np.median(sustained)) if sustained else None,
        "error_per_observed_step_32": sum(s["action_error_32"] for s in switches) / n32 if n32 else None,
        "command_per_observed_step_32": sum(s["command_energy_32"] for s in switches) / n32 if n32 else None,
    }


def summarize(root):
    result = {}
    for env in ENVS:
        records = [json.loads((root / env / f"seed_{s}.json").read_text())["episodes"] for s in suite.SEEDS]
        for group, routes in records[0].items():
            oracle = "oracle" if group == "polarity" else "matched_oracle"
            paired_falls = {route: dict(route_only=0, oracle_only=0, both=0) for route in routes}
            for record in records:
                ref_rows = record[group][oracle]
                for name, rows in record[group].items():
                    for r, o in zip(rows, ref_rows, strict=True):
                        if (r["episode_seed"], r["schedule"]) != (o["episode_seed"], o["schedule"]):
                            raise ValueError("Episode pairing differs")
                        rf, of = r["terminated_early"], o["terminated_early"]
                        if rf or of:
                            paired_falls[name]["both" if rf and of else "route_only" if rf else "oracle_only"] += 1
            all_stats = {route: route_stats([r for rec in records for r in rec[group][route]]) for route in routes}
            means = {route: [route_stats(rec[group][route])["return"] for rec in records] for route in routes}
            result[f"{env}/{group}"] = {
                "pooled_episode_diagnostics": all_stats,
                "returns_by_seed": means,
                "paired_falls_vs_matched_oracle": paired_falls,
                "bapr_vs_alternatives": {r: suite.paired_stats(means["bapr_selfcal"], means[r])
                                         for r in routes if r != "bapr_selfcal"},
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("audit", "smoke", "summarize"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--env", choices=ENVS, default=ENVS[0])
    p.add_argument("--seed", type=int, choices=suite.SEEDS, default=suite.SEEDS[0])
    args = p.parse_args()
    if args.mode == "summarize":
        suite.write_json(args.out / "analysis.json", summarize(args.out))
    else:
        audit(args.env, args.seed, args.out, smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
