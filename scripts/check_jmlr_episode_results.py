#!/usr/bin/env python3
"""Recompute switch coverage, censoring, delays and return contrasts from compact records."""

import statistics

import numpy as np

from check_manuscript_results import (
    ROOT, V50_SEEDS, V51_ENVS, V51_SEEDS, _check_paired_dict, read, reacher,
)


def check_number(got, want, label):
    if got is None or want is None:
        ok = got is want
    else:
        ok = np.isclose(got, want, rtol=0.0, atol=1e-8)
    if not ok:
        raise ValueError(f"{label}: {got!r} != {want!r}")


def recovery(rows):
    records = [s for r in rows for s in r["switch_records"] if s["reached"]]
    out = {}
    for key in ("first", "sustained"):
        values = [s[f"{key}_delay"] for s in records if s[f"{key}_delay"] is not None]
        out[f"{key}_delay_median"] = statistics.median(values) if values else None
        out[f"unresolved_{key}"] = len(records) - len(values)
    return out


def verify_pair(rows, oracle):
    for row, other in zip(rows, oracle, strict=True):
        if (row["episode_seed"], row["schedule"]) != (other["episode_seed"], other["schedule"]):
            raise ValueError("Episode or schedule differs between paired routes")


def paired_switch_recovery(hard_rows, soft_rows):
    """Compare matched switch opportunities and complete 32-step windows."""
    reached = complete = 0
    unresolved = [0, 0]
    recovered_32 = [0, 0]
    for hard, soft in zip(hard_rows, soft_rows, strict=True):
        for a, b in zip(hard["switch_records"], soft["switch_records"], strict=True):
            if a["step"] != b["step"]:
                raise ValueError("Switch time differs between paired routes")
            if not (a["reached"] and b["reached"]):
                continue
            reached += 1
            full_window = a["steps_32"] == b["steps_32"] == 32
            complete += full_window
            for i, record in enumerate((a, b)):
                delay = record["sustained_delay"]
                unresolved[i] += delay is None
                # A run starting at delay 24 occupies steps 24 through 31.
                recovered_32[i] += full_window and delay is not None and delay + 8 <= 32
    return reached, unresolved, complete, recovered_32


def check_mujoco_group(routes, saved, label):
    oracle = routes["oracle"]
    for arm, rows in routes.items():
        verify_pair(rows, oracle)
        attempted = observed = d1_misses = d8_misses = 0
        is_causal = rows[0]["exact_steps"] is not None
        full_regret, cond_regret = [], []
        for row, other in zip(rows, oracle, strict=True):
            attempted += len(row["switch_records"])
            observed += sum(s["reached"] for s in row["switch_records"])
            for rec in row["switch_records"]:
                start = rec["step"]
                if rec["reached"] != (row["steps"] > start):
                    raise ValueError(f"{label}/{arm}: invalid switch censoring")
                if not rec["reached"]:
                    if rec["first_delay"] is not None or rec["sustained_delay"] is not None:
                        raise ValueError(f"{label}/{arm}: recovery on an unreached switch")
                elif is_causal:
                    d1_misses += rec["first_delay"] is None
                    d8_misses += rec["sustained_delay"] is None
            gaps = [o["reward_32"] - r["reward_32"] for r, o in
                    zip(row["switch_records"], other["switch_records"], strict=True)]
            full_regret.append(sum(gaps))
            cond_regret.extend(gap for gap, r, o in zip(gaps, row["switch_records"], other["switch_records"], strict=True)
                               if r["reached"] and o["reached"])
        values = {
            "switching_mean": statistics.fmean(r["return"] for r in rows),
            "termination_rate": statistics.fmean(r["terminated_early"] for r in rows),
            "switches_scheduled_total": attempted, "switches_observed_total": observed,
            **(recovery(rows) if is_causal else {}),
        }
        if arm != "oracle":
            values["post_switch_regret_32"] = statistics.fmean(full_regret)
            values["conditional_post_switch_regret_32"] = statistics.fmean(cond_regret) if cond_regret else None
        if is_causal:
            values["mode_accuracy"] = sum(r["exact_steps"] for r in rows) / sum(r["steps"] for r in rows)
            count = sum(r["post_switch_steps"] for r in rows)
            values["post_switch_mode_accuracy"] = sum(r["post_switch_exact_steps"] for r in rows) / count if count else None
            check_number(d1_misses, values["unresolved_first"], f"{label}/{arm}/first")
            check_number(d8_misses, values["unresolved_sustained"], f"{label}/{arm}/sustained")
        for metric, val in values.items():
            check_number(val, saved[arm][metric], f"{label}/{arm}/{metric}")


def check_mujoco_replays():
    base = ROOT / "reproducibility/jmlr_mujoco_v51/episodes"
    for env in V51_ENVS:
        for seed in V51_SEEDS:
            audit = read(base / env / f"seed_{seed}.json")
            e = audit["episodes"]
            for group in ("primary_4mode", "lossy_projection", "hypercube_2d"):
                if e[group]:
                    saved = audit[group]["routes"] if group == "hypercube_2d" else audit[group]
                    check_mujoco_group(e[group], saved, f"{env}/{seed}/{group}")
            for arm, rows in e["stationary_4mode"].items():
                check_number(statistics.fmean(r["return"] for r in rows),
                             audit["primary_4mode"][arm]["stationary_mean"], f"{env}/{seed}/{arm}/stationary")


def brax_diagnostics(base):
    joint, treated = [], {}
    windows = floor_steps = steps = 0
    margin_pass = error_pass = conjunction = 0
    decay, hazard = 0.98, 0.002
    required = decay**8 * np.log((1 - hazard + hazard / 3) / (hazard / 3)) / sum(decay**n for n in range(8))
    for seed in V50_SEEDS:
        a = read(base / f"seed_{seed}.json")
        e = a["episodes"]
        for group, data in (
            ("primary_4mode", a["part1_primary_ablations"]),
            ("hypercube_64", a["part2_hypercube_64"]["all_15_episodes"]),
            ("lossy_projection", a["part3_lossy_projection"]),
        ):
            for arm, row in data.items():
                eps = e[group][arm]
                recovery_stats = recovery(eps) if eps[0]["mode_accuracy"] is not None else {}
                for metric, val in {
                    "switching_mean": statistics.fmean(r["return"] for r in eps),
                    "total_switches": sum(len(r["switch_records"]) for r in eps),
                    **recovery_stats,
                }.items():
                    check_number(val, row[metric], f"{seed}/{group}/{arm}/{metric}")
        for ep in e["primary_4mode"]["untempered_lockin_h0_d100"]:
            steps += ep["steps"]
            floor_steps += ep["floored_prior_steps"]
        for ep in e["primary_4mode"]["bapr_d098"]:
            for sw in ep["switch_records"]:
                windows += 1
                adequate = sw["min_margin_8"] > required
                accurate = sw["all_heads_within_epsilon_8"]
                margin_pass += adequate
                error_pass += accurate
                conjunction += adequate and accurate
        hypercube = e["hypercube_64"]
        joint.append(statistics.fmean(r["return"] for r in hypercube["joint_64"]))
        for arm, eps in hypercube.items():
            if arm.startswith("factored"):
                verify_pair(eps, hypercube["joint_64"])
                treated.setdefault(arm, []).append(statistics.fmean(r["return"] for r in eps))
    print(f"Brax N=8, epsilon=.1, required margin >{required:.6f}: "
          f"excitation {margin_pass}/{windows}, all errors {error_pass}/{windows}, both {conjunction}/{windows}")
    print(f"Brax zero-hazard lock-in: floor active on {floor_steps}/{steps} action steps")
    for arm, vals in treated.items():
        delta = np.asarray(vals) - joint
        mean, radius = reacher.interval(delta)
        print(f"{arm}: {statistics.fmean(vals):.2f} vs joint {statistics.fmean(joint):.2f}; "
              f"paired {mean:+.2f} [{mean-radius:.2f},{mean+radius:.2f}], {sum(delta > 0)}/10 wins")


def check_frozen():
    root = ROOT / "reproducibility/jmlr_frozen_interface"
    for key, analysis in read(root / "analysis.json").items():
        env, group = key.split("/")
        arms = analysis["returns_by_seed"]
        pooled = {arm: [] for arm in arms}
        oracle = "oracle" if group == "polarity" else "matched_oracle"
        for i, seed in enumerate(V51_SEEDS):
            eps = read(root / env / f"seed_{seed}.json")["episodes"][group]
            for arm, means in arms.items():
                check_number(statistics.fmean(r["return"] for r in eps[arm]), means[i], f"{env}/{seed}/{arm}")
                verify_pair(eps[arm], eps["bapr_selfcal"])
                pooled[arm].extend(eps[arm])
        for arm, row in analysis["bapr_vs_alternatives"].items():
            _check_paired_dict(arms["bapr_selfcal"], arms[arm], row, f"{key}/{arm}")
        for arm, rows in pooled.items():
            records = [s for r in rows for s in r["switch_records"] if s["reached"]]
            observed_steps = sum(s["steps_32"] for s in records)
            expected = analysis["pooled_episode_diagnostics"][arm]
            for metric, value in {
                "falls": sum(r["terminated_early"] for r in rows),
                "reached": len(records),
                "scheduled": sum(r["switches_scheduled"] for r in rows),
                "error_per_observed_step_32": sum(s["action_error_32"] for s in records) / observed_steps,
                "command_per_observed_step_32": sum(s["command_energy_32"] for s in records) / observed_steps,
                "observed_without_sustained": recovery(rows)["unresolved_sustained"]
                if rows[0]["exact_steps"] is not None else None,
            }.items():
                check_number(value, expected[metric], f"{key}/{arm}/{metric}")
            pair = [(r["terminated_early"], o["terminated_early"])
                    for r, o in zip(rows, pooled[oracle], strict=True)]
            for metric, values in (("route_only", (True, False)), ("oracle_only", (False, True)),
                                   ("both", (True, True))):
                check_number(pair.count(values), analysis["paired_falls_vs_matched_oracle"][arm][metric],
                             f"{key}/{arm}/{metric}")

        if key == "walker2d/polarity":
            reached, unresolved, complete, recovered = paired_switch_recovery(
                pooled["bapr_selfcal"], pooled["bapr_soft"])
            print(f"Walker2d matched reached switches, unresolved: hard {unresolved[0]}/{reached}, "
                  f"soft {unresolved[1]}/{reached}")
            print(f"Walker2d matched full 32-step windows, recovered within window: "
                  f"hard {recovered[0]}/{complete}, soft {recovered[1]}/{complete}")
        if group.startswith("attenuation_"):
            delta = np.asarray(arms["matched_oracle"]) - np.asarray(arms["nominal_oracle"])
            mean, radius = reacher.interval(delta)
            print(f"{key} matched minus nominal oracle: "
                  f"{mean:+.1f} [{mean-radius:.1f}, {mean+radius:.1f}]")


if __name__ == "__main__":
    check_frozen()
    check_mujoco_replays()
    brax_diagnostics(ROOT / "reproducibility/jmlr_brax_mechanism_v50/episodes")
    print("Paired cohorts and released episode-level rollup checks passed.")
