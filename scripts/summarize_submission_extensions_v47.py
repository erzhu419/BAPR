"""Reduce completed V47 audits to small, independently recomputable results."""
import json
import math
from pathlib import Path
import statistics
import sys

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jax_experiments.analysis import submission_extensions_protocol_v47 as p

PACKAGE = ROOT / "reproducibility/submission_extensions_v47"
SUITES = {"estimator": (p.REACHER_SEEDS, 200), "hc-estimator": (p.HC_SEEDS, 1000),
          "permutation": (p.PERMUTATION_SEEDS, 200), "source": (p.SOURCE_SEEDS, 1000)}


def read(path):
    return json.loads(path.read_text())


def summarize_route(events, stationary, horizon):
    if len(events) != 15 or len(stationary) != 20:
        raise ValueError("incomplete paired switching/stationary coverage")
    if any(row["steps"] != horizon or row["switches_observed"] != 3
           or row["termination"] for row in events):
        raise ValueError("switching audit lost its horizon or switch windows")
    if any(row["steps"] != horizon or row["switches_observed"] != 0
           or row["termination"] for row in stationary):
        raise ValueError("stationary audit differs from the fixed-mode protocol")
    returns = [row["return"] for row in events]
    static_returns = [row["return"] for row in stationary]
    if not all(math.isfinite(value) for value in returns + static_returns):
        raise ValueError("non-finite return")
    accuracy = [row["mode_accuracy"] for row in events if row["mode_accuracy"] is not None]
    switches = [row for episode in events for row in episode["delays"]]
    detected = [row["delay"] for row in switches if row["delay"] is not None]
    return {"switching_returns": returns, "stationary_returns": static_returns,
            "switching_mean": statistics.fmean(returns),
            "stationary_mean": statistics.fmean(static_returns),
            "switching_horizon": horizon, "switches_per_episode": 3,
            "termination_count": 0,
            "mode_accuracy": statistics.fmean(accuracy) if accuracy else None,
            "detected_delays": detected, "unresolved_switches": len(switches) - len(detected)}


def paired(left, right):
    differences = [a - b for a, b in zip(left, right, strict=True)]
    mean = statistics.fmean(differences)
    radius = student_t.ppf(.975, len(differences) - 1) * statistics.stdev(differences) / math.sqrt(len(differences))
    return {"n_policy_seeds": len(differences), "mean": mean,
            "ci95": [mean - radius, mean + radius],
            "wins": sum(value > 0 for value in differences), "differences": differences}


def validate_source(metadata):
    for arm, row in metadata.items():
        if row["budget"] != p.source_budget() or row["torch"] != "2.4.1+cu121":
            raise ValueError("source trainer budget or Torch runtime differs")
        config = row["config"]["trainer_kwargs"]
        expected = .01 if arm == "full" else 0.
        if (config["weight_reg"], config["beta"], config["beta_ood"], config["beta_bc"]) != (expected, -2., .01, .001):
            raise ValueError("source regularization configuration differs")
    full, no_shift = [dict(metadata[arm]["config"]["trainer_kwargs"]) for arm in p.SOURCE_ARMS]
    full.pop("weight_reg")
    no_shift.pop("weight_reg")
    if full != no_shift:
        raise ValueError("no-shift changed more than the registered coefficient")


def build_summary():
    result = {"protocol": p.PROTOCOL, "runtime_versions": p.RUNTIME,
              "event_seeds": list(p.EVENT_SEEDS), "stationary_event_seed": p.STATIONARY_EVENT_SEED,
              "historical_cohorts_pooled": False,
              "inference_unit": "paired policy-seed means; estimator seeds are crossed factors",
              "suites": {}}
    for suite, (seeds, horizon) in SUITES.items():
        aggregate = read(p.analysis_path(suite))
        if aggregate["policy_seeds"] != list(seeds) or aggregate["protocol"] != p.PROTOCOL:
            raise ValueError(f"mixed aggregate: {suite}")
        audits = []
        for seed in seeds:
            row = read(p.audit_path(suite, seed))
            if (row["seed"], row["suite"], row["protocol"], row["runtime_versions"]) != (
                    seed, suite, p.PROTOCOL, p.RUNTIME):
                raise ValueError(f"mixed audit: {suite}/{seed}")
            routes = {arm: summarize_route(events, row["stationary"][arm], horizon)
                      for arm, events in row["events"].items()}
            if "native_reference" in routes:
                if routes["oracle"]["switching_returns"] != routes["native_reference"]["switching_returns"]:
                    raise ValueError("true-mode/native return equivalence failed")
            compact = {"seed": seed, "routes": routes}
            if suite == "source":
                validate_source(row["source_metadata"])
                compact["source_metadata"] = row["source_metadata"]
                reference = row["reference_metadata"]
                if reference["budget"] != aggregate["budgets"]["bapr"] or reference["runtime_versions"] != p.RUNTIME:
                    raise ValueError("paired canonical controller budget/runtime differs")
                compact["reference_budget"] = reference["budget"]
            elif suite != "hc-estimator":
                for arm, metadata in row["budgets"].items():
                    if (metadata["next_iteration"], metadata["total_steps"], metadata["update_count"]) != (400, 1600000, 100000):
                        raise ValueError(f"controller budget differs: {suite}/{seed}/{arm}")
                compact["controller_budgets"] = {arm: {key: value[key] for key in (
                    "next_iteration", "total_steps", "update_count")} for arm, value in row["budgets"].items()}
            audits.append(compact)
        values = {arm: [row["routes"][arm]["switching_mean"] for row in audits] for arm in audits[0]["routes"]}
        for arm, means in values.items():
            if not all(math.isclose(x, y, rel_tol=0, abs_tol=1e-8)
                       for x, y in zip(means, aggregate["seed_means"][arm], strict=True)):
                raise ValueError("episode returns disagree with cached seed means")
        comparisons = {}
        for name, cached in aggregate["comparisons"].items():
            left, right = name.split("-minus-")
            computed = paired(values[left], values[right])
            if computed["wins"] != cached["wins"] or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
                    for a, b in zip([computed["mean"], *computed["ci95"]], [cached["mean"], *cached["ci95"]], strict=True)):
                raise ValueError("paired interval disagrees with the raw episode audit")
            comparisons[name] = computed
        result["suites"][suite] = {"scope": aggregate["scope"], "policy_seeds": list(seeds),
            "estimator_seeds": aggregate.get("estimator_seeds"), "audits": audits,
            "means": {arm: statistics.fmean(values[arm]) for arm in values}, "comparisons": comparisons}
    return result


def main():
    summary = build_summary()
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for suite, row in summary["suites"].items():
        print(suite, {arm: round(value, 3) for arm, value in row["means"].items()})
    print(f"Wrote {PACKAGE / 'summary.json'}")


if __name__ == "__main__":
    main()
