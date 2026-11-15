"""Aggregate the registered three-seed bus policy-bank headroom screen."""
from __future__ import annotations

import numpy as np

from bus_experiments import policy_bank_headroom_v1 as protocol


def main() -> None:
    protocol.validate_registration()
    audits = [
        protocol.read_json(protocol.audit_result(seed))
        for seed in protocol.TRAINING_SEEDS
    ]
    for seed, audit in zip(protocol.TRAINING_SEEDS, audits):
        if audit.get("schema") != protocol.AUDIT_SCHEMA:
            raise ValueError(f"unexpected bus audit schema for seed {seed}")
        if int(audit.get("training_seed", -1)) != seed:
            raise ValueError(f"bus audit seed mismatch for {seed}")

    stationary_gains = [
        float(audit["stationary_summary"]["relative_gain"])
        for audit in audits
    ]
    switching_gains = [
        float(audit["switching_summary"]["relative_gain"])
        for audit in audits
    ]
    passes = [bool(audit["passes_policy_seed_gate"]) for audit in audits]
    authorize = all(passes)
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "seed_passes": {
            str(seed): passed
            for seed, passed in zip(protocol.TRAINING_SEEDS, passes)
        },
        "stationary_relative_gain_mean": float(np.mean(stationary_gains)),
        "stationary_relative_gain_min": float(np.min(stationary_gains)),
        "switching_relative_gain_mean": float(np.mean(switching_gains)),
        "switching_relative_gain_min": float(np.min(switching_gains)),
        "policy_seed_pass_count": int(sum(passes)),
        "required_policy_seed_pass_count": len(protocol.TRAINING_SEEDS),
        "authorize_frozen_estimator_training": authorize,
        "decision": (
            "policy-bank headroom passed; train the frozen causal bus mode "
            "estimator next"
            if authorize else
            "policy-bank headroom failed; do not fit an estimator or claim "
            "frozen BAPR bus effectiveness"
        ),
        "audits": {
            str(seed): {
                "stationary": audit["stationary_summary"],
                "switching": audit["switching_summary"],
                "policy_selection": audit["policy_selection"],
            }
            for seed, audit in zip(protocol.TRAINING_SEEDS, audits)
        },
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)

    lines = [
        "# Bus frozen policy-bank headroom V1",
        "",
        "| policy seed | stationary gain | switching gain | pass |",
        "|---:|---:|---:|:---:|",
    ]
    for seed, stationary, switching, passed in zip(
            protocol.TRAINING_SEEDS, stationary_gains, switching_gains, passes):
        lines.append(
            f"| {seed} | {stationary:+.1%} | {switching:+.1%} | "
            f"{'yes' if passed else 'no'} |")
    lines.extend([
        "",
        f"Decision: {payload['decision']}.",
        "",
        "This is a true-mode upper-bound screen, not a learned-adaptation result.",
    ])
    protocol.write_text_atomic(
        protocol.analysis_markdown(), "\n".join(lines) + "\n")
    print(
        "BUS_POLICY_BANK_ANALYSIS_COMPLETE "
        f"seed_passes={sum(passes)}/{len(passes)} "
        f"authorize_estimator={authorize}", flush=True)


if __name__ == "__main__":
    main()
