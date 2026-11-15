"""Run the unmodified V37 controllers on an independent five-seed cohort."""
from __future__ import annotations

import argparse

import jax

from jax_experiments.analysis import reacher_information_confirmation_protocol_v38 as p
from jax_experiments.analysis import run_reacher_information_v37 as shared


def analyze():
    result = shared.summarize(protocol=p)
    primary = result["comparisons"][f"causal_{p.PRIMARY_MODEL}-minus-conditioned"]
    confirmed = primary["ci95"][0] > 0
    result.update({
        "confirmation_evidence": {"positive_primary_paired_ci": confirmed, "primary_wins": primary["wins"],
                                  "primary": "filtered compensation minus filtered conditioned SAC",
                                  "secondary": "instantaneous compensation minus instantaneous conditioned SAC"},
        "inference_unit": "five untouched policy seeds; V35/V36/V37 seeds excluded; episodes are not independent seeds",
        "next": ("integrate the independent matched comparison without retuning"
                 if confirmed else "retain negative or inconclusive confirmation; do not select a replacement primary"),
    })
    shared.common.write_json(p.ANALYSIS, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--seed", type=int, choices=p.POLICY_SEEDS)
    parser.add_argument("--arm", choices=p.ARMS)
    args = parser.parse_args()
    shared.source.require_runtime()
    shared.confirmation.frozen_model_metadata()
    if args.mode in ("train", "audit") and args.seed is None:
        parser.error("train/audit require an untouched confirmation seed")
    if args.mode == "train":
        if args.arm is None:
            parser.error("train requires an arm")
        if jax.default_backend() != "gpu":
            raise ValueError("training requires GPU updates")
        shared.train(args.arm, args.seed, protocol=p)
    elif args.mode == "audit":
        shared.audit(args.seed, protocol=p)
    elif args.mode == "smoke":
        shared.smoke(protocol=p)
    else:
        analyze()
    print(f"REACHER_INFORMATION_CONFIRMATION_COMPLETE mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
