"""Restore only the bus reward-batch transform; retain the native Torch recipe."""
import argparse
import json
import math
from statistics import mean, stdev

from jax_experiments.analysis import resac_reward_transfer_protocol_v44 as p
from jax_experiments.analysis import run_resac_torch_native_v43 as native


def bus_reward(torch, reward):
    std = reward.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return 10.0 * (reward - reward.mean(dim=0)) / std


def reward_trainer(api, transform):
    base, torch = api["RESACTrainer"], api["torch"]

    class RewardTrainer(base):
        def train_from_torch(self, batch):
            raw = batch["rewards"]
            reward = bus_reward(torch, raw) if transform == "bus_batch_scale10" else raw
            record = self._need_to_update_eval_statistics
            super().train_from_torch({**batch, "rewards": reward})
            if record:
                self.eval_statistics.update({
                    "Raw Reward Mean": raw.mean().item(), "Raw Reward Std": raw.std().item(),
                    "Training Reward Mean": reward.mean().item(), "Training Reward Std": reward.std().item()})

    return RewardTrainer


class RewardRun(native.NativeRun):
    def __init__(self, arm, seed, api, gpu=True, protocol=p):
        api = dict(api)
        api["RESACTrainer"] = reward_trainer(api, protocol.variant(arm)["reward_transform"])
        super().__init__(arm, seed, api, gpu=gpu, protocol=protocol)


def paired(values):
    radius = math.sqrt(2 * 0.95**2 / (1 - 0.95**2)) * stdev(values) / math.sqrt(3)
    return {"per_seed": values, "mean": mean(values),
            "ci95": [mean(values) - radius, mean(values) + radius], "wins": sum(x > 0 for x in values)}


def analyze():
    rows = {arm: [json.loads(p.result(arm, seed).read_text()) for seed in p.SEEDS] for arm in p.ARMS}
    for arm, values in rows.items():
        for seed, row in zip(p.SEEDS, values):
            p.validate(row, arm, seed)
    returns = {arm: [mean(row["final_evaluation"]["returns"]) for row in values] for arm, values in rows.items()}
    difference = lambda left, right: [a - b for a, b in zip(returns[left], returns[right])]
    raw_gap = difference("raw_full", "raw_no_shift")
    bus_gap = difference("bus_reward_full", "bus_reward_no_shift")
    contrasts = {"raw_full_minus_zero": raw_gap, "bus_full_minus_zero": bus_gap,
        "reward_gain_full": difference("bus_reward_full", "raw_full"),
        "reward_gain_zero": difference("bus_reward_no_shift", "raw_no_shift"),
        "reward_by_shift_interaction": [a - b for a, b in zip(bus_gap, raw_gap)]}
    p.write(p.COMPACT / "analysis/analysis.json", {"protocol": p.PROTOCOL, "seeds": p.SEEDS,
        "returns": returns, "contrasts": {key: paired(value) for key, value in contrasts.items()}, "runs": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "analyze"))
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--seed", type=int, choices=p.SEEDS)
    args = parser.parse_args()
    if args.mode == "analyze":
        analyze()
    elif args.mode == "smoke":
        native.smoke(native.imports(), protocol=p, run_class=RewardRun)
    else:
        native.train(args.arm, args.seed, native.imports(), protocol=p, run_class=RewardRun)


if __name__ == "__main__":
    main()
