"""Export a small fixed-batch trace from the original, unmodified Torch trainer."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

MINE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(MINE / "dsac"), str(MINE / "RE-SAC/experiment")]
from resac_networks import VectorizedQNetwork
from resac_trainer import RESACTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy


def named_tensors(model, policy=False):
    if policy:
        for index, layer in enumerate(model.fcs):
            yield f"layers/{index}/kernel", layer.weight, True
            yield f"layers/{index}/bias", layer.bias, False
        for name, layer in (("mean_head", model.last_fc), ("log_std_head", model.last_fc_log_std)):
            yield name + "/kernel", layer.weight, True
            yield name + "/bias", layer.bias, False
    else:
        for index, layer in enumerate(model.net):
            if hasattr(layer, "ensemble_size"):
                number = (0, 3, 5, 7).index(index)
                yield f"layers/{number}/kernel", layer.weight, False
                yield f"layers/{number}/bias", layer.bias, False
            elif isinstance(layer, torch.nn.LayerNorm):
                yield "norm/scale", layer.weight, False
                yield "norm/bias", layer.bias, False


def snapshot(output, prefix, trainer, moments):
    for name, model, optimizer in (("policy", trainer.policy, trainer.policy_optimizer),
                                    ("critic", trainer.qf, trainer.qf_optimizer),
                                    ("target_critic", trainer.target_qf, None)):
        for key, tensor, transpose in named_tensors(model, name == "policy"):
            value = tensor.detach().numpy()
            output[f"{prefix}/{name}/{key}"] = (value.T if transpose else value).copy()
            if moments and optimizer is not None:
                for label, field in (("m", "exp_avg"), ("v", "exp_avg_sq")):
                    value = optimizer.state[tensor][field].detach().numpy()
                    output[f"{prefix}/{name}_adam_{label}/{key}"] = (value.T if transpose else value).copy()
    output[prefix + "/log_alpha"] = trainer.log_alpha.detach().numpy().copy()
    if moments:
        for label, field in (("m", "exp_avg"), ("v", "exp_avg_sq")):
            output[prefix + "/alpha_adam_" + label] = trainer.alpha_optimizer.state[
                trainer.log_alpha][field].detach().numpy().copy()


def export(destination):
    torch.set_num_threads(2)
    rng = np.random.default_rng(2027)
    steps, batch_size, obs_dim, act_dim = 4, 6, 3, 2
    output = {name: rng.normal(size=(steps, batch_size, dim)).astype(np.float32)
              for name, dim in (("obs", obs_dim), ("act", act_dim), ("rew", 1),
                                ("next_obs", obs_dim), ("policy_noise", act_dim), ("next_noise", act_dim))}
    output["act"] = np.tanh(output["act"])
    output["done"] = rng.integers(0, 2, size=(steps, batch_size, 1)).astype(np.float32)
    records = {}
    for arm, weight in (("full", 0.01), ("no_shift", 0.0)):
        torch.manual_seed(2027)
        critic = VectorizedQNetwork(obs_dim, act_dim, 8, 3)
        policy = TanhGaussianPolicy([8, 8], obs_dim, act_dim)
        trainer = RESACTrainer(SimpleNamespace(action_space=SimpleNamespace(shape=(act_dim,))),
            policy, critic, deepcopy(critic), ensemble_size=3, weight_reg=weight, clip_norm=1.0)
        snapshot(output, arm + "/initial", trainer, False)
        records[arm] = []
        for step in range(steps):
            batch = {target: torch.from_numpy(output[source][step]) for source, target in
                     (("obs", "observations"), ("act", "actions"), ("rew", "rewards"),
                      ("next_obs", "next_observations"), ("done", "terminals"))}
            draws = [torch.from_numpy(output[key][step]) for key in ("policy_noise", "next_noise")]
            with patch("torch.distributions.Normal.sample", side_effect=draws) as sampler:
                trainer.train_from_torch(batch)
                assert sampler.call_count == 2
            snapshot(output, f"{arm}/step{step}", trainer, True)
            records[arm].append({key: float(value) for key, value in trainer.get_diagnostics().items()
                                 if np.isscalar(value)})
            trainer.end_epoch(step)
    output["metadata"] = np.asarray(json.dumps({"torch": torch.__version__, "steps": steps,
        "records": records, "source": str(MINE / "RE-SAC/experiment/resac_trainer.py"),
        "precision": "FP32; CUDA/AMP disabled; same tensors and Gaussian noise"}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **output)
    print(f"Original Torch trace: {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)
