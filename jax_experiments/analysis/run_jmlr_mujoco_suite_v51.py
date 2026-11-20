#!/usr/bin/env python3
"""Standard Gymnasium MuJoCo v5 multi-morphology suite at strict 500k total-step parity (SAME_STEP autoreset).

Training and evaluation protocol:
  1. Standard Gymnasium 1.2.0 / MuJoCo 3.2.7 rigid-body contact dynamics across 4 morphologies:
     HalfCheetah-v5 (d=6), Ant-v5 (d=8), Walker2d-v5 (d=6), and Hopper-v5 (d=3),
     using explicit AutoresetMode.SAME_STEP so every vectorized step executes real physics transitions
     and extracts terminal observations strictly from infos["final_obs"].
  2. Each controller collects 500,000 physics transitions:
     - bapr: Canonical SAC trained strictly in fixed reference mode 0 (500k steps, with 5%
       4-step uniform exploratory bursts at rate 0.0125), fitting both its 5-head inverse-dynamics
       ensemble and a structure-matched 5-head shared-physics forward-dynamics ensemble directly on
       its own 500k mode-0 replay buffer (4,500 AdamW updates, 0 external steps, 0 mode labels).
     - bapr_noburst: Controlled ablation identical to bapr with 0% exploratory bursts (pure natural
       Gaussian SAC exploration in mode 0), directly testing burst effects on canonical headroom and
       self-calibrated identification.
     - sac: Memoryless multimode SAC (500k steps, [256,256] actor, [256,256,256] twin critics).
     - escp256: Task-adapted ESCP ([256,256] actor, [256,256,256] twin critics, FC(256)->GRU(128)->8D
       recurrent history encoder) with an RMDM-style objective (EMA task
       prototypes, within-regime consistency loss, and cross-regime RBF kernel log-determinant diversity).
     - rma: Rapid Motor Adaptation (Kumar et al., 2021) — 400k steps Phase-1 privileged extrinsics
       base policy ([256,256] actor, [256,256,256] critics) + 100k steps Phase-2 online history
       adaptation module = 500k total environment steps.
     - belief_sac: Belief-conditioned SAC with 500k multimode steps plus access to the frozen inverse
       ensemble fitted on the paired BAPR controller's replay.
  3. Complete checkpoint + replay buffer + RNG state persistence for exact preemption recovery.
  4. Zero-shot scaling to M = 2^6 = 64 (HalfCheetah) and M = 2^8 = 256 (Ant) polarity hypercubes
     via O(d) channel-factored tempered filtering, plus non-bijective 0.5x lossy gain projection,
     with full-horizon zero-padded 32-step switch regret and explicit switch-coverage accounting.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - allows offline JSON aggregation without simulator installed
    gym = None
import numpy as np
from scipy.stats import t as student_t
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    F = None
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "jmlr-mujoco-v5-multimorphology-v51-samestep"
CHECKPOINT_NAME = "training_checkpoint.npz"
SEEDS = (51003, 51021, 51039, 51057, 51079)
ARMS = ("bapr", "bapr_noburst", "sac", "escp256", "rma", "belief_sac")
ENV_SPECS = {
    "halfcheetah": ("HalfCheetah-v5", {}),
    "ant": ("Ant-v5", {"include_cfrc_ext_in_observation": False}),
    "walker2d": ("Walker2d-v5", {}),
    "hopper": ("Hopper-v5", {}),
}
ENV_NAMES = tuple(ENV_SPECS.keys())

AUTORESET_MODE = "SAME_STEP"
TOTAL_STEPS = 500_000
NUM_ENVS = 8
VEC_ITERS = TOTAL_STEPS // NUM_ENVS  # 62,500 vectorized steps -> exact 500,000 physics transitions
WARMUP_STEPS = 4_800
BATCH_SIZE = 256
UPDATES_PER_VEC_STEP = 2             # 125k gradient updates over 500k env steps (UTD = 0.25)
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
HISTORY_LEN = 16
NOISE_STD = 0.02
DWELL_STEPS = 250
EPISODE_HORIZON = 1000
BURST_LEN = 4
BURST_PROB = 0.0125                  # 4 * 0.0125 = 0.05 (~5% exploratory steps in mode 0)
INVERSE_UPDATES = 4500

# Released Luo et al. (2022) ESCP RMDM hyperparameters (matching jax_experiments/algos/escp.py)
ESCP_RBF_RADIUS = 80.0
ESCP_CONS_WEIGHT = 50.0
ESCP_DIV_WEIGHT = 0.025
ESCP_PROTO_TAU = 0.995
ESCP_BOTTLENECK_SIGMA = 0.01
ESCP_CONTEXT_WARMUP_STEPS = 25_000

SWITCH_EVENT_SEEDS = (310101, 310117, 310133)
SWITCH_BASE_SCHEDULES = {
    310101: (2, 0, 3, 1),
    310117: (1, 3, 0, 2),
    310133: (3, 2, 1, 0),
}
STATIONARY_EVENT_SEED = 310001

RUN_ROOT = ROOT / "jax_experiments/results_jmlr_mujoco_v51_samestep"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_jmlr_mujoco_v51_samestep"
RELEASE_DIR = ROOT / "reproducibility/jmlr_mujoco_v51"

EXPECTED_RUNTIME = {
    "torch_prefix": "2.5.1",
    "gymnasium": "1.2.0",
    "mujoco": "3.2.7",
}


def runtime_info(strict: bool = True) -> dict[str, str]:
    info = {
        "torch": torch.__version__,
        "gymnasium": version("gymnasium"),
        "mujoco": version("mujoco"),
        "numpy": version("numpy"),
    }
    if strict:
        if not info["torch"].startswith(EXPECTED_RUNTIME["torch_prefix"]):
            raise ValueError(f"Expected torch 2.5.1*, got {info['torch']}")
        if info["gymnasium"] != EXPECTED_RUNTIME["gymnasium"]:
            raise ValueError(f"Expected gymnasium {EXPECTED_RUNTIME['gymnasium']}, got {info['gymnasium']}")
        if info["mujoco"] != EXPECTED_RUNTIME["mujoco"]:
            raise ValueError(f"Expected mujoco {EXPECTED_RUNTIME['mujoco']}, got {info['mujoco']}")
    return info


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def emit_progress(current: int, total: int, started: float, unit: str = "step") -> None:
    print("SCHEDULEURM_PROGRESS " + json.dumps({
        "current": current,
        "total": total,
        "unit": unit,
        "elapsed_seconds": time.monotonic() - started,
    }), flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Actuator Regime Definitions (General d = 3, 6, 8)
# ---------------------------------------------------------------------------

def canonical_polarity_gains(act_dim: int) -> np.ndarray:
    """Four balanced actuator-polarity regimes for arbitrary action dimension d."""
    half = act_dim // 2
    g0 = np.ones(act_dim, dtype=np.float32)
    g0[:half] = -1.0  # matches HalfCheetah reference (-1,-1,-1,+1,+1,+1) when d=6
    g1 = -g0
    g2 = np.where(np.arange(act_dim) % 2 == 0, -1.0, 1.0).astype(np.float32)
    g3 = -g2
    return np.stack([g0, g1, g2, g3], axis=0)


def lossy_degradation_gains(act_dim: int) -> np.ndarray:
    """Non-bijective 50% motor power loss combined with polarity inversions."""
    p = canonical_polarity_gains(act_dim)
    half = act_dim // 2
    g0 = p[0].copy()  # unattenuated reference
    g1 = p[1].copy()
    g1[:half] *= 0.5  # lower-half 50% torque loss + upper-half sign flip
    g2 = p[2].copy()
    g2[half:] *= 0.5  # upper-half 50% torque loss + even sign flip
    g3 = p[3].copy()
    g3[::2] *= 0.5    # even-channel 50% torque loss + odd sign flip
    return np.stack([g0, g1, g2, g3], axis=0)


def hypercube_polarity_gains(act_dim: int) -> np.ndarray:
    """All 2^d sign vectors in {-1, +1}^d with mode 0 matching canonical_polarity_gains[0]."""
    ref = tuple(float(x) for x in canonical_polarity_gains(act_dim)[0])
    all_vecs = [tuple(float(x) for x in row) for row in itertools.product((-1.0, 1.0), repeat=act_dim)]
    all_vecs.remove(ref)
    return np.asarray([ref, *all_vecs], dtype=np.float32)


def make_single_env(env_key: str):
    env_id, kwargs = ENV_SPECS[env_key]
    return gym.make(env_id, **kwargs)


def make_vec_env(env_key: str, num_envs: int, seed: int):
    """Create SyncVectorEnv with explicit SAME_STEP autoreset so every step executes physics."""
    env_id, kwargs = ENV_SPECS[env_key]
    venv = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_id, **kwargs) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    venv.reset(seed=seed)
    return venv


def snapshot_vector_env(venv):
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    states = []
    for env in venv.envs:
        base = env.unwrapped
        physics = np.empty(mujoco.mj_stateSize(base.model, spec), dtype=np.float64)
        mujoco.mj_getState(base.model, base.data, physics, spec)
        states.append((physics, env._elapsed_steps, base.np_random.bit_generator.state))
    return states


def restore_vector_env(venv, states):
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    for env, (physics, elapsed_steps, rng_state) in zip(venv.envs, states, strict=True):
        base = env.unwrapped
        mujoco.mj_setState(base.model, base.data, physics, spec)
        mujoco.mj_forward(base.model, base.data)
        # Restore integration inputs such as solver warm-start after recomputing derived fields.
        mujoco.mj_setState(base.model, base.data, physics, spec)
        env._elapsed_steps = elapsed_steps
        base.np_random.bit_generator.state = rng_state


# ---------------------------------------------------------------------------
# Neural Network Modules (Matched [256, 256] Actor / [256, 256, 256] Critics)
# ---------------------------------------------------------------------------

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class SquashedGaussianActor(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden: int = 256, leaky: bool = False):
        super().__init__()
        act_cls = (lambda: nn.LeakyReLU(0.2)) if leaky else nn.ReLU
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act_cls(),
            nn.Linear(hidden, hidden),
            act_cls(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)
        log_prob = (
            -0.5 * (eps.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class TwinQCritic(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden: int = 256, leaky: bool = False):
        super().__init__()
        act_cls = (lambda: nn.LeakyReLU(0.2)) if leaky else nn.ReLU
        self.q1 = nn.Sequential(
            nn.Linear(in_dim + act_dim, hidden), act_cls(),
            nn.Linear(hidden, hidden), act_cls(),
            nn.Linear(hidden, hidden), act_cls(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim + act_dim, hidden), act_cls(),
            nn.Linear(hidden, hidden), act_cls(),
            nn.Linear(hidden, hidden), act_cls(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class HistoryContextEncoder(nn.Module):
    """Recurrent history encoder (used by capacity-matched ESCP-256 and RMA Phase 2)."""
    def __init__(self, obs_dim: int, act_dim: int, context_dim: int = 8, fc_dim: int = 256, gru_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + act_dim, fc_dim),
            nn.LeakyReLU(0.2),
        )
        self.gru = nn.GRU(fc_dim, gru_dim, batch_first=True)
        self.out = nn.Linear(gru_dim, context_dim)

    def forward(self, hist_obs_act: torch.Tensor) -> torch.Tensor:
        # hist_obs_act: [B, H, obs_dim + act_dim]
        B, H, D = hist_obs_act.shape
        x = self.fc(hist_obs_act.reshape(B * H, D)).reshape(B, H, -1)
        _, h_n = self.gru(x)
        return torch.tanh(self.out(h_n[0]))


@torch.no_grad()
def update_escp_prototypes(
    prototypes: torch.Tensor,
    proto_valid: torch.Tensor,
    contexts: torch.Tensor,
    task_ids: torch.Tensor,
    tau: float = ESCP_PROTO_TAU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update EMA per-regime prototypes matching jax_experiments/algos/escp.py:update_prototypes."""
    new_proto = prototypes.clone()
    new_valid = proto_valid.clone()
    n_modes = prototypes.shape[0]
    for m in range(n_modes):
        mask = task_ids == m
        if torch.any(mask):
            mean_m = contexts[mask].mean(dim=0)
            if bool(proto_valid[m].item()):
                new_proto[m] = tau * prototypes[m] + (1.0 - tau) * mean_m
            else:
                new_proto[m] = mean_m
            new_valid[m] = True
    return new_proto, new_valid


def compute_escp_rmdm_loss(
    contexts: torch.Tensor,
    task_ids: torch.Tensor,
    prototypes: torch.Tensor,
    proto_valid: torch.Tensor,
    rbf_radius: float = ESCP_RBF_RADIUS,
    cons_weight: float = ESCP_CONS_WEIGHT,
    div_weight: float = ESCP_DIV_WEIGHT,
) -> torch.Tensor:
    """RMDM-style objective adapted from the repository's recurrent ESCP implementation.

    Combines within-regime embedding consistency against EMA task prototypes with
    cross-regime Relational Matrix Determinant Maximization (-log det K) over element-wise RBF kernels.
    """
    n_modes, ctx_dim = prototypes.shape
    device = contexts.device
    means = []
    seen_flags = []
    ref_list = []
    for m in range(n_modes):
        mask = task_ids == m
        has_m = bool(torch.any(mask).item())
        seen_flags.append(has_m)
        if has_m:
            mean_m = contexts[mask].mean(dim=0)
        else:
            mean_m = prototypes[m].detach()
        means.append(mean_m)
        if bool(proto_valid[m].item()):
            ref_list.append(prototypes[m].detach())
        else:
            ref_list.append(mean_m.detach())

    ref_stack = torch.stack(ref_list, dim=0)  # [M, D]
    target_refs = ref_stack[task_ids]         # [B, D]
    sq_dev = (contexts - target_refs).pow(2).sum() / max(float(contexts.numel()), 1.0)
    consistency = torch.sqrt(torch.clamp(sq_dev, min=1e-12))

    seen_t = torch.tensor(seen_flags, dtype=torch.bool, device=device)
    valid_t = proto_valid | seen_t
    n_valid = int(valid_t.sum().item())
    if n_valid < 2:
        return cons_weight * consistency

    reps = torch.stack([
        means[m] if seen_flags[m] else prototypes[m].detach()
        for m in range(n_modes)
    ], dim=0)  # [M, D]
    diffs = reps.unsqueeze(1) - reps.unsqueeze(0)  # [M, M, D]
    kernel = torch.exp(-rbf_radius * diffs.pow(2)).mean(dim=-1)  # [M, M]

    valid_pair = valid_t.unsqueeze(1) & valid_t.unsqueeze(0)
    matrix = torch.where(valid_pair, kernel, torch.zeros_like(kernel))
    diag_vals = torch.where(
        valid_t,
        torch.full((n_modes,), 1.001, device=device, dtype=contexts.dtype),
        torch.ones((n_modes,), device=device, dtype=contexts.dtype),
    )
    matrix = matrix - torch.diag(torch.diag(matrix)) + torch.diag(diag_vals)
    chol = torch.linalg.cholesky(matrix)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    diversity = -logdet

    if float(consistency.detach().item()) >= 0.1:
        return cons_weight * consistency
    return cons_weight * consistency + div_weight * diversity


class ExtrinsicsEncoder(nn.Module):
    """RMA Phase-1 privileged gain-vector encoder E(g_z) -> R^8."""
    def __init__(self, act_dim: int, context_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim, 64),
            nn.SiLU(),
            nn.Linear(64, context_dim),
            nn.Tanh(),
        )

    def forward(self, gain_vec: torch.Tensor) -> torch.Tensor:
        return self.net(gain_vec)


class VectorizedInverseEnsemble(nn.Module):
    """Five-head transition-only inverse dynamics ensemble matching ExecutedActionInverse."""
    def __init__(self, obs_dim: int, act_dim: int, heads: int = 5, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.heads = heads
        in_dim = 3 * obs_dim
        self.w1 = nn.Parameter(torch.empty(heads, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w2 = nn.Parameter(torch.empty(heads, hidden, hidden))
        self.b2 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w3 = nn.Parameter(torch.empty(heads, hidden, hidden))
        self.b3 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w4 = nn.Parameter(torch.empty(heads, hidden, act_dim))
        self.b4 = nn.Parameter(torch.zeros(heads, 1, act_dim))
        for w in (self.w1, self.w2, self.w3, self.w4):
            for k in range(heads):
                nn.init.kaiming_uniform_(w.data[k], a=math.sqrt(5))

    def features(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            torch.tanh(obs / 5.0),
            torch.tanh(next_obs / 5.0),
            torch.tanh((next_obs - obs) / 0.1),
        ], dim=-1)

    def forward_shared(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        x = self.features(obs, next_obs).unsqueeze(0).expand(self.heads, -1, -1)
        return self._forward_x(x)

    def forward_head_batches(self, obs_kb: torch.Tensor, next_obs_kb: torch.Tensor) -> torch.Tensor:
        x = self.features(obs_kb, next_obs_kb)
        return self._forward_x(x)

    def _forward_x(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(torch.bmm(x, self.w1) + self.b1)
        h = F.silu(torch.bmm(h, self.w2) + self.b2)
        h = F.silu(torch.bmm(h, self.w3) + self.b3)
        return torch.tanh(torch.bmm(h, self.w4) + self.b4)


class VectorizedForwardEnsemble(nn.Module):
    """Structure-matched shared-physics 5-head forward ensemble g_psi(s_t, v_t) -> tanh((s_{t+1}-s_t)/0.1).

    Trained strictly on the exact same reference-mode replay split and 4,500 AdamW updates as
    VectorizedInverseEnsemble. At deployment, evaluates candidate nominal actuations v_t(z) = clip(G_z a_t, -1, 1)
    through the single canonical physics model.
    """
    def __init__(self, obs_dim: int, act_dim: int, heads: int = 5, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.heads = heads
        in_dim = obs_dim + act_dim
        self.w1 = nn.Parameter(torch.empty(heads, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w2 = nn.Parameter(torch.empty(heads, hidden, hidden))
        self.b2 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w3 = nn.Parameter(torch.empty(heads, hidden, hidden))
        self.b3 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w4 = nn.Parameter(torch.empty(heads, hidden, obs_dim))
        self.b4 = nn.Parameter(torch.zeros(heads, 1, obs_dim))
        for w in (self.w1, self.w2, self.w3, self.w4):
            for k in range(heads):
                nn.init.kaiming_uniform_(w.data[k], a=math.sqrt(5))

    @staticmethod
    def target_delta(obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh((next_obs - obs) / 0.1)

    def features(self, obs: torch.Tensor, nom_act: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.tanh(obs / 5.0), torch.clamp(nom_act, -1.0, 1.0)], dim=-1)

    def forward_head_batches(self, obs_kb: torch.Tensor, act_kb: torch.Tensor) -> torch.Tensor:
        x = self.features(obs_kb, act_kb)
        return self._forward_x(x)

    def forward_candidates(self, obs: torch.Tensor, cand_act: torch.Tensor) -> torch.Tensor:
        # obs: [B, obs_dim], cand_act: [B, M, act_dim] -> [K, B, M, obs_dim]
        B, M, _ = cand_act.shape
        obs_exp = obs.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
        act_flat = cand_act.reshape(B * M, -1)
        x = self.features(obs_exp, act_flat).unsqueeze(0).expand(self.heads, -1, -1)
        out = self._forward_x(x)  # [K, B*M, obs_dim]
        return out.view(self.heads, B, M, self.obs_dim)

    def _forward_x(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(torch.bmm(x, self.w1) + self.b1)
        h = F.silu(torch.bmm(h, self.w2) + self.b2)
        h = F.silu(torch.bmm(h, self.w3) + self.b3)
        return torch.tanh(torch.bmm(h, self.w4) + self.b4)


@torch.no_grad()
def joint_emission_scores(
    inv_model: VectorizedInverseEnsemble,
    variance: torch.Tensor,
    gains: torch.Tensor,
    obs: torch.Tensor,
    command: torch.Tensor,
    next_obs: torch.Tensor,
) -> torch.Tensor:
    """Compute log-mixture emission scores [B, M] for candidate diagonal gains [M, d]."""
    pred = inv_model.forward_shared(obs, next_obs)  # [K, B, d]
    cand = torch.clamp(command.unsqueeze(1) * gains.unsqueeze(0), -1.0, 1.0)  # [B, M, d]
    var = torch.clamp(variance, 1e-6, 1.0).view(1, 1, 1, -1)
    err = cand.unsqueeze(0) - pred.unsqueeze(2)  # [K, B, M, d]
    head_ll = -0.5 * torch.mean(err.pow(2) / var + torch.log(2.0 * math.pi * var), dim=-1)  # [K, B, M]
    return torch.logsumexp(head_ll, dim=0) - math.log(float(pred.shape[0]))  # [B, M]


@torch.no_grad()
def shared_forward_emission_scores(
    fwd_model: VectorizedForwardEnsemble,
    fwd_variance: torch.Tensor,
    gains: torch.Tensor,
    obs: torch.Tensor,
    command: torch.Tensor,
    next_obs: torch.Tensor,
) -> torch.Tensor:
    """Compute log-mixture emission scores [B, M] from shared-physics forward model g_psi(s_t, clip(G_z a_t, -1, 1))."""
    cand = torch.clamp(command.unsqueeze(1) * gains.unsqueeze(0), -1.0, 1.0)  # [B, M, d_a]
    pred_delta = fwd_model.forward_candidates(obs, cand)  # [K, B, M, d_s]
    obs_delta = VectorizedForwardEnsemble.target_delta(obs, next_obs).view(1, -1, 1, obs.shape[-1])
    var = torch.clamp(fwd_variance, 1e-6, 1.0).view(1, 1, 1, -1)
    err = obs_delta - pred_delta  # [K, B, M, d_s]
    head_ll = -0.5 * torch.mean(err.pow(2) / var + torch.log(2.0 * math.pi * var), dim=-1)  # [K, B, M]
    return torch.logsumexp(head_ll, dim=0) - math.log(float(pred_delta.shape[0]))  # [B, M]


@torch.no_grad()
def update_sticky_belief(
    belief: torch.Tensor,
    scores: torch.Tensor,
    hazard: float = 0.002,
    scale: float = 1.0,
    decay: float = 0.98,
    temporal: bool = True,
) -> torch.Tensor:
    if temporal:
        n_modes = belief.shape[-1]
        prior = (1.0 - hazard) * belief + (hazard / float(n_modes - 1)) * (1.0 - belief) if hazard > 0 else belief
        logits = decay * torch.log(torch.clamp(prior, 1e-30, 1.0)) + scale * scores
    else:
        logits = scale * scores
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def update_factored_channel_belief(
    prob_pos: torch.Tensor,
    inv_model: VectorizedInverseEnsemble,
    variance: torch.Tensor,
    obs: torch.Tensor,
    command: torch.Tensor,
    next_obs: torch.Tensor,
    hazard: float = 0.002,
    scale: float = 0.5,
    decay: float = 0.98,
) -> torch.Tensor:
    """O(d) independent binary channel belief update over {-1, +1}^d."""
    pred = inv_model.forward_shared(obs, next_obs)  # [K, B, d]
    var = torch.clamp(variance, 1e-6, 1.0).view(1, 1, -1)
    c_pos = torch.clamp(command, -1.0, 1.0).unsqueeze(0)
    c_neg = torch.clamp(-command, -1.0, 1.0).unsqueeze(0)
    ll_pos = torch.logsumexp(-0.5 * (c_pos - pred).pow(2) / var, dim=0) - math.log(float(pred.shape[0]))
    ll_neg = torch.logsumexp(-0.5 * (c_neg - pred).pow(2) / var, dim=0) - math.log(float(pred.shape[0]))
    prior_pos = (1.0 - hazard) * prob_pos + hazard * (1.0 - prob_pos)
    log_odds = decay * (
        torch.log(torch.clamp(prior_pos, 1e-30, 1.0)) - torch.log(torch.clamp(1.0 - prior_pos, 1e-30, 1.0))
    ) + scale * (ll_pos - ll_neg)
    return torch.sigmoid(log_odds)


# ---------------------------------------------------------------------------
# Fast Preallocated Replay Buffers with Exact State Serialization
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, extra_dim: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.mode = np.zeros((capacity,), dtype=np.int64)
        self.extra = np.zeros((capacity, extra_dim), dtype=np.float32) if extra_dim > 0 else None
        self.next_extra = np.zeros((capacity, extra_dim), dtype=np.float32) if extra_dim > 0 else None
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, act, rew, next_obs, done, mode, extra=None, next_extra=None):
        n = len(obs)
        idxs = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idxs] = obs
        self.act[idxs] = act
        self.rew[idxs, 0] = rew
        self.next_obs[idxs] = next_obs
        self.done[idxs, 0] = done
        self.mode[idxs] = mode
        if self.extra is not None and extra is not None:
            self.extra[idxs] = extra
            self.next_extra[idxs] = next_extra
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator | None = None):
        idxs = (rng.integers(0, self.size, size=batch_size) if rng is not None
                else np.random.randint(0, self.size, size=batch_size))
        out = (
            torch.as_tensor(self.obs[idxs], device=device),
            torch.as_tensor(self.act[idxs], device=device),
            torch.as_tensor(self.rew[idxs], device=device),
            torch.as_tensor(self.next_obs[idxs], device=device),
            torch.as_tensor(self.done[idxs], device=device),
            torch.as_tensor(self.mode[idxs], device=device),
        )
        if self.extra is not None:
            return (*out, torch.as_tensor(self.extra[idxs], device=device), torch.as_tensor(self.next_extra[idxs], device=device))
        return out

    def save_npz(self, path: Path, learner_state: np.ndarray) -> None:
        n = self.size
        arrays = {
            "learner_state": learner_state,
            "ptr": np.asarray(self.ptr, dtype=np.int64),
            "size": np.asarray(self.size, dtype=np.int64),
            "obs": self.obs[:n],
            "next_obs": self.next_obs[:n],
            "act": self.act[:n],
            "rew": self.rew[:n],
            "done": self.done[:n],
            "mode": self.mode[:n],
        }
        if self.extra is not None:
            arrays["extra"] = self.extra[:n]
            arrays["next_extra"] = self.next_extra[:n]
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    def load_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            n = int(data["size"])
            self.ptr = int(data["ptr"])
            self.size = n
            self.obs[:n] = data["obs"]
            self.next_obs[:n] = data["next_obs"]
            self.act[:n] = data["act"]
            self.rew[:n] = data["rew"]
            self.done[:n] = data["done"]
            self.mode[:n] = data["mode"]
            if self.extra is not None and "extra" in data:
                self.extra[:n] = data["extra"]
                self.next_extra[:n] = data["next_extra"]


class SequenceReplayBuffer:
    """Stores transitions plus 16-step (obs, act) histories for ESCP-256."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, hist_len: int = HISTORY_LEN):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.mode = np.zeros((capacity,), dtype=np.int64)
        self.hist = np.zeros((capacity, hist_len, obs_dim + act_dim), dtype=np.float16)
        self.next_hist = np.zeros((capacity, hist_len, obs_dim + act_dim), dtype=np.float16)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, act, rew, next_obs, done, mode, hist, next_hist):
        n = len(obs)
        idxs = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idxs] = obs
        self.act[idxs] = act
        self.rew[idxs, 0] = rew
        self.next_obs[idxs] = next_obs
        self.done[idxs, 0] = done
        self.mode[idxs] = mode
        self.hist[idxs] = hist.astype(np.float16)
        self.next_hist[idxs] = next_hist.astype(np.float16)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator | None = None):
        idxs = (rng.integers(0, self.size, size=batch_size) if rng is not None
                else np.random.randint(0, self.size, size=batch_size))
        return (
            torch.as_tensor(self.obs[idxs], device=device),
            torch.as_tensor(self.act[idxs], device=device),
            torch.as_tensor(self.rew[idxs], device=device),
            torch.as_tensor(self.next_obs[idxs], device=device),
            torch.as_tensor(self.done[idxs], device=device),
            torch.as_tensor(self.mode[idxs], device=device),
            torch.as_tensor(self.hist[idxs].astype(np.float32), device=device),
            torch.as_tensor(self.next_hist[idxs].astype(np.float32), device=device),
        )

    def save_npz(self, path: Path, learner_state: np.ndarray) -> None:
        n = self.size
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            learner_state=learner_state,
            ptr=np.asarray(self.ptr, dtype=np.int64),
            size=np.asarray(self.size, dtype=np.int64),
            obs=self.obs[:n],
            next_obs=self.next_obs[:n],
            act=self.act[:n],
            rew=self.rew[:n],
            done=self.done[:n],
            mode=self.mode[:n],
            hist=self.hist[:n],
            next_hist=self.next_hist[:n],
        )
        os.replace(tmp, path)

    def load_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            n = int(data["size"])
            self.ptr = int(data["ptr"])
            self.size = n
            self.obs[:n] = data["obs"]
            self.next_obs[:n] = data["next_obs"]
            self.act[:n] = data["act"]
            self.rew[:n] = data["rew"]
            self.done[:n] = data["done"]
            self.mode[:n] = data["mode"]
            self.hist[:n] = data["hist"]
            self.next_hist[:n] = data["next_hist"]


# ---------------------------------------------------------------------------
# Training Loops for All 6 Arms (Strict 500k Real Physics Transitions)
# ---------------------------------------------------------------------------

def bundle_dir(env_key: str, arm: str, seed: int) -> Path:
    return BUNDLE_ROOT / env_key / arm / f"seed_{seed}"


def run_dir(env_key: str, arm: str, seed: int) -> Path:
    return RUN_ROOT / env_key / arm / f"seed_{seed}"


def fit_self_calibrated_inverse(
    replay: ReplayBuffer,
    obs_dim: int,
    act_dim: int,
    ref_gain_np: np.ndarray,
    seed: int,
    device: torch.device,
    updates: int = INVERSE_UPDATES,
) -> tuple[VectorizedInverseEnsemble, np.ndarray]:
    """Fit 5-head inverse ensemble purely from reference-mode replay buffer (v_t = G_0 a_t)."""
    n_total = replay.size
    perm = np.random.default_rng(seed + 901).permutation(n_total)
    n_val = max(8, n_total // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    obs_all = replay.obs[:n_total]
    next_obs_all = replay.next_obs[:n_total]
    targets_all = np.clip(replay.act[:n_total] * ref_gain_np[None, :], -1.0, 1.0).astype(np.float32)

    inv = VectorizedInverseEnsemble(obs_dim, act_dim, heads=5, hidden=256).to(device)
    opt = torch.optim.AdamW(inv.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 902)

    for _ in range(updates):
        head_idxs = rng.integers(0, len(tr_idx), size=(5, BATCH_SIZE))
        flat = tr_idx[head_idxs]
        b_obs = torch.as_tensor(obs_all[flat], device=device)
        b_nobs = torch.as_tensor(next_obs_all[flat], device=device)
        b_tgt = torch.as_tensor(targets_all[flat], device=device)
        pred = inv.forward_head_batches(b_obs, b_nobs)
        loss = torch.mean((pred - b_tgt).pow(2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(inv.parameters(), 10.0)
        opt.step()

    inv.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(val_idx), 2048):
            sl = val_idx[start:start + 2048]
            vo = torch.as_tensor(obs_all[sl], device=device)
            vno = torch.as_tensor(next_obs_all[sl], device=device)
            preds.append(inv.forward_shared(vo, vno).mean(dim=0).cpu().numpy())
    mean_pred = np.concatenate(preds, axis=0)
    variance = np.clip(np.mean((mean_pred - targets_all[val_idx]) ** 2, axis=0), 1e-4, 0.25).astype(np.float32)
    return inv, variance


def fit_self_calibrated_shared_forward(
    replay: ReplayBuffer,
    obs_dim: int,
    act_dim: int,
    ref_gain_np: np.ndarray,
    seed: int,
    device: torch.device,
    updates: int = INVERSE_UPDATES,
) -> tuple[VectorizedForwardEnsemble, np.ndarray]:
    """Fit structure-matched 5-head shared-physics forward ensemble on the exact same mode-0 split."""
    n_total = replay.size
    perm = np.random.default_rng(seed + 901).permutation(n_total)
    n_val = max(8, n_total // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    obs_all = replay.obs[:n_total]
    next_obs_all = replay.next_obs[:n_total]
    nom_act_all = np.clip(replay.act[:n_total] * ref_gain_np[None, :], -1.0, 1.0).astype(np.float32)
    delta_all = np.tanh((next_obs_all - obs_all) / 0.1).astype(np.float32)

    fwd = VectorizedForwardEnsemble(obs_dim, act_dim, heads=5, hidden=256).to(device)
    opt = torch.optim.AdamW(fwd.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 903)

    for _ in range(updates):
        head_idxs = rng.integers(0, len(tr_idx), size=(5, BATCH_SIZE))
        flat = tr_idx[head_idxs]
        b_obs = torch.as_tensor(obs_all[flat], device=device)
        b_act = torch.as_tensor(nom_act_all[flat], device=device)
        b_tgt = torch.as_tensor(delta_all[flat], device=device)
        pred = fwd.forward_head_batches(b_obs, b_act)
        loss = torch.mean((pred - b_tgt).pow(2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(fwd.parameters(), 10.0)
        opt.step()

    fwd.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(val_idx), 2048):
            sl = val_idx[start:start + 2048]
            vo = torch.as_tensor(obs_all[sl], device=device)
            va = torch.as_tensor(nom_act_all[sl], device=device).unsqueeze(1)  # [B, 1, act_dim]
            preds.append(fwd.forward_candidates(vo, va)[:, :, 0, :].mean(dim=0).cpu().numpy())
    mean_pred = np.concatenate(preds, axis=0)
    variance = np.clip(np.mean((mean_pred - delta_all[val_idx]) ** 2, axis=0), 1e-4, 0.25).astype(np.float32)
    return fwd, variance


def soft_update(target: nn.Module, source: nn.Module, tau: float = TAU) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def train_arm(
    env_key: str,
    arm: str,
    seed: int,
    smoke: bool = False,
    stop_after_iters: int | None = None,
) -> dict:
    runtime = runtime_info(strict=not smoke)
    dest = bundle_dir(env_key, arm, seed)
    if not smoke and stop_after_iters is None and (dest / "bundle.json").is_file():
        existing = json.loads((dest / "bundle.json").read_text())
        if existing.get("protocol") == PROTOCOL:
            print(f"Already completed: {dest / 'bundle.json'}", flush=True)
            return existing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(seed)
    started = time.monotonic()

    num_envs = 4 if smoke else NUM_ENVS
    total_steps = 512 if smoke else TOTAL_STEPS
    warmup_steps = 256 if smoke else WARMUP_STEPS
    vec_iters = total_steps // num_envs
    inv_updates = 4 if smoke else INVERSE_UPDATES
    is_single_mode_bapr = arm in ("bapr", "bapr_noburst")
    use_bursts = (arm == "bapr")

    venv = make_vec_env(env_key, num_envs, seed)
    obs_dim = int(venv.single_observation_space.shape[0])
    act_dim = int(venv.single_action_space.shape[0])
    gains_np = canonical_polarity_gains(act_dim)
    gains_t = torch.as_tensor(gains_np, device=device)
    n_modes = gains_np.shape[0]
    target_entropy = -float(act_dim)

    ckpt_dir = run_dir(env_key, arm, seed) / "checkpoints"
    ckpt_path = ckpt_dir / CHECKPOINT_NAME
    checkpoint_identity = (PROTOCOL, env_key, arm, seed, num_envs, total_steps, device.type, runtime)

    frozen_inv = None
    frozen_var = None
    if arm == "belief_sac":
        bapr_dest = bundle_dir(env_key, "bapr", seed)
        inv_ckpt = torch.load(bapr_dest / "inverse.pt", map_location=device, weights_only=False)
        frozen_inv = VectorizedInverseEnsemble(obs_dim, act_dim, heads=5, hidden=256).to(device)
        frozen_inv.load_state_dict(inv_ckpt["state_dict"])
        frozen_inv.eval()
        frozen_var = torch.as_tensor(inv_ckpt["residual_variance"], dtype=torch.float32, device=device)

    in_dim = (
        obs_dim + 4 if arm == "belief_sac"
        else obs_dim + 8 if arm in ("escp256", "rma")
        else obs_dim
    )
    leaky = (arm == "escp256")
    actor = SquashedGaussianActor(in_dim, act_dim, hidden=256, leaky=leaky).to(device)
    critic = TwinQCritic(in_dim, act_dim, hidden=256, leaky=leaky).to(device)
    target_critic = TwinQCritic(in_dim, act_dim, hidden=256, leaky=leaky).to(device)
    target_critic.load_state_dict(critic.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    ext_enc = ExtrinsicsEncoder(act_dim, 8).to(device) if arm == "rma" else None
    hist_enc = HistoryContextEncoder(obs_dim, act_dim, 8, 256, 128).to(device) if arm in ("escp256", "rma") else None
    escp_prototypes = torch.zeros((n_modes, 8), dtype=torch.float32, device=device)
    escp_proto_valid = torch.zeros((n_modes,), dtype=torch.bool, device=device)

    actor_params = list(actor.parameters()) + (list(ext_enc.parameters()) if ext_enc is not None else [])
    critic_params = list(critic.parameters()) + (list(ext_enc.parameters()) if ext_enc is not None else [])
    actor_opt = torch.optim.Adam(actor_params, lr=LR)
    critic_opt = torch.optim.Adam(critic_params, lr=LR)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR)
    enc_opt = torch.optim.Adam(hist_enc.parameters(), lr=LR) if hist_enc is not None else None

    if arm == "escp256":
        replay = SequenceReplayBuffer(total_steps, obs_dim, act_dim, HISTORY_LEN)
    elif arm == "belief_sac":
        replay = ReplayBuffer(total_steps, obs_dim, act_dim, extra_dim=4)
    else:
        replay = ReplayBuffer(total_steps, obs_dim, act_dim)

    obs, _ = venv.reset(seed=seed)
    obs = obs.astype(np.float32)
    env_modes = np.zeros(num_envs, dtype=np.int64) if is_single_mode_bapr else np.arange(num_envs, dtype=np.int64) % n_modes
    dwell_count = np.zeros(num_envs, dtype=np.int64)
    burst_rem = np.zeros(num_envs, dtype=np.int64)
    beliefs = np.full((num_envs, 4), 0.25, dtype=np.float32)
    hist_buf = np.zeros((num_envs, HISTORY_LEN, obs_dim + act_dim), dtype=np.float32)
    rng = np.random.default_rng(seed + 101)
    sample_rng = np.random.default_rng(seed + 202)

    start_iter = 0
    update_count = 0
    real_physics_steps = 0
    episodes_terminated = 0
    episodes_truncated = 0
    burst_steps_applied = 0
    resumed_from_checkpoint = False

    if not ckpt_path.is_file() and any((ckpt_dir / name).is_file() for name in ("train_state.pt", "replay_state.npz")):
        raise ValueError(f"Legacy checkpoint in {ckpt_dir} lacks MuJoCo/RNG state. Archive it before a fresh run.")
    if ckpt_path.is_file():
        with np.load(ckpt_path, allow_pickle=False) as archive:
            state_bytes = io.BytesIO(archive["learner_state"].tobytes())
        saved = torch.load(state_bytes, map_location="cpu", weights_only=False)
        if saved["identity"] != checkpoint_identity:
            raise ValueError(f"Checkpoint identity or runtime differs for {ckpt_path}")
        replay.load_npz(ckpt_path)
        if saved["real_physics_steps"] != saved["next_iter"] * num_envs:
            raise ValueError(f"Checkpoint step budget differs for {ckpt_path}")
        if is_single_mode_bapr and replay.size != saved["real_physics_steps"]:
            raise ValueError(f"Checkpoint replay is incomplete for {ckpt_path}")
        restore_vector_env(venv, saved["vector_env"])
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        target_critic.load_state_dict(saved["target_critic"])
        log_alpha.data.copy_(saved["log_alpha"])
        actor_opt.load_state_dict(saved["actor_opt"])
        critic_opt.load_state_dict(saved["critic_opt"])
        alpha_opt.load_state_dict(saved["alpha_opt"])
        if ext_enc is not None and saved.get("ext_enc") is not None:
            ext_enc.load_state_dict(saved["ext_enc"])
        if hist_enc is not None and saved.get("hist_enc") is not None:
            hist_enc.load_state_dict(saved["hist_enc"])
        if enc_opt is not None and saved.get("enc_opt") is not None:
            enc_opt.load_state_dict(saved["enc_opt"])
        if saved.get("escp_prototypes") is not None:
            escp_prototypes.copy_(saved["escp_prototypes"].to(device))
            escp_proto_valid.copy_(saved["escp_proto_valid"].to(device))
        obs = saved["obs"].copy()
        env_modes = saved["env_modes"].copy()
        dwell_count = saved["dwell_count"].copy()
        burst_rem = saved["burst_rem"].copy()
        beliefs = saved["beliefs"].copy()
        hist_buf = saved["hist_buf"].copy()
        rng.bit_generator.state = saved["rng_state"]
        sample_rng.bit_generator.state = saved["sample_rng_state"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        start_iter = int(saved["next_iter"])
        update_count = int(saved["update_count"])
        real_physics_steps = int(saved["real_physics_steps"])
        episodes_terminated = int(saved["episodes_terminated"])
        episodes_truncated = int(saved["episodes_truncated"])
        burst_steps_applied = int(saved["burst_steps_applied"])
        resumed_from_checkpoint = True
        print(
            f"Resumed {env_key}/{arm}/seed_{seed} at iter {start_iter}/{vec_iters} "
            f"(replay.size={replay.size})",
            flush=True,
        )

    def save_checkpoint(next_it: int) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_bytes = io.BytesIO()
        torch.save({
            "identity": checkpoint_identity,
            "vector_env": snapshot_vector_env(venv),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "log_alpha": log_alpha.data.cpu(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "alpha_opt": alpha_opt.state_dict(),
            "ext_enc": ext_enc.state_dict() if ext_enc is not None else None,
            "hist_enc": hist_enc.state_dict() if hist_enc is not None else None,
            "enc_opt": enc_opt.state_dict() if enc_opt is not None else None,
            "escp_prototypes": escp_prototypes.cpu(),
            "escp_proto_valid": escp_proto_valid.cpu(),
            "obs": obs.copy(),
            "env_modes": env_modes.copy(),
            "dwell_count": dwell_count.copy(),
            "burst_rem": burst_rem.copy(),
            "beliefs": beliefs.copy(),
            "hist_buf": hist_buf.copy(),
            "rng_state": rng.bit_generator.state,
            "sample_rng_state": sample_rng.bit_generator.state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "next_iter": next_it,
            "update_count": update_count,
            "real_physics_steps": real_physics_steps,
            "episodes_terminated": episodes_terminated,
            "episodes_truncated": episodes_truncated,
            "burst_steps_applied": burst_steps_applied,
        }, state_bytes)
        replay.save_npz(ckpt_path, np.frombuffer(state_bytes.getbuffer(), dtype=np.uint8))

    rma_phase1_iters = int(0.8 * vec_iters)  # 400k base + 100k online adapter = 500k
    escp_warmup = 16 if smoke else ESCP_CONTEXT_WARMUP_STEPS
    emit_progress(start_iter * num_envs, total_steps, started, unit="step")

    end_iter = vec_iters if stop_after_iters is None else min(vec_iters, stop_after_iters)
    pbar = tqdm(range(start_iter, end_iter), desc=f"{env_key}:{arm}:s{seed}", mininterval=15.0)
    for it in pbar:
        steps_so_far = it * num_envs

        # Mode dwell switching for multimode arms
        if not is_single_mode_bapr:
            switch_mask = dwell_count >= (16 if smoke else DWELL_STEPS)
            if np.any(switch_mask):
                for e_i in np.flatnonzero(switch_mask):
                    choices = [m for m in range(n_modes) if m != env_modes[e_i]]
                    env_modes[e_i] = int(rng.choice(choices))
                    dwell_count[e_i] = 0

        # Select action
        if steps_so_far < warmup_steps:
            act_np = rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device)
                if is_single_mode_bapr:
                    act_np = actor.sample(obs_t)[0].cpu().numpy()
                    if use_bursts:
                        start_burst = (burst_rem == 0) & (rng.random(num_envs) < BURST_PROB)
                        burst_rem[start_burst] = BURST_LEN
                        active_burst = burst_rem > 0
                        if np.any(active_burst):
                            n_b = int(active_burst.sum())
                            act_np[active_burst] = rng.uniform(-1.0, 1.0, size=(n_b, act_dim)).astype(np.float32)
                            burst_rem[active_burst] -= 1
                            burst_steps_applied += n_b
                elif arm == "sac":
                    act_np = actor.sample(obs_t)[0].cpu().numpy()
                elif arm == "belief_sac":
                    aug_t = torch.cat([obs_t, torch.as_tensor(beliefs, device=device)], dim=-1)
                    act_np = actor.sample(aug_t)[0].cpu().numpy()
                elif arm == "escp256":
                    if steps_so_far >= escp_warmup:
                        ctx_t = hist_enc(torch.as_tensor(hist_buf, device=device))
                    else:
                        ctx_t = torch.zeros((num_envs, 8), dtype=torch.float32, device=device)
                    act_np = actor.sample(torch.cat([obs_t, ctx_t], dim=-1))[0].cpu().numpy()
                elif arm == "rma":
                    if it < rma_phase1_iters:
                        z_t = ext_enc(gains_t[torch.as_tensor(env_modes, device=device)])
                        act_np = actor.sample(torch.cat([obs_t, z_t], dim=-1))[0].cpu().numpy()
                    else:
                        z_hat = hist_enc(torch.as_tensor(hist_buf, device=device))
                        act_np = actor.deterministic(torch.cat([obs_t, z_hat], dim=-1)).cpu().numpy()

        # Apply persistent actuator transform + Gaussian noise before physics step
        g_cur = gains_np[env_modes]
        noise = rng.normal(0.0, NOISE_STD, size=act_np.shape).astype(np.float32)
        exec_act = np.clip(g_cur * act_np + noise, -1.0, 1.0)
        next_obs_raw, rew, terminated, truncated, infos = venv.step(exec_act)
        next_obs_raw = next_obs_raw.astype(np.float32)
        rew = rew.astype(np.float32)
        reset_mask = terminated | truncated

        real_physics_steps += num_envs
        episodes_terminated += int(np.sum(terminated))
        episodes_truncated += int(np.sum(truncated))
        dwell_count += 1

        # Under AutoresetMode.SAME_STEP, true pre-reset terminal observation MUST be in infos["final_obs"]
        real_next_obs = next_obs_raw.copy()
        if np.any(reset_mask):
            final_obs_arr = infos.get("final_obs")
            if final_obs_arr is None:
                raise RuntimeError(f"SyncVectorEnv SAME_STEP missing infos['final_obs'] (keys={list(infos.keys())})")
            for e_i in np.flatnonzero(reset_mask):
                if final_obs_arr[e_i] is None:
                    raise RuntimeError(f"SyncVectorEnv SAME_STEP infos['final_obs'][{e_i}] is None on reset!")
                real_next_obs[e_i] = np.asarray(final_obs_arr[e_i], dtype=np.float32)

        prev_beliefs = beliefs.copy()
        prev_hist = hist_buf.copy()

        if arm == "belief_sac":
            scores = joint_emission_scores(
                frozen_inv, frozen_var, gains_t,
                torch.as_tensor(obs, device=device),
                torch.as_tensor(act_np, device=device),
                torch.as_tensor(real_next_obs, device=device),
            )
            next_b = update_sticky_belief(torch.as_tensor(prev_beliefs, device=device), scores).cpu().numpy()
            beliefs = next_b.copy()
            if np.any(reset_mask):
                beliefs[reset_mask] = 0.25
            replay.add_batch(obs, act_np, rew, real_next_obs, terminated.astype(np.float32), env_modes, prev_beliefs, next_b)
        elif arm in ("escp256", "rma"):
            step_pair = np.concatenate([obs, act_np], axis=-1)[:, None, :]
            next_hist = np.concatenate([prev_hist[:, 1:, :], step_pair], axis=1)
            if arm == "escp256":
                replay.add_batch(obs, act_np, rew, real_next_obs, terminated.astype(np.float32), env_modes, prev_hist, next_hist)
            elif it < rma_phase1_iters:
                replay.add_batch(obs, act_np, rew, real_next_obs, terminated.astype(np.float32), env_modes)
            else:
                h_t = torch.as_tensor(prev_hist, device=device)
                with torch.no_grad():
                    z_target = ext_enc(gains_t[torch.as_tensor(env_modes, device=device)])
                z_pred = hist_enc(h_t)
                loss_adapt = F.mse_loss(z_pred, z_target)
                enc_opt.zero_grad(set_to_none=True)
                loss_adapt.backward()
                nn.utils.clip_grad_norm_(hist_enc.parameters(), 10.0)
                enc_opt.step()
                update_count += 1
            hist_buf = next_hist
            if np.any(reset_mask):
                hist_buf[reset_mask] = 0.0
        else:
            replay.add_batch(obs, act_np, rew, real_next_obs, terminated.astype(np.float32), env_modes)

        if np.any(reset_mask):
            dwell_count[reset_mask] = 0
            burst_rem[reset_mask] = 0
            if not is_single_mode_bapr:
                env_modes[reset_mask] = rng.integers(0, n_modes, size=int(reset_mask.sum()))

        obs = next_obs_raw

        # Gradient updates
        if replay.size >= max(BATCH_SIZE, warmup_steps) and not (arm == "rma" and it >= rma_phase1_iters):
            n_upd = 2 if smoke else UPDATES_PER_VEC_STEP
            for _ in range(n_upd):
                batch = replay.sample(BATCH_SIZE, device, rng=sample_rng)
                alpha = log_alpha.exp().detach()

                if arm in ("bapr", "bapr_noburst", "sac"):
                    b_obs, b_act, b_rew, b_nobs, b_done, _ = batch
                    b_in_critic, b_nin_critic = b_obs, b_nobs
                    b_in_actor, b_nin_actor = b_obs, b_nobs
                elif arm == "belief_sac":
                    b_obs, b_act, b_rew, b_nobs, b_done, _, b_ext, b_next = batch
                    b_in_critic = torch.cat([b_obs, b_ext], dim=-1)
                    b_nin_critic = torch.cat([b_nobs, b_next], dim=-1)
                    b_in_actor, b_nin_actor = b_in_critic, b_nin_critic
                elif arm == "rma":
                    b_obs, b_act, b_rew, b_nobs, b_done, b_mode = batch
                    z_c = ext_enc(gains_t[b_mode])
                    b_in_critic = torch.cat([b_obs, z_c], dim=-1)
                    b_nin_critic = torch.cat([b_nobs, z_c.detach()], dim=-1)
                    b_in_actor = b_in_critic.detach()
                    b_nin_actor = b_nin_critic
                elif arm == "escp256":
                    b_obs, b_act, b_rew, b_nobs, b_done, b_mode, b_hist, b_nhist = batch
                    context_active = steps_so_far >= escp_warmup
                    # RMDM-style context update and EMA task prototypes.
                    if context_active:
                        ctx_train = hist_enc(b_hist)
                        enc_loss = compute_escp_rmdm_loss(
                            ctx_train, b_mode, escp_prototypes, escp_proto_valid,
                        )
                        enc_opt.zero_grad(set_to_none=True)
                        enc_loss.backward()
                        nn.utils.clip_grad_norm_(hist_enc.parameters(), 10.0)
                        enc_opt.step()

                    with torch.no_grad():
                        inferred_cur = hist_enc(b_hist)
                        inferred_nxt = hist_enc(b_nhist)
                        escp_prototypes, escp_proto_valid = update_escp_prototypes(
                            escp_prototypes, escp_proto_valid, inferred_cur, b_mode,
                        )
                        if context_active:
                            valid_m = escp_proto_valid[b_mode]
                            task_ctx = torch.where(valid_m.unsqueeze(-1), escp_prototypes[b_mode], inferred_cur)
                            act_ctx_cur = inferred_cur + ESCP_BOTTLENECK_SIGMA * torch.randn_like(inferred_cur)
                            act_ctx_nxt = inferred_nxt + ESCP_BOTTLENECK_SIGMA * torch.randn_like(inferred_nxt)
                        else:
                            zeros_ctx = torch.zeros_like(inferred_cur)
                            task_ctx = zeros_ctx
                            act_ctx_cur = zeros_ctx
                            act_ctx_nxt = zeros_ctx
                    b_in_critic = torch.cat([b_obs, task_ctx], dim=-1)
                    b_nin_critic = torch.cat([b_nobs, task_ctx], dim=-1)
                    b_in_actor = torch.cat([b_obs, act_ctx_cur], dim=-1)
                    b_nin_actor = torch.cat([b_nobs, act_ctx_nxt], dim=-1)

                with torch.no_grad():
                    nxt_a, nxt_lp = actor.sample(b_nin_actor)
                    tq1, tq2 = target_critic(b_nin_critic, nxt_a)
                    tmin = torch.min(tq1, tq2) - alpha * nxt_lp
                    backup = b_rew + GAMMA * (1.0 - b_done) * tmin

                q1, q2 = critic(b_in_critic, b_act)
                c_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
                critic_opt.zero_grad(set_to_none=True)
                c_loss.backward()
                critic_opt.step()

                pi_a, pi_lp = actor.sample(b_in_actor)
                aq1, aq2 = critic(b_in_critic.detach(), pi_a)
                q_pi = 0.5 * (aq1 + aq2) if arm != "escp256" else torch.min(aq1, aq2)
                a_loss = (alpha * pi_lp - q_pi).mean()
                actor_opt.zero_grad(set_to_none=True)
                a_loss.backward()
                actor_opt.step()

                al_loss = -(log_alpha * (pi_lp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad(set_to_none=True)
                al_loss.backward()
                alpha_opt.step()

                soft_update(target_critic, critic, TAU)
                update_count += 1

        done_steps = (it + 1) * num_envs
        if not smoke and (done_steps % 50_000 == 0 or it + 1 == end_iter):
            emit_progress(done_steps, total_steps, started, unit="step")
        if (not smoke and (done_steps % 100_000 == 0 or it + 1 == end_iter)) or (stop_after_iters is not None and it + 1 == end_iter):
            save_checkpoint(it + 1)

    venv.close()

    if stop_after_iters is not None and end_iter < vec_iters:
        return {"partial": True, "next_iter": end_iter, "replay_size": replay.size}

    if real_physics_steps != total_steps:
        raise RuntimeError(f"Expected {total_steps} real physics steps, got {real_physics_steps}")
    if is_single_mode_bapr and replay.size != total_steps:
        raise RuntimeError(f"Expected full replay size {total_steps} for {arm}, got {replay.size}")

    dest.mkdir(parents=True, exist_ok=True)
    actor_payload = {
        "protocol": PROTOCOL,
        "actor": actor.cpu().state_dict(),
        "in_dim": in_dim,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "leaky": leaky,
        "hist_enc": hist_enc.cpu().state_dict() if hist_enc is not None else None,
        "ext_enc": ext_enc.cpu().state_dict() if ext_enc is not None else None,
    }
    torch.save(actor_payload, dest / "actor.pt")

    inv_meta = None
    fwd_meta = None
    if is_single_mode_bapr:
        inv_model, variance = fit_self_calibrated_inverse(
            replay, obs_dim, act_dim, gains_np[0], seed, device, updates=inv_updates
        )
        inv_meta = {
            "source": "self_calibrated_from_own_reference_mode_replay_buffer",
            "external_provider_interactions": 0,
            "multimode_labels_used": False,
            "exploratory_bursts_enabled": use_bursts,
            "burst_length": BURST_LEN if use_bursts else 0,
            "burst_trigger_prob": BURST_PROB if use_bursts else 0.0,
            "burst_steps_applied": int(burst_steps_applied),
            "inverse_updates": inv_updates,
            "residual_variance": variance.tolist(),
        }
        torch.save({
            "state_dict": inv_model.cpu().state_dict(),
            "residual_variance": variance,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        }, dest / "inverse.pt")

        if arm == "bapr":
            fwd_model, fwd_var = fit_self_calibrated_shared_forward(
                replay, obs_dim, act_dim, gains_np[0], seed, device, updates=inv_updates
            )
            fwd_meta = {
                "source": "shared_physics_forward_on_same_mode0_replay_split",
                "forward_updates": inv_updates,
                "residual_variance_mean": float(np.mean(fwd_var)),
            }
            torch.save({
                "state_dict": fwd_model.cpu().state_dict(),
                "residual_variance": fwd_var,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            }, dest / "forward_shared.pt")

    manifest = {
        "protocol": PROTOCOL,
        "env": env_key,
        "gym_id": ENV_SPECS[env_key][0],
        "arm": arm,
        "seed": seed,
        "autoreset_mode": AUTORESET_MODE,
        "total_steps": total_steps,
        "real_physics_steps_executed": int(real_physics_steps),
        "episodes_terminated_during_training": int(episodes_terminated),
        "episodes_truncated_during_training": int(episodes_truncated),
        "resumed_from_checkpoint": bool(resumed_from_checkpoint),
        "update_count": int(update_count),
        "external_interactions": 0,
        "escp_objective": "luo2022_rmdm_consistency_plus_rbf_logdet" if arm == "escp256" else None,
        "inverse_calibration": inv_meta,
        "shared_forward_calibration": fwd_meta,
        "runtime_versions": runtime,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(dest / "bundle.json", manifest)
    if not smoke:
        ckpt_path.unlink(missing_ok=True)
    print(f"Completed {env_key}/{arm}/seed_{seed} in {manifest['elapsed_seconds']:.1f}s", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Deterministic Paired Audit Across Primary, Hypercube (M=64/256), and Lossy Regimes
# ---------------------------------------------------------------------------

def load_trained_bundle(env_key: str, arm: str, seed: int, device: torch.device):
    dest = bundle_dir(env_key, arm, seed)
    bmeta = json.loads((dest / "bundle.json").read_text())
    if bmeta.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Stale bundle protocol for {dest}: {bmeta.get('protocol')}")
    data = torch.load(dest / "actor.pt", map_location=device, weights_only=False)
    actor = SquashedGaussianActor(data["in_dim"], data["act_dim"], hidden=256, leaky=data["leaky"]).to(device)
    actor.load_state_dict(data["actor"])
    actor.eval()
    hist_enc = None
    if data.get("hist_enc") is not None:
        hist_enc = HistoryContextEncoder(data["obs_dim"], data["act_dim"], 8, 256, 128).to(device)
        hist_enc.load_state_dict(data["hist_enc"])
        hist_enc.eval()
    inv_model, variance = None, None
    if arm in ("bapr", "bapr_noburst"):
        inv_data = torch.load(dest / "inverse.pt", map_location=device, weights_only=False)
        inv_model = VectorizedInverseEnsemble(data["obs_dim"], data["act_dim"], 5, 256).to(device)
        inv_model.load_state_dict(inv_data["state_dict"])
        inv_model.eval()
        variance = torch.as_tensor(inv_data["residual_variance"], dtype=torch.float32, device=device)
    fwd_model, fwd_var = None, None
    if arm == "bapr" and (dest / "forward_shared.pt").is_file():
        fwd_data = torch.load(dest / "forward_shared.pt", map_location=device, weights_only=False)
        fwd_model = VectorizedForwardEnsemble(data["obs_dim"], data["act_dim"], 5, 256).to(device)
        fwd_model.load_state_dict(fwd_data["state_dict"])
        fwd_model.eval()
        fwd_var = torch.as_tensor(fwd_data["residual_variance"], dtype=torch.float32, device=device)
    return actor, hist_enc, inv_model, variance, fwd_model, fwd_var


BELIEF_ROUTES = {
    "bapr_selfcal",
    "bapr_soft",
    "bapr_factored",
    "bapr_d100_lockin",
    "bapr_instantaneous",
    "bapr_forward_shared",
    "bapr_noburst_selfcal",
    "belief_sac",
}


def polarity_mean_command(belief, gains, target):
    """Box-feasible posterior expected-error minimizer for sign gains."""
    return (belief @ gains) * target


def rollout_episode(
    env,
    schedule: np.ndarray,
    gains_np: np.ndarray,
    ep_seed: int,
    route: str,
    models: dict,
    device: torch.device,
    horizon: int = EPISODE_HORIZON,
    dwell: int = DWELL_STEPS,
    *,
    physical_gains: np.ndarray | None = None,
) -> dict:
    obs, _ = env.reset(seed=ep_seed)
    obs = obs.astype(np.float32)
    obs_dim = obs.shape[0]
    act_dim = gains_np.shape[1]
    n_modes = gains_np.shape[0]
    gains_t = torch.as_tensor(gains_np, device=device)
    ref_gain = gains_np[0]
    plant_gains = gains_np if physical_gains is None else physical_gains
    noise_rng = np.random.default_rng(ep_seed * 31 + 7)

    belief = torch.full((1, n_modes), 1.0 / n_modes, dtype=torch.float32, device=device)
    prob_pos = torch.full((1, act_dim), 0.5, dtype=torch.float32, device=device)
    hist_buf = np.zeros((1, HISTORY_LEN, obs_dim + act_dim), dtype=np.float32)

    rewards = []
    true_modes = []
    exact_matches = []
    sq_errors = []
    sq_commands = []
    terminated_early = False

    bapr_actor, _, inv_model, inv_var, fwd_model, fwd_var = models["bapr"]
    nb_actor, _, nb_inv_model, nb_inv_var, _, _ = models["bapr_noburst"]

    for step in range(horizon):
        true_mode = int(schedule[(step // dwell) % len(schedule)])
        phys_mode = 0 if route == "native_reference" else true_mode
        obs_t = torch.as_tensor(obs[None, :], device=device)

        with torch.no_grad():
            if route == "native_reference":
                q = bapr_actor.deterministic(obs_t)[0].cpu().numpy()
                cmd = q
                exact = True
            elif route == "oracle":
                q = bapr_actor.deterministic(obs_t)[0].cpu().numpy()
                cmd = np.clip((ref_gain * q) / gains_np[true_mode], -1.0, 1.0)
                exact = True
            elif route == "bapr_noburst_oracle":
                q = nb_actor.deterministic(obs_t)[0].cpu().numpy()
                cmd = np.clip((ref_gain * q) / gains_np[true_mode], -1.0, 1.0)
                exact = True
            elif route == "uncompensated":
                q = bapr_actor.deterministic(obs_t)[0].cpu().numpy()
                cmd = q
                exact = (true_mode == 0)
            elif route in ("bapr_selfcal", "bapr_soft", "bapr_d100_lockin", "bapr_instantaneous", "bapr_forward_shared"):
                q = bapr_actor.deterministic(obs_t)[0].cpu().numpy()
                est_mode = int(torch.argmax(belief[0]).item())
                cmd = np.clip((ref_gain * q) / gains_np[est_mode], -1.0, 1.0)
                if route == "bapr_soft" and step > 0:
                    cmd = polarity_mean_command(belief[0].cpu().numpy(), gains_np, ref_gain * q)
                exact = (est_mode == true_mode)
            elif route == "bapr_noburst_selfcal":
                q = nb_actor.deterministic(obs_t)[0].cpu().numpy()
                est_mode = int(torch.argmax(belief[0]).item())
                cmd = np.clip((ref_gain * q) / gains_np[est_mode], -1.0, 1.0)
                exact = (est_mode == true_mode)
            elif route == "bapr_factored":
                q = bapr_actor.deterministic(obs_t)[0].cpu().numpy()
                est_gain = np.where(prob_pos[0].cpu().numpy() >= 0.5, 1.0, -1.0).astype(np.float32)
                cmd = est_gain * (ref_gain * q)
                exact = bool(np.all(est_gain == gains_np[true_mode]))
            elif route == "sac":
                cmd = models["sac"][0].deterministic(obs_t)[0].cpu().numpy()
                q = cmd
                exact = False
            elif route in ("escp256", "rma"):
                actor_m, hist_m = models[route][0], models[route][1]
                ctx = hist_m(torch.as_tensor(hist_buf, device=device))
                cmd = actor_m.deterministic(torch.cat([obs_t, ctx], dim=-1))[0].cpu().numpy()
                q = cmd
                exact = False
            elif route == "belief_sac":
                actor_m = models["belief_sac"][0]
                cmd = actor_m.deterministic(torch.cat([obs_t, belief], dim=-1))[0].cpu().numpy()
                q = cmd
                exact = (int(torch.argmax(belief[0]).item()) == true_mode)
            else:
                raise ValueError(f"unknown route {route}")

        noise = noise_rng.normal(0.0, NOISE_STD, size=cmd.shape).astype(np.float32)
        exec_act = np.clip(plant_gains[phys_mode] * cmd + noise, -1.0, 1.0)
        next_obs, rew, term, trunc, _ = env.step(exec_act)
        next_obs = next_obs.astype(np.float32)

        realized_nom = np.clip(plant_gains[phys_mode] * cmd, -1.0, 1.0)
        ref_nom = ref_gain * q
        sq_errors.append(float(np.sum((realized_nom - ref_nom) ** 2)))
        sq_commands.append(float(np.dot(cmd, cmd)))
        rewards.append(float(rew))
        true_modes.append(true_mode)
        exact_matches.append(bool(exact))

        cmd_t = torch.as_tensor(cmd[None, :], device=device)
        nobs_t = torch.as_tensor(next_obs[None, :], device=device)
        if route in ("bapr_selfcal", "bapr_soft", "belief_sac"):
            sc = joint_emission_scores(inv_model, inv_var, gains_t, obs_t, cmd_t, nobs_t)
            belief = update_sticky_belief(belief, sc, hazard=0.002, scale=1.0, decay=0.98, temporal=True)
        elif route == "bapr_noburst_selfcal":
            sc = joint_emission_scores(nb_inv_model, nb_inv_var, gains_t, obs_t, cmd_t, nobs_t)
            belief = update_sticky_belief(belief, sc, hazard=0.002, scale=1.0, decay=0.98, temporal=True)
        elif route == "bapr_forward_shared":
            sc = shared_forward_emission_scores(fwd_model, fwd_var, gains_t, obs_t, cmd_t, nobs_t)
            belief = update_sticky_belief(belief, sc, hazard=0.002, scale=1.0, decay=0.98, temporal=True)
        elif route == "bapr_d100_lockin":
            sc = joint_emission_scores(inv_model, inv_var, gains_t, obs_t, cmd_t, nobs_t)
            belief = update_sticky_belief(belief, sc, hazard=0.0, scale=1.0, decay=1.00, temporal=True)
        elif route == "bapr_instantaneous":
            sc = joint_emission_scores(inv_model, inv_var, gains_t, obs_t, cmd_t, nobs_t)
            belief = update_sticky_belief(belief, sc, temporal=False)
        elif route == "bapr_factored":
            prob_pos = update_factored_channel_belief(prob_pos, inv_model, inv_var, obs_t, cmd_t, nobs_t)
        elif route in ("escp256", "rma"):
            pair = np.concatenate([obs, cmd], axis=-1)[None, None, :]
            hist_buf = np.concatenate([hist_buf[:, 1:, :], pair], axis=1)

        obs = next_obs
        if term or trunc:
            if term and step + 1 < horizon:
                terminated_early = True
            break

    rew_arr = np.asarray(rewards, dtype=np.float64)
    tm_arr = np.asarray(true_modes, dtype=np.int64)
    em_arr = np.asarray(exact_matches, dtype=bool)
    switched = np.flatnonzero(tm_arr[1:] != tm_arr[:-1]) + 1 if len(tm_arr) > 1 else np.asarray([], dtype=np.int64)
    scheduled_switches = [
        s for s in range(dwell, horizon, dwell)
        if schedule[(s // dwell) % len(schedule)] != schedule[(s // dwell - 1) % len(schedule)]
    ]

    has_belief = route in BELIEF_ROUTES
    d1_list, d8_list = [], []
    unres_1 = unres_8 = 0
    post_switch_exact = 0
    post_switch_steps = 0
    records = []
    if has_belief:
        if len(em_arr) > dwell:
            post_switch_exact = int(np.sum(em_arr[dwell:]))
            post_switch_steps = int(len(em_arr) - dwell)
    for idx, st in enumerate(scheduled_switches):
        en = min(len(tm_arr), scheduled_switches[idx + 1] if idx + 1 < len(scheduled_switches) else horizon)
        reached = st < en
        d1 = d8 = None
        if has_belief and reached:
            d1 = next((s - st for s in range(st, en) if em_arr[s]), None)
            if d1 is None:
                unres_1 += 1
            else:
                d1_list.append(int(d1))
            d8 = next((s - st for s in range(st, en - 7) if np.all(em_arr[s:s + 8])), None)
            if d8 is None:
                unres_8 += 1
            else:
                d8_list.append(int(d8))
        stop = min(st + 32, en)
        records.append({
            "step": st, "reached": reached, "first_delay": d1, "sustained_delay": d8,
            "steps_32": max(stop - st, 0),
            "reward_32": float(rew_arr[st:stop].sum()),
            "action_error_32": float(sum(sq_errors[st:stop])),
            "command_energy_32": float(sum(sq_commands[st:stop])),
        })

    return {
        "return": float(rew_arr.sum()),
        "rewards": rew_arr,
        "horizon": horizon,
        "scheduled_switches": scheduled_switches,
        "steps": len(rewards),
        "terminated_early": terminated_early,
        "terminated": bool(term),
        "truncated": bool(trunc),
        "switch_records": records,
        "episode_seed": ep_seed,
        "schedule": schedule.tolist(),
        "switches_scheduled": len(scheduled_switches),
        "switches_observed": len(switched),
        "switched": switched,
        "exact_steps": int(np.sum(em_arr)) if has_belief else None,
        "post_switch_exact_steps": post_switch_exact if has_belief else None,
        "post_switch_steps": post_switch_steps if has_belief else None,
        "mode_accuracy": float(np.mean(em_arr)) if has_belief else None,
        "action_error_energy": float(np.sum(sq_errors)),
        "first_delays": d1_list,
        "unresolved_first": unres_1,
        "sustained_delays": d8_list,
        "unresolved_sustained": unres_8,
    }


def reduce_episode_group(rows: list[dict], oracle_rows: list[dict] | None = None, stat_returns: list[float] | None = None) -> dict:
    d1 = [d for r in rows for d in r["first_delays"]]
    d8 = [d for r in rows for d in r["sustained_delays"]]
    has_acc = any(r["mode_accuracy"] is not None for r in rows)
    total_steps = sum(r["steps"] for r in rows)
    total_exact = sum(r["exact_steps"] for r in rows if r["exact_steps"] is not None) if has_acc else 0
    ps_steps = sum(r["post_switch_steps"] for r in rows if r["post_switch_steps"] is not None) if has_acc else 0
    ps_exact = sum(r["post_switch_exact_steps"] for r in rows if r["post_switch_exact_steps"] is not None) if has_acc else 0

    full_regrets = []
    cond_regrets = []
    if oracle_rows is not None:
        if len(rows) != len(oracle_rows):
            raise ValueError("rows vs oracle_rows length mismatch")
        for r, o in zip(rows, oracle_rows):
            H = int(o["horizon"])
            r_pad = np.zeros(H, dtype=np.float64)
            o_pad = np.zeros(H, dtype=np.float64)
            r_pad[:len(r["rewards"])] = r["rewards"]
            o_pad[:len(o["rewards"])] = o["rewards"]
            # Full-horizon zero-padded 32-step regret across all scheduled switch windows
            reg_full = 0.0
            for st in o["scheduled_switches"]:
                en = min(int(st) + 32, H)
                reg_full += float(o_pad[int(st):en].sum() - r_pad[int(st):en].sum())
            full_regrets.append(reg_full)
            # Survival-conditioned regret on switches reached by both trajectories (still zero-padded inside [st, st+32))
            for st in o["scheduled_switches"]:
                if len(r["rewards"]) > int(st) and len(o["rewards"]) > int(st):
                    en = min(int(st) + 32, H)
                    cond_regrets.append(float(o_pad[int(st):en].sum() - r_pad[int(st):en].sum()))

    sw_sched = sum(r["switches_scheduled"] for r in rows)
    sw_obs = sum(r["switches_observed"] for r in rows)
    return {
        "switching_mean": statistics.fmean(r["return"] for r in rows),
        "stationary_mean": statistics.fmean(stat_returns) if stat_returns else None,
        "termination_rate": statistics.fmean(1.0 if r["terminated_early"] else 0.0 for r in rows),
        "mean_steps": statistics.fmean(r["steps"] for r in rows),
        "mode_accuracy": (float(total_exact) / float(total_steps)) if (has_acc and total_steps > 0) else None,
        "post_switch_mode_accuracy": (float(ps_exact) / float(ps_steps)) if (has_acc and ps_steps > 0) else None,
        "post_switch_steps_total": int(ps_steps) if has_acc else 0,
        "action_error_energy": statistics.fmean(r["action_error_energy"] for r in rows),
        "first_delay_median": float(np.median(d1)) if d1 else None,
        "unresolved_first": sum(r["unresolved_first"] for r in rows),
        "sustained_delay_median": float(np.median(d8)) if d8 else None,
        "unresolved_sustained": sum(r["unresolved_sustained"] for r in rows),
        "switches_scheduled_total": int(sw_sched),
        "switches_observed_total": int(sw_obs),
        "switch_coverage": (float(sw_obs) / float(sw_sched)) if sw_sched > 0 else 0.0,
        "post_switch_regret_32": statistics.fmean(full_regrets) if full_regrets else None,
        "conditional_post_switch_regret_32": statistics.fmean(cond_regrets) if cond_regrets else None,
    }


def compact_episodes(group: dict) -> dict:
    return {route: [{k: v for k, v in row.items() if k not in ("rewards", "switched")}
                    for row in rows] for route, rows in group.items()}


def audit_seed(env_key: str, seed: int, smoke: bool = False, episode_dir: Path | None = None) -> dict:
    runtime = runtime_info(strict=not smoke)
    out_path = RELEASE_DIR / "audits" / env_key / f"seed_{seed}.json"
    if not smoke and episode_dir is None and out_path.is_file():
        existing = json.loads(out_path.read_text())
        if existing.get("protocol") == PROTOCOL:
            print(f"Audit already exists: {out_path}", flush=True)
            return existing

    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {arm: load_trained_bundle(env_key, arm, seed, device) for arm in ARMS}
    arm_step_records = {
        arm: json.loads((bundle_dir(env_key, arm, seed) / "bundle.json").read_text())["real_physics_steps_executed"]
        for arm in ARMS
    }

    env = make_single_env(env_key)
    act_dim = int(env.action_space.shape[0])
    gains_4 = canonical_polarity_gains(act_dim)
    horizon = 64 if smoke else EPISODE_HORIZON
    dwell = 16 if smoke else DWELL_STEPS
    ev_seeds = SWITCH_EVENT_SEEDS[:1] if smoke else SWITCH_EVENT_SEEDS
    ep_per_ev = 1 if smoke else 5

    primary_routes = (
        "native_reference", "oracle", "uncompensated",
        "bapr_selfcal", "bapr_factored", "bapr_d100_lockin", "bapr_instantaneous", "bapr_forward_shared",
        "bapr_noburst_oracle", "bapr_noburst_selfcal",
        "sac", "escp256", "rma", "belief_sac",
    )
    sw_rows = {r: [] for r in primary_routes}
    st_rows = {
        r: [] for r in (
            "oracle", "bapr_selfcal", "bapr_factored", "bapr_forward_shared",
            "bapr_noburst_oracle", "bapr_noburst_selfcal",
            "sac", "escp256", "rma", "belief_sac",
        )
    }
    stationary_records = {r: [] for r in st_rows}

    for ev_seed in ev_seeds:
        base = np.asarray(SWITCH_BASE_SCHEDULES[ev_seed], dtype=np.int64)
        for ep_i in range(ep_per_ev):
            sched = np.roll(base, ep_i % 4)
            ep_seed = ev_seed * 100 + ep_i
            for r in primary_routes:
                sw_rows[r].append(rollout_episode(env, sched, gains_4, ep_seed, r, models, device, horizon, dwell))

    for mode in range(1 if smoke else 4):
        for ep_i in range(1 if smoke else 5):
            sched = np.full(4, mode, dtype=np.int64)
            ep_seed = STATIONARY_EVENT_SEED * 100 + mode * 10 + ep_i
            for r in st_rows:
                res = rollout_episode(env, sched, gains_4, ep_seed, r, models, device, horizon, dwell)
                st_rows[r].append(res["return"])
                if episode_dir is not None:
                    stationary_records[r].append(res)

    primary_summary = {
        r: reduce_episode_group(sw_rows[r], sw_rows["oracle"] if r != "oracle" else None, st_rows.get(r))
        for r in primary_routes
    }

    # Hypercube 2^d evaluation on HalfCheetah (M=64) and Ant (M=256)
    hypercube_summary = None
    hc_rows = {}
    if env_key in ("halfcheetah", "ant"):
        gains_hc = hypercube_polarity_gains(act_dim)
        m_total = gains_hc.shape[0]
        rng = np.random.default_rng(20265100 + seed % 100)
        hc_routes = ("oracle", "uncompensated", "bapr_factored")
        hc_rows = {r: [] for r in hc_routes}
        n_hc_ep = 2 if smoke else 15
        visited_modes = set()
        for ep_i in range(n_hc_ep):
            if ep_i < 9:
                sched = rng.choice(m_total, size=4, replace=False)
            else:
                cur = gains_hc[int(rng.integers(0, m_total))].copy()
                joints = rng.choice(act_dim, size=4, replace=False)
                seq = []
                for b in range(4):
                    if b > 0:
                        cur[joints[b]] *= -1.0
                    idx = int(np.where(np.all(gains_hc == cur[None, :], axis=1))[0][0])
                    seq.append(idx)
                sched = np.asarray(seq, dtype=np.int64)
            visited_modes.update(int(x) for x in sched)
            ep_seed = 410000 + ep_i
            for r in hc_routes:
                hc_rows[r].append(rollout_episode(env, sched, gains_hc, ep_seed, r, models, device, horizon, dwell))
        hypercube_summary = {
            "num_modes": int(m_total),
            "distinct_modes_sampled": len(visited_modes),
            "routes": {r: reduce_episode_group(hc_rows[r], hc_rows["oracle"] if r != "oracle" else None) for r in hc_routes},
        }

    # Lossy 0.5x actuator attenuation projection evaluation
    gains_lossy = lossy_degradation_gains(act_dim)
    lossy_routes = ("native_reference", "oracle", "bapr_selfcal", "uncompensated")
    lossy_rows = {r: [] for r in lossy_routes}
    for ev_seed in ev_seeds:
        base = np.asarray(SWITCH_BASE_SCHEDULES[ev_seed], dtype=np.int64)
        for ep_i in range(ep_per_ev):
            sched = np.roll(base, ep_i % 4)
            ep_seed = (ev_seed + 70000) * 100 + ep_i
            for r in lossy_routes:
                lossy_rows[r].append(rollout_episode(env, sched, gains_lossy, ep_seed, r, models, device, horizon, dwell))
    lossy_summary = {
        r: reduce_episode_group(lossy_rows[r], lossy_rows["oracle"] if r != "oracle" else None)
        for r in lossy_routes
    }

    env.close()
    payload = {
        "protocol": PROTOCOL,
        "env": env_key,
        "gym_id": ENV_SPECS[env_key][0],
        "seed": seed,
        "autoreset_mode": AUTORESET_MODE,
        "runtime_versions": runtime,
        "online_budget_per_arm": TOTAL_STEPS,
        "real_physics_steps_by_arm": arm_step_records,
        "external_interactions_bapr": 0,
        "recipe": {
            "burst_length": BURST_LEN,
            "burst_trigger_prob": BURST_PROB,
            "inverse_updates": INVERSE_UPDATES,
            "escp_objective": "luo2022_rmdm_consistency_plus_rbf_logdet",
        },
        "primary_4mode": primary_summary,
        "hypercube_2d": hypercube_summary,
        "lossy_projection": lossy_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    emit_progress(1, 1, started, unit="audit")
    if episode_dir is not None:
        write_json(episode_dir / env_key / f"seed_{seed}.json", {
            **payload,
            "episodes": {
                "primary_4mode": compact_episodes(sw_rows),
                "stationary_4mode": compact_episodes(stationary_records),
                "hypercube_2d": compact_episodes(hc_rows),
                "lossy_projection": compact_episodes(lossy_rows),
            },
        })
    elif not smoke:
        write_json(out_path, payload)
    print(f"Completed audit {env_key}/seed_{seed} in {payload['elapsed_seconds']:.1f}s", flush=True)
    return payload


def paired_stats(xs: list[float], ys: list[float]) -> dict:
    if len(xs) != len(ys):
        raise ValueError(f"paired_stats length mismatch: {len(xs)} != {len(ys)}")
    diffs = [a - b for a, b in zip(xs, ys)]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n_seeds": n,
        "mean": mean,
        "ci95": [mean - half, mean + half],
        "wins": sum(d > 0 for d in diffs),
    }


def analyze_all() -> dict:
    summary = {
        "protocol": PROTOCOL,
        "autoreset_mode": AUTORESET_MODE,
        "seeds": list(SEEDS),
        "recipe": {
            "total_steps": TOTAL_STEPS,
            "num_envs": NUM_ENVS,
            "warmup_steps": WARMUP_STEPS,
            "burst_length": BURST_LEN,
            "burst_trigger_prob": BURST_PROB,
            "inverse_updates": INVERSE_UPDATES,
            "escp_objective": "luo2022_rmdm_consistency_plus_rbf_logdet",
        },
        "environments": {},
    }
    for env_key in ENV_NAMES:
        rows = [json.loads((RELEASE_DIR / "audits" / env_key / f"seed_{s}.json").read_text()) for s in SEEDS]
        for row in rows:
            if row.get("protocol") != PROTOCOL:
                raise RuntimeError(f"Audit {env_key}/seed_{row.get('seed')} has stale protocol {row.get('protocol')}")
            for arm_k, st_cnt in row["real_physics_steps_by_arm"].items():
                if st_cnt != TOTAL_STEPS:
                    raise RuntimeError(f"Arm {env_key}/{arm_k}/seed_{row['seed']} executed {st_cnt} != {TOTAL_STEPS}")
        summary["runtime_versions"] = rows[0]["runtime_versions"]

        def agg_group(getter):
            routes = list(getter(rows[0]).keys())
            out = {}
            for r in routes:
                per_s = [getter(row)[r] for row in rows]
                st = [x["stationary_mean"] for x in per_s if x["stationary_mean"] is not None]
                ac = [x["mode_accuracy"] for x in per_s if x["mode_accuracy"] is not None]
                ps_ac = [x["post_switch_mode_accuracy"] for x in per_s if x.get("post_switch_mode_accuracy") is not None]
                d1 = [x["first_delay_median"] for x in per_s if x["first_delay_median"] is not None]
                d8 = [x["sustained_delay_median"] for x in per_s if x["sustained_delay_median"] is not None]
                rg = [x["post_switch_regret_32"] for x in per_s if x["post_switch_regret_32"] is not None]
                crg = [x["conditional_post_switch_regret_32"] for x in per_s if x.get("conditional_post_switch_regret_32") is not None]
                sw_sched_tot = sum(x["switches_scheduled_total"] for x in per_s)
                sw_obs_tot = sum(x["switches_observed_total"] for x in per_s)
                seeds_w_sw = sum(1 for x in per_s if x["switches_observed_total"] > 0)
                out[r] = {
                    "switching_mean": statistics.fmean(x["switching_mean"] for x in per_s),
                    "switching_seed_means": [x["switching_mean"] for x in per_s],
                    "stationary_mean": statistics.fmean(st) if st else None,
                    "termination_rate": statistics.fmean(x["termination_rate"] for x in per_s),
                    "mean_steps": statistics.fmean(x["mean_steps"] for x in per_s),
                    "mode_accuracy": statistics.fmean(ac) if ac else None,
                    "post_switch_mode_accuracy": statistics.fmean(ps_ac) if ps_ac else None,
                    "seeds_with_post_switch_data": len(ps_ac),
                    "first_delay_median_of_medians": float(np.median(d1)) if d1 else None,
                    "sustained_delay_median_of_medians": float(np.median(d8)) if d8 else None,
                    "seeds_with_observed_switches": int(seeds_w_sw),
                    "unresolved_sustained_total": sum(x["unresolved_sustained"] for x in per_s),
                    "switches_scheduled_total": int(sw_sched_tot),
                    "switches_observed_total": int(sw_obs_tot),
                    "switch_coverage": (float(sw_obs_tot) / float(sw_sched_tot)) if sw_sched_tot > 0 else 0.0,
                    "post_switch_regret_32": statistics.fmean(rg) if rg else None,
                    "conditional_post_switch_regret_32": statistics.fmean(crg) if crg else None,
                }
            return out

        prim = agg_group(lambda row: row["primary_4mode"])
        lossy = agg_group(lambda row: row["lossy_projection"])
        hc = None
        if rows[0]["hypercube_2d"] is not None:
            hc = {
                "num_modes": rows[0]["hypercube_2d"]["num_modes"],
                "distinct_modes_sampled_per_seed": [row["hypercube_2d"]["distinct_modes_sampled"] for row in rows],
                "routes": agg_group(lambda row: row["hypercube_2d"]["routes"]),
            }
            hc["comparisons"] = {
                "factored_minus_uncompensated": paired_stats(
                    hc["routes"]["bapr_factored"]["switching_seed_means"],
                    hc["routes"]["uncompensated"]["switching_seed_means"],
                )
            }

        summary["environments"][env_key] = {
            "gym_id": rows[0]["gym_id"],
            "primary_4mode": prim,
            "comparisons": {
                **{
                    f"bapr_selfcal_minus_{comp}": paired_stats(
                        prim["bapr_selfcal"]["switching_seed_means"],
                        prim[comp]["switching_seed_means"],
                    )
                    for comp in (
                        "sac", "escp256", "rma", "belief_sac", "uncompensated",
                        "bapr_instantaneous", "bapr_d100_lockin", "bapr_forward_shared",
                        "bapr_noburst_selfcal",
                    )
                },
                "oracle_burst_minus_oracle_noburst": paired_stats(
                    prim["oracle"]["switching_seed_means"],
                    prim["bapr_noburst_oracle"]["switching_seed_means"],
                ),
            },
            "hypercube_2d": hc,
            "lossy_projection": {
                "routes": lossy,
                "decomposition": {
                    "infeasibility_gap_proj_oracle_minus_native": paired_stats(
                        lossy["oracle"]["switching_seed_means"],
                        lossy["native_reference"]["switching_seed_means"],
                    ),
                    "inference_loss_bapr_minus_proj_oracle": paired_stats(
                        lossy["bapr_selfcal"]["switching_seed_means"],
                        lossy["oracle"]["switching_seed_means"],
                    ),
                    "bapr_projected_minus_uncompensated": paired_stats(
                        lossy["bapr_selfcal"]["switching_seed_means"],
                        lossy["uncompensated"]["switching_seed_means"],
                    ),
                },
            },
        }
    write_json(RELEASE_DIR / "analysis.json", summary)
    print(f"Wrote {RELEASE_DIR / 'analysis.json'}", flush=True)
    return summary


def main() -> None:
    global BUNDLE_ROOT, RELEASE_DIR, RUN_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--env", choices=ENV_NAMES, default="halfcheetah")
    parser.add_argument("--arm", choices=ARMS, default="bapr")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--episode-dir", type=Path, help="Replay the frozen audit and export compact episode diagnostics.")
    args = parser.parse_args()

    if args.mode == "smoke":
        orig_bundle = BUNDLE_ROOT
        smoke_root = RUN_ROOT / "smoke_sandbox"
        BUNDLE_ROOT = smoke_root / "bundles"
        RELEASE_DIR = smoke_root / "release"
        RUN_ROOT = smoke_root / "runs"
        for arm in ARMS:
            train_arm(args.env, arm, args.seed, smoke=True)
        audit_seed(args.env, args.seed, smoke=True)
        write_json(orig_bundle / "smoke/ready.json", {"ok": True, "protocol": PROTOCOL, "runtime_versions": runtime_info(strict=False)})
    elif args.mode == "train":
        train_arm(args.env, args.arm, args.seed, smoke=False)
    elif args.mode == "audit":
        audit_seed(args.env, args.seed, smoke=False, episode_dir=args.episode_dir)
    else:
        analyze_all()


if __name__ == "__main__":
    main()
