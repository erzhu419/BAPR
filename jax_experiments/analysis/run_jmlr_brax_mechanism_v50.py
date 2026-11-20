"""JMLR Brax mechanism audit: unified delays, delta/lock-in sweep, M=64 factored filter, and lossy gain projection."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.stats import t as student_t

from jax_experiments.analysis import run_mechanism_gap_v34 as common

jax.config.update("jax_compilation_cache_dir", "/tmp/bapr_jax_cache_v50")

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "jmlr-brax-mechanism-v50"
RUNTIME = {"jax": "0.4.38", "brax": "0.10.5", "flax": "0.10.4", "mujoco": "3.1.6"}
SEEDS = (88003, 88021, 88039, 88057, 88079, 88103, 88121, 88139, 88157, 88179)
V32_SWITCH_SEEDS = (197201, 197217, 197233)
V32_BASE_SCHEDULES = {
    197201: (2, 0, 3, 1),
    197217: (1, 3, 0, 2),
    197233: (3, 2, 1, 0),
}
V32_STATIONARY_SEED = 197101

OUT_DIR = ROOT / "reproducibility/jmlr_brax_mechanism_v50"

# All 64 hypercube polarity modes in {-1, +1}^6, ordered so mode 0 is (-1,-1,-1,+1,+1,+1)
_REF_SIGN = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
_ALL_SIGNS = [tuple(float(x) for x in row) for row in itertools.product((-1.0, 1.0), repeat=6)]
_ALL_SIGNS.remove(_REF_SIGN)
GAINS_64 = np.asarray([_REF_SIGN, *_ALL_SIGNS], dtype=np.float32)

# Four persistent lossy actuator-degradation modes (50% motor attenuation + polarity flips)
GAINS_LOSSY = np.asarray([
    [-1.0, -1.0, -1.0,  1.0,  1.0,  1.0],
    [-0.5, -0.5, -0.5, -1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0,  0.5, -0.5,  0.5],
    [ 1.0, -0.5,  0.5, -1.0,  0.5, -0.5],
], dtype=np.float32)

# Dynamic mode_kind codes so ONE JIT kernel handles all filter/ablation settings for a given n_modes!
KIND_NATIVE = 0
KIND_ORACLE = 1
KIND_UNCOMP = 2
KIND_CAUSAL = 3


def check_runtime() -> dict[str, str]:
    actual = {name: version(name) for name in RUNTIME}
    if actual != RUNTIME:
        raise ValueError(f"runtime mismatch: expected {RUNTIME}, got {actual}")
    return actual


def emit_progress(current: int, total: int, started: float, unit: str = "phase") -> None:
    print("SCHEDULEURM_PROGRESS " + json.dumps({
        "current": current,
        "total": total,
        "unit": unit,
        "elapsed_seconds": time.monotonic() - started,
    }), flush=True)


def pad_gains_64(gains_arr) -> jax.Array:
    g = np.asarray(gains_arr, dtype=np.float32)
    out = np.repeat(g[:1], 64, axis=0)
    out[:g.shape[0]] = g
    return jnp.asarray(out, dtype=jnp.float32)


def build_batched_joint_runner(world, actor_graph, model_graph, model_params, variance: jax.Array):
    """Single vmapped JIT kernel for ALL mode cardinalities (M <= 64) and filter ablations!"""
    mode_idx = jnp.arange(64, dtype=jnp.int32)
    reset = world.reset
    simulator_step = world.step
    sys = world.sys
    horizon, dwell = world.horizon, world.dwell

    def single_episode(actor_params, gains_padded, active_modes, key, schedule, kind_code, hazard, scale, decay, temporal_flag, single_head_flag):
        reset_key, rollout_key = jax.random.split(key)
        initial = reset(sys, reset_key)
        ref_gain = gains_padded[0]
        active_mask = mode_idx < active_modes
        active_f = active_modes.astype(jnp.float32)
        uniform = jnp.where(active_mask, 1.0 / active_f, 0.0)

        def body(carry, inputs):
            state, belief = carry
            step, noise_key = inputs
            true_mode = schedule[(step // dwell) % schedule.shape[0]]
            est_mode = jnp.argmax(jnp.where(active_mask, belief, -1.0))
            q = nnx.merge(actor_graph, actor_params).deterministic(state.obs[None])[0]
            ref_target = ref_gain * q

            phys_mode = jnp.where(kind_code == KIND_NATIVE, jnp.asarray(0, jnp.int32), true_mode)
            oracle_cmd = jnp.clip(ref_target / gains_padded[true_mode], -1.0, 1.0)
            causal_cmd = jnp.clip(ref_target / gains_padded[est_mode], -1.0, 1.0)
            command = jnp.where(
                (kind_code == KIND_NATIVE) | (kind_code == KIND_UNCOMP),
                q,
                jnp.where(kind_code == KIND_ORACLE, oracle_cmd, causal_cmd),
            )

            executed = common.apply_action_disturbance(command, noise_key, gains_padded[phys_mode], 0.02)
            following = simulator_step(state, executed)

            pred5 = nnx.merge(model_graph, model_params).predict(state.obs[None], following.obs[None])
            ev5 = common.candidate_action_evidence(pred5, command[None], gains_padded, variance)[0][0]
            ev1 = common.candidate_action_evidence(pred5[:1], command[None], gains_padded, variance)[0][0]
            ev = jnp.where(single_head_flag, ev1, ev5)

            prior_raw = jnp.where(
                hazard > 0.0,
                (1.0 - hazard) * belief + (hazard / jnp.maximum(active_f - 1.0, 1.0)) * (1.0 - belief),
                belief,
            )
            prior = jnp.where(active_mask, prior_raw, 0.0)
            logits_temp = decay * jnp.log(jnp.clip(prior, 1e-30, 1.0)) + scale * ev
            logits_raw = jnp.where(temporal_flag, logits_temp, scale * ev)
            logits = jnp.where(active_mask, logits_raw, -1e30)
            after = jnp.where(kind_code == KIND_CAUSAL, jax.nn.softmax(logits), belief)

            realized_nominal = jnp.clip(gains_padded[phys_mode] * command, -1.0, 1.0)
            sq_err = jnp.sum(jnp.square(realized_nominal - ref_target))
            infeas_sq = jnp.sum(jnp.square(jnp.maximum(0.0, jnp.abs(ref_target) - jnp.abs(gains_padded[phys_mode]))))
            return (following, after), (following.reward, true_mode, est_mode, sq_err, infeas_sq)

        return jax.lax.scan(
            body,
            (initial, uniform),
            (jnp.arange(horizon), jax.random.split(rollout_key, horizon)),
        )[1]

    @jax.jit
    def run_batch(actor_params, gains_padded, active_modes, keys, schedules, kind_code, hazard, scale, decay, temporal_flag, single_head_flag):
        return jax.vmap(
            single_episode,
            in_axes=(None, None, None, 0, 0, None, None, None, None, None, None),
        )(actor_params, gains_padded, active_modes, keys, schedules, kind_code, hazard, scale, decay, temporal_flag, single_head_flag)

    return run_batch


def build_batched_factored_runner(world, actor_graph, model_graph, model_params, variance: jax.Array, act_dim: int = 6):
    """Single vmapped JIT kernel for O(d) channel-factored binary polarity filtering (works for ANY M!)."""
    var_clipped = jnp.clip(variance, 1e-6, 1.0)
    uniform_pos = jnp.full(act_dim, 0.5, dtype=jnp.float32)
    reset = world.reset
    simulator_step = world.step
    sys = world.sys
    horizon, dwell = world.horizon, world.dwell

    def single_episode(actor_params, gains, key, schedule, hazard, scale, decay):
        reset_key, rollout_key = jax.random.split(key)
        initial = reset(sys, reset_key)
        ref_gain = gains[0]

        def body(carry, inputs):
            state, prob_pos = carry
            step, noise_key = inputs
            true_mode = schedule[(step // dwell) % schedule.shape[0]]
            est_gain = jnp.where(prob_pos >= 0.5, 1.0, -1.0)
            q = nnx.merge(actor_graph, actor_params).deterministic(state.obs[None])[0]
            ref_target = ref_gain * q
            command = est_gain * ref_target

            executed = common.apply_action_disturbance(command, noise_key, gains[true_mode], 0.02)
            following = simulator_step(state, executed)

            pred = nnx.merge(model_graph, model_params).predict(state.obs[None], following.obs[None])[:, 0, :]
            cand_pos = jnp.clip(command, -1.0, 1.0)[None, :]
            cand_neg = jnp.clip(-command, -1.0, 1.0)[None, :]
            ll_pos = jax.scipy.special.logsumexp(
                -0.5 * jnp.square(cand_pos - pred) / var_clipped[None, :], axis=0
            ) - jnp.log(float(pred.shape[0]))
            ll_neg = jax.scipy.special.logsumexp(
                -0.5 * jnp.square(cand_neg - pred) / var_clipped[None, :], axis=0
            ) - jnp.log(float(pred.shape[0]))

            prior_pos = (1.0 - hazard) * prob_pos + hazard * (1.0 - prob_pos)
            prior_neg = 1.0 - prior_pos
            log_odds = decay * (
                jnp.log(jnp.clip(prior_pos, 1e-30, 1.0)) - jnp.log(jnp.clip(prior_neg, 1e-30, 1.0))
            ) + scale * (ll_pos - ll_neg)
            after_pos = jax.nn.sigmoid(log_odds)

            exact_match = jnp.all(est_gain == gains[true_mode])
            channel_acc = jnp.mean((est_gain == gains[true_mode]).astype(jnp.float32))
            sq_err = jnp.sum(jnp.square(gains[true_mode] * command - ref_target))
            return (following, after_pos), (following.reward, exact_match, channel_acc, sq_err)

        return jax.lax.scan(
            body,
            (initial, uniform_pos),
            (jnp.arange(horizon), jax.random.split(rollout_key, horizon)),
        )[1]

    @jax.jit
    def run_batch(actor_params, gains, keys, schedules, hazard, scale, decay):
        return jax.vmap(single_episode, in_axes=(None, None, 0, 0, None, None, None))(
            actor_params, gains, keys, schedules, hazard, scale, decay
        )

    return run_batch


def summarize_joint_batch(batch_vals, oracle_rewards_batch=None, causal: bool = True) -> list[dict]:
    rewards_b, true_b, est_b, sq_b, inf_b = map(np.asarray, batch_vals)
    rows = []
    for ep_i in range(rewards_b.shape[0]):
        rewards = rewards_b[ep_i]
        true_modes = true_b[ep_i]
        est_modes = est_b[ep_i]
        switched = np.flatnonzero(true_modes[1:] != true_modes[:-1]) + 1
        d1_list, d8_list = [], []
        u1 = u8 = 0
        if causal:
            for idx, start in enumerate(switched):
                end = int(switched[idx + 1]) if idx + 1 < len(switched) else len(true_modes)
                tgt = true_modes[start]
                d1 = next((s - int(start) for s in range(int(start), end) if est_modes[s] == tgt), None)
                if d1 is None:
                    u1 += 1
                else:
                    d1_list.append(int(d1))
                d8 = next((s - int(start) for s in range(int(start), end - 7) if np.all(est_modes[s:s + 8] == tgt)), None)
                if d8 is None:
                    u8 += 1
                else:
                    d8_list.append(int(d8))
        reg32 = None
        if oracle_rewards_batch is not None:
            mask = np.zeros(len(rewards), dtype=bool)
            for start in switched:
                mask[start:min(int(start) + 32, len(rewards))] = True
            reg32 = float(oracle_rewards_batch[ep_i][mask].sum() - rewards[mask].sum())
        rows.append({
            "return": float(rewards.sum()),
            "mode_accuracy": float(np.mean(est_modes == true_modes)) if causal else None,
            "action_error_energy": float(sq_b[ep_i].sum()),
            "infeasibility_energy": float(inf_b[ep_i].sum()),
            "first_delays": d1_list,
            "unresolved_first": u1,
            "sustained_delays": d8_list,
            "unresolved_sustained": u8,
            "switches": len(switched),
            "post_switch_regret_32": reg32,
        })
    return rows


def summarize_factored_batch(batch_vals, oracle_rewards_batch, dwell: int) -> list[dict]:
    rewards_b, exact_b, ch_b, sq_b = map(np.asarray, batch_vals)
    rows = []
    for ep_i in range(rewards_b.shape[0]):
        rewards = rewards_b[ep_i]
        exact = exact_b[ep_i]
        H = len(rewards)
        switched = [dwell, 2 * dwell, 3 * dwell]
        d1_list, d8_list = [], []
        u1 = u8 = 0
        for idx, start in enumerate(switched):
            end = switched[idx + 1] if idx + 1 < len(switched) else H
            d1 = next((s - start for s in range(start, end) if exact[s]), None)
            if d1 is None:
                u1 += 1
            else:
                d1_list.append(int(d1))
            d8 = next((s - start for s in range(start, max(start, end - 7)) if np.all(exact[s:s + 8])), None)
            if d8 is None:
                u8 += 1
            else:
                d8_list.append(int(d8))
        mask = np.zeros(H, dtype=bool)
        for start in switched:
            mask[start:min(start + 32, H)] = True
        reg32 = float(oracle_rewards_batch[ep_i][mask].sum() - rewards[mask].sum())
        rows.append({
            "return": float(rewards.sum()),
            "mode_accuracy": float(np.mean(exact)),
            "channel_accuracy": float(np.mean(ch_b[ep_i])),
            "action_error_energy": float(sq_b[ep_i].sum()),
            "infeasibility_energy": 0.0,
            "first_delays": d1_list,
            "unresolved_first": u1,
            "sustained_delays": d8_list,
            "unresolved_sustained": u8,
            "switches": len(switched),
            "post_switch_regret_32": reg32,
        })
    return rows


def schedules_64(smoke: bool = False) -> list[tuple[int, np.ndarray]]:
    rng = np.random.default_rng(20265064)
    out = []
    n_ep = 4 if smoke else 15
    split_1j = 2 if smoke else 9
    for ep in range(n_ep):
        if ep < split_1j:
            seq = rng.choice(64, size=4, replace=False).astype(np.int32)
        else:
            cur = GAINS_64[int(rng.integers(0, 64))].copy()
            seq_list = []
            used_joints = rng.choice(6, size=4, replace=False)
            for b in range(4):
                if b > 0:
                    cur[used_joints[b]] *= -1.0
                idx = int(np.where(np.all(GAINS_64 == cur[None, :], axis=1))[0][0])
                seq_list.append(idx)
            seq = np.asarray(seq_list, dtype=np.int32)
        out.append((264000 + ep, seq))
    return out


def reduce_arm(rows: list[dict], stat_returns: list[float] | None = None) -> dict:
    d1 = [d for r in rows for d in r["first_delays"]]
    d8 = [d for r in rows for d in r["sustained_delays"]]
    acc = [r["mode_accuracy"] for r in rows if r.get("mode_accuracy") is not None]
    ch_acc = [r["channel_accuracy"] for r in rows if r.get("channel_accuracy") is not None]
    reg = [r["post_switch_regret_32"] for r in rows if r.get("post_switch_regret_32") is not None]
    return {
        "switching_mean": statistics.fmean(r["return"] for r in rows),
        "stationary_mean": statistics.fmean(stat_returns) if stat_returns else None,
        "mode_accuracy": statistics.fmean(acc) if acc else None,
        "channel_accuracy": statistics.fmean(ch_acc) if ch_acc else None,
        "action_error_energy": statistics.fmean(r["action_error_energy"] for r in rows),
        "infeasibility_energy": statistics.fmean(r.get("infeasibility_energy", 0.0) for r in rows),
        "first_delay_median": float(np.median(d1)) if d1 else None,
        "unresolved_first": sum(r["unresolved_first"] for r in rows),
        "sustained_delay_median": float(np.median(d8)) if d8 else None,
        "unresolved_sustained": sum(r["unresolved_sustained"] for r in rows),
        "total_switches": sum(r["switches"] for r in rows),
        "post_switch_regret_32": statistics.fmean(reg) if reg else None,
    }


def run_seed(seed: int, smoke: bool = False) -> dict:
    started = time.monotonic()
    runtime = check_runtime()
    out_path = OUT_DIR / "seeds" / f"seed_{seed}.json"
    if not smoke and out_path.is_file():
        print(f"Seed {seed} already completed: {out_path}", flush=True)
        return json.loads(out_path.read_text())

    world = common.World("hc", seed)
    if smoke:
        world.horizon, world.dwell = 64, 16
    actor_graph, actor_params = common.load_frozen_actor(seed)
    inv_model, inv_params, variance, _ = common.frozen_inverse()
    model_graph = nnx.graphdef(inv_model)

    # Compile ONLY TWO fixed-shape vmapped kernels total!
    runner_joint = build_batched_joint_runner(world, actor_graph, model_graph, inv_params, variance)
    runner_factored = build_batched_factored_runner(world, actor_graph, model_graph, inv_params, variance, act_dim=6)
    batch_fixed = 4 if smoke else 35

    def _pad_inputs(keys_arr: jax.Array, scheds_arr: jax.Array):
        n = int(keys_arr.shape[0])
        if n == batch_fixed:
            return keys_arr, scheds_arr, n
        pad_n = batch_fixed - n
        k_pad = jnp.concatenate([keys_arr, jnp.repeat(keys_arr[:1], pad_n, axis=0)], axis=0)
        s_pad = jnp.concatenate([scheds_arr, jnp.repeat(scheds_arr[:1], pad_n, axis=0)], axis=0)
        return k_pad, s_pad, n

    def call_joint(gains_padded: jax.Array, active_m: int, keys_arr: jax.Array, scheds_arr: jax.Array, kind_c: int, h: float, sc: float, dec: float, temp: bool, sh: bool):
        k_in, s_in, n = _pad_inputs(keys_arr, scheds_arr)
        out = runner_joint(
            actor_params,
            gains_padded,
            jnp.int32(active_m),
            k_in,
            s_in,
            jnp.int32(kind_c),
            jnp.float32(h),
            jnp.float32(sc),
            jnp.float32(dec),
            jnp.bool_(temp),
            jnp.bool_(sh),
        )
        return tuple(np.asarray(x)[:n] for x in out)

    def call_factored(gains_padded: jax.Array, keys_arr: jax.Array, scheds_arr: jax.Array, h: float, sc: float, dec: float):
        k_in, s_in, n = _pad_inputs(keys_arr, scheds_arr)
        out = runner_factored(
            actor_params,
            gains_padded,
            k_in,
            s_in,
            jnp.float32(h),
            jnp.float32(sc),
            jnp.float32(dec),
        )
        return tuple(np.asarray(x)[:n] for x in out)

    emit_progress(0, 3, started)

    # --- Part 1: 4-mode switching & stationary ablations ---
    gains_4_p = pad_gains_64(world.gains)
    ev_seeds = V32_SWITCH_SEEDS[:1] if smoke else V32_SWITCH_SEEDS
    ep_per_ev = 2 if smoke else 5
    sw_keys_list, sw_scheds_list = [], []
    for ev_seed in ev_seeds:
        base = np.asarray(V32_BASE_SCHEDULES[ev_seed], dtype=np.int32)
        for ep_idx in range(ep_per_ev):
            sw_scheds_list.append(np.roll(base, ep_idx % 4))
            sw_keys_list.append(jax.random.PRNGKey(ev_seed * 100 + ep_idx))
    sw_keys = jnp.stack(sw_keys_list)
    sw_scheds = jnp.asarray(sw_scheds_list, dtype=jnp.int32)
    n_sw = int(sw_keys.shape[0])

    st_keys_list, st_scheds_list = [], []
    for mode in range(1 if smoke else 4):
        for ep_idx in range(1 if smoke else 5):
            st_scheds_list.append(np.full(4, mode, dtype=np.int32))
            st_keys_list.append(jax.random.PRNGKey(V32_STATIONARY_SEED * 100 + mode * 5 + ep_idx))
    st_keys = jnp.stack(st_keys_list)
    st_scheds = jnp.asarray(st_scheds_list, dtype=jnp.int32)
    comb_keys = jnp.concatenate([sw_keys, st_keys], axis=0)
    comb_scheds = jnp.concatenate([sw_scheds, st_scheds], axis=0)

    ablation_cfgs = {
        "oracle": (KIND_ORACLE, 0.002, 1.0, 0.98, True, False, False),
        "native_reference": (KIND_NATIVE, 0.002, 1.0, 0.98, True, False, False),
        "uncompensated": (KIND_UNCOMP, 0.002, 1.0, 0.98, True, False, False),
        "bapr_d098": (KIND_CAUSAL, 0.002, 1.0, 0.98, True, False, True),
        "sticky_hmm_d100": (KIND_CAUSAL, 0.002, 1.0, 1.00, True, False, True),
        "untempered_lockin_h0_d100": (KIND_CAUSAL, 0.0, 1.0, 1.00, True, False, True),
        "tempered_h0_d098": (KIND_CAUSAL, 0.0, 1.0, 0.98, True, False, True),
        "tempered_d095": (KIND_CAUSAL, 0.002, 1.0, 0.95, True, False, True),
        "tempered_d090": (KIND_CAUSAL, 0.002, 1.0, 0.90, True, False, True),
        "instantaneous": (KIND_CAUSAL, 0.002, 1.0, 0.98, False, False, True),
        "single_head_d098": (KIND_CAUSAL, 0.002, 1.0, 0.98, True, True, True),
    }

    oracle_comb_4 = call_joint(gains_4_p, 4, comb_keys, comb_scheds, KIND_ORACLE, 0.002, 1.0, 0.98, True, False)
    oracle_rew_4 = oracle_comb_4[0][:n_sw]

    part1_summary = {}
    stat_arms = {"oracle", "bapr_d098", "sticky_hmm_d100", "untempered_lockin_h0_d100", "instantaneous", "single_head_d098"}
    for name, (kind_c, h, sc, dec, temp, sh, causal) in ablation_cfgs.items():
        if name == "oracle":
            b_all = oracle_comb_4
            b_sw = tuple(x[:n_sw] for x in b_all)
            st_rets = [float(x) for x in b_all[0][n_sw:].sum(axis=1)]
        elif name in stat_arms:
            b_all = call_joint(gains_4_p, 4, comb_keys, comb_scheds, kind_c, h, sc, dec, temp, sh)
            b_sw = tuple(x[:n_sw] for x in b_all)
            st_rets = [float(x) for x in b_all[0][n_sw:].sum(axis=1)]
        else:
            b_sw = call_joint(gains_4_p, 4, sw_keys, sw_scheds, kind_c, h, sc, dec, temp, sh)
            st_rets = None
        rows = summarize_joint_batch(b_sw, None if name == "oracle" else oracle_rew_4, causal=causal)
        part1_summary[name] = reduce_arm(rows, st_rets)

    f4_all = call_factored(gains_4_p, comb_keys, comb_scheds, 0.002, 0.5, 0.98)
    part1_summary["factored_d098"] = reduce_arm(
        summarize_factored_batch(tuple(x[:n_sw] for x in f4_all), oracle_rew_4, world.dwell),
        [float(x) for x in f4_all[0][n_sw:].sum(axis=1)],
    )

    emit_progress(1, 3, started)

    # --- Part 2: M = 64 Full Polarity Hypercube ---
    gains_64_j = jnp.asarray(GAINS_64, dtype=jnp.float32)
    s64 = schedules_64(smoke=smoke)
    split_1j = 2 if smoke else 9
    k64 = jnp.stack([jax.random.PRNGKey(ep_s + seed % 97) for ep_s, _ in s64])
    sch64 = jnp.asarray([seq for _, seq in s64], dtype=jnp.int32)

    o64_vals = call_joint(gains_64_j, 64, k64, sch64, KIND_ORACLE, 0.002, 1.0, 0.98, True, False)
    o64_rew = o64_vals[0]
    u64_vals = call_joint(gains_64_j, 64, k64, sch64, KIND_UNCOMP, 0.002, 1.0, 0.98, True, False)
    j64_vals = call_joint(gains_64_j, 64, k64, sch64, KIND_CAUSAL, 0.002, 1.0, 0.98, True, False)
    f64_vals = call_factored(gains_64_j, k64, sch64, 0.002, 0.5, 0.98)

    rows_64 = {
        "oracle_64": summarize_joint_batch(o64_vals, None, causal=False),
        "uncompensated_64": summarize_joint_batch(u64_vals, o64_rew, causal=False),
        "joint_64": summarize_joint_batch(j64_vals, o64_rew, causal=True),
        "factored_64": summarize_factored_batch(f64_vals, o64_rew, world.dwell),
    }

    emit_progress(2, 3, started)

    # --- Part 3: Non-bijective lossy actuator gain attenuation (0.5x power loss + sign flips) ---
    gains_lossy_p = pad_gains_64(GAINS_LOSSY)
    klossy = jnp.stack([
        jax.random.PRNGKey((ev_seed + 50000) * 100 + ep_idx)
        for ev_seed in ev_seeds for ep_idx in range(ep_per_ev)
    ])
    proj_vals = call_joint(gains_lossy_p, 4, klossy, sw_scheds, KIND_ORACLE, 0.002, 1.0, 0.98, True, False)
    proj_rew = proj_vals[0]
    nat_vals = call_joint(gains_lossy_p, 4, klossy, sw_scheds, KIND_NATIVE, 0.002, 1.0, 0.98, True, False)
    caus_vals = call_joint(gains_lossy_p, 4, klossy, sw_scheds, KIND_CAUSAL, 0.002, 1.0, 0.98, True, False)
    unc_vals = call_joint(gains_lossy_p, 4, klossy, sw_scheds, KIND_UNCOMP, 0.002, 1.0, 0.98, True, False)

    part3_summary = {
        "native_reference": reduce_arm(summarize_joint_batch(nat_vals, proj_rew, causal=False)),
        "projected_oracle": reduce_arm(summarize_joint_batch(proj_vals, None, causal=False)),
        "causal_projected_bapr": reduce_arm(summarize_joint_batch(caus_vals, proj_rew, causal=True)),
        "uncompensated": reduce_arm(summarize_joint_batch(unc_vals, proj_rew, causal=False)),
    }

    payload = {
        "protocol": PROTOCOL,
        "seed": seed,
        "runtime_versions": runtime,
        "part1_primary_ablations": part1_summary,
        "part2_hypercube_64": {
            "all_15_episodes": {k: reduce_arm(v) for k, v in rows_64.items()},
            "single_joint_flip_episodes": {k: reduce_arm(v[split_1j:]) for k, v in rows_64.items()},
        },
        "part3_lossy_projection": part3_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    emit_progress(3, 3, started)
    if not smoke:
        common.write_json(out_path, payload)
    print(f"Completed {PROTOCOL} seed={seed} in {payload['elapsed_seconds']:.1f}s", flush=True)
    return payload


def paired_ci(xs: list[float], ys: list[float]) -> dict:
    if len(xs) != len(ys):
        raise ValueError(f"paired_ci length mismatch: {len(xs)} != {len(ys)}")
    diffs = [a - b for a, b in zip(xs, ys)]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {
        "n_seeds": n,
        "mean": mean,
        "ci95": [mean - half, mean + half],
        "wins": sum(d > 0 for d in diffs),
    }


def analyze() -> dict:
    seed_rows = [json.loads((OUT_DIR / "seeds" / f"seed_{s}.json").read_text()) for s in SEEDS]
    distinct_modes_64 = sorted({int(m) for _, seq in schedules_64(smoke=False) for m in seq})

    def agg_section(section_getter):
        arms = list(section_getter(seed_rows[0]).keys())
        summary = {}
        for arm in arms:
            per_seed = [section_getter(r)[arm] for r in seed_rows]
            stat_vals = [x["stationary_mean"] for x in per_seed if x["stationary_mean"] is not None]
            acc_vals = [x["mode_accuracy"] for x in per_seed if x["mode_accuracy"] is not None]
            ch_vals = [x["channel_accuracy"] for x in per_seed if x["channel_accuracy"] is not None]
            d1_vals = [x["first_delay_median"] for x in per_seed if x["first_delay_median"] is not None]
            d8_vals = [x["sustained_delay_median"] for x in per_seed if x["sustained_delay_median"] is not None]
            reg_vals = [x["post_switch_regret_32"] for x in per_seed if x["post_switch_regret_32"] is not None]
            summary[arm] = {
                "switching_mean": statistics.fmean(x["switching_mean"] for x in per_seed),
                "switching_seed_means": [x["switching_mean"] for x in per_seed],
                "stationary_mean": statistics.fmean(stat_vals) if stat_vals else None,
                "mode_accuracy": statistics.fmean(acc_vals) if acc_vals else None,
                "channel_accuracy": statistics.fmean(ch_vals) if ch_vals else None,
                "action_error_energy": statistics.fmean(x["action_error_energy"] for x in per_seed),
                "infeasibility_energy": statistics.fmean(x["infeasibility_energy"] for x in per_seed),
                "first_delay_median_of_medians": float(np.median(d1_vals)) if d1_vals else None,
                "unresolved_first_total": sum(x["unresolved_first"] for x in per_seed),
                "sustained_delay_median_of_medians": float(np.median(d8_vals)) if d8_vals else None,
                "unresolved_sustained_total": sum(x["unresolved_sustained"] for x in per_seed),
                "total_switches": sum(x["total_switches"] for x in per_seed),
                "post_switch_regret_32": statistics.fmean(reg_vals) if reg_vals else None,
            }
        return summary

    p1 = agg_section(lambda r: r["part1_primary_ablations"])
    p2_all = agg_section(lambda r: r["part2_hypercube_64"]["all_15_episodes"])
    p2_1j = agg_section(lambda r: r["part2_hypercube_64"]["single_joint_flip_episodes"])
    p3 = agg_section(lambda r: r["part3_lossy_projection"])

    result = {
        "protocol": PROTOCOL,
        "runtime_versions": seed_rows[0]["runtime_versions"],
        "policy_seeds": list(SEEDS),
        "part1_primary_ablations": p1,
        "part1_paired_comparisons": {
            "bapr_d098_minus_instantaneous": paired_ci(p1["bapr_d098"]["switching_seed_means"], p1["instantaneous"]["switching_seed_means"]),
            "bapr_d098_minus_untempered_lockin_h0_d100": paired_ci(p1["bapr_d098"]["switching_seed_means"], p1["untempered_lockin_h0_d100"]["switching_seed_means"]),
            "tempered_h0_d098_minus_untempered_lockin_h0_d100": paired_ci(p1["tempered_h0_d098"]["switching_seed_means"], p1["untempered_lockin_h0_d100"]["switching_seed_means"]),
            "bapr_d098_minus_sticky_hmm_d100": paired_ci(p1["bapr_d098"]["switching_seed_means"], p1["sticky_hmm_d100"]["switching_seed_means"]),
            "bapr_d098_minus_single_head_d098": paired_ci(p1["bapr_d098"]["switching_seed_means"], p1["single_head_d098"]["switching_seed_means"]),
            "bapr_d098_minus_factored_d098": paired_ci(p1["bapr_d098"]["switching_seed_means"], p1["factored_d098"]["switching_seed_means"]),
        },
        "part2_hypercube_64": {
            "candidate_modes_scored_online": 64,
            "distinct_modes_sampled": len(distinct_modes_64),
            "all_15_episodes": p2_all,
            "single_joint_flip_episodes": p2_1j,
            "paired_comparisons": {
                "factored_64_minus_uncompensated_64": paired_ci(p2_all["factored_64"]["switching_seed_means"], p2_all["uncompensated_64"]["switching_seed_means"]),
                "factored_64_minus_joint_64": paired_ci(p2_all["factored_64"]["switching_seed_means"], p2_all["joint_64"]["switching_seed_means"]),
                "single_joint_factored_64_minus_uncompensated_64": paired_ci(p2_1j["factored_64"]["switching_seed_means"], p2_1j["uncompensated_64"]["switching_seed_means"]),
                "single_joint_factored_64_minus_joint_64": paired_ci(p2_1j["factored_64"]["switching_seed_means"], p2_1j["joint_64"]["switching_seed_means"]),
            },
        },
        "part3_lossy_projection": {
            "arms": p3,
            "two_term_decomposition": {
                "infeasibility_gap_projected_minus_native": paired_ci(p3["projected_oracle"]["switching_seed_means"], p3["native_reference"]["switching_seed_means"]),
                "inference_loss_causal_minus_projected": paired_ci(p3["causal_projected_bapr"]["switching_seed_means"], p3["projected_oracle"]["switching_seed_means"]),
                "total_gap_causal_minus_native": paired_ci(p3["causal_projected_bapr"]["switching_seed_means"], p3["native_reference"]["switching_seed_means"]),
                "causal_projected_minus_uncompensated": paired_ci(p3["causal_projected_bapr"]["switching_seed_means"], p3["uncompensated"]["switching_seed_means"]),
            },
        },
    }
    common.write_json(OUT_DIR / "analysis.json", result)
    print(f"Wrote {OUT_DIR / 'analysis.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "seed", "analyze"))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    args = parser.parse_args()
    if args.mode == "smoke":
        res = run_seed(args.seed, smoke=True)
        common.write_json(OUT_DIR / "smoke/ready.json", {"ok": True, "runtime_versions": res["runtime_versions"]})
    elif args.mode == "seed":
        run_seed(args.seed, smoke=False)
    else:
        analyze()


if __name__ == "__main__":
    main()
