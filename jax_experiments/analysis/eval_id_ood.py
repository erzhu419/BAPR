"""ID/OOD evaluation: load trained checkpoints and evaluate on ID vs OOD tasks.

Evaluates trained agents on:
  - In-Distribution (ID): tasks sampled from the same range as training
  - Out-of-Distribution (OOD): tasks sampled from a wider range (ood_scale × training)

Usage:
    python -m jax_experiments.analysis.eval_id_ood \
        --results_root jax_experiments/results \
        --envs HalfCheetah-v2 Hopper-v2 Walker2d-v2 Ant-v2 \
        --algos bapr escp resac \
        --output paper/fig_id_ood.pdf
"""
import argparse
import json
import multiprocessing as mp
import os
import glob
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt

# Ensure CUDA libs available
NVIDIA_LIB = None
for p in sys.path:
    candidate = os.path.join(p, "nvidia")
    if os.path.isdir(candidate):
        NVIDIA_LIB = candidate
        break
if NVIDIA_LIB is None:
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "nvidia")
        if os.path.isdir(candidate):
            NVIDIA_LIB = candidate
            break
if NVIDIA_LIB is not None:
    lib_dirs = []
    for subdir in os.listdir(NVIDIA_LIB):
        lib_path = os.path.join(NVIDIA_LIB, subdir, "lib")
        if os.path.isdir(lib_path):
            lib_dirs.append(lib_path)
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.common.checkpoint import load_checkpoint, has_checkpoint
from jax_experiments.envs.brax_env import BraxNonstationaryEnv as NonstationaryEnv


ALGO_COLORS = {
    "bapr": "#2196F3", "escp": "#FF9800", "resac": "#4CAF50",
    "sac": "#9E9E9E", "bapr_no_bocd": "#E91E63",
    "bapr_fixed_decay": "#FF5722",
}

ALGO_LABELS = {
    "bapr": "BAPR", "escp": "ESCP", "resac": "RE-SAC", "sac": "SAC",
    "bapr_no_bocd": "w/o BOCD", "bapr_fixed_decay": "FixedDecay",
}


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def eval_policy_module(agent):
    return agent.ema_policy if hasattr(agent, 'ema_policy') else agent.policy

# Environment configs (must match training)
ENV_PARAMS = {
    "HalfCheetah-v2": (["gravity"], 0.75),
    "Hopper-v2": (["gravity"], 0.75),
    "Walker2d-v2": (["dof_damping", "body_mass"], 1.5),
    "Ant-v2": (["gravity"], 0.75),
}


def eval_on_tasks(agent, env, tasks, config, n_episodes=5):
    """Evaluate agent on a specific set of tasks."""
    policy_params = nnx.state(eval_policy_module(agent), nnx.Param)
    context_params = None
    if hasattr(agent, 'context_net'):
        context_params = nnx.state(agent.context_net, nnx.Param)
    belief_vec = None
    if hasattr(agent, 'belief_dim') and agent.belief_dim > 0:
        if hasattr(agent, '_build_belief_jax'):
            belief_vec = agent._build_belief_jax()
        else:
            belief_vec = jnp.asarray(agent.belief_tracker.belief,
                                     dtype=jnp.float32)

    all_rewards = []
    for task in tasks:
        env.set_task(task)
        for ep_idx in range(n_episodes):
            rng_key = jax.random.fold_in(jax.random.PRNGKey(42), ep_idx)
            rew_np, done_np = env.eval_rollout(
                policy_params, config.max_episode_steps, rng_key,
                context_params=context_params, belief_vec=belief_vec)

            ep_r = 0.0
            for i in range(config.max_episode_steps):
                ep_r += rew_np[i]
                if done_np[i] > 0.5:
                    break
            all_rewards.append(ep_r)

    return np.array(all_rewards) if all_rewards else np.array([0.0])


def load_agent_from_checkpoint(ckpt_dir, algo, obs_dim, act_dim, config):
    """Construct agent and load checkpoint weights."""
    from jax_experiments.train import make_algo
    from jax_experiments.common.replay_buffer import ReplayBuffer
    from jax_experiments.common.logging import Logger
    import tempfile

    agent = make_algo(algo, obs_dim, act_dim, config)
    # Dummy replay buffer / logger (not used, but required by load_checkpoint)
    rb = ReplayBuffer(obs_dim, act_dim, capacity=1000)
    tmp_log = tempfile.mkdtemp()
    logger = Logger(tmp_log)

    load_checkpoint(ckpt_dir, agent, rb, logger, algo, load_replay_buffer=False)
    return agent


def find_checkpoints(results_root, algo, env, tag_prefix=None):
    """Find all checkpoint directories for algo/env."""
    # Patterns cover legacy algo_env_seed dirs and paper tags such as
    # paper_full_v2_bapr_Ant_dw60_s0/checkpoints.
    env_short = env.replace("-v2", "")
    tag_glob = f"{tag_prefix}*" if tag_prefix else "*"
    patterns = [
        os.path.join(results_root, f"{algo}_{env}_*/checkpoints"),
        os.path.join(results_root, f"{algo}_{env}/checkpoints"),
        os.path.join(results_root, f"{tag_glob}_{algo}_{env_short}_*/checkpoints"),
    ]
    ckpt_dirs = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            if has_checkpoint(d, require_replay_buffer=False):
                ckpt_dirs.append(d)
    return ckpt_dirs


def checkpoint_run_name(ckpt_dir):
    return os.path.basename(os.path.dirname(os.path.normpath(ckpt_dir)))


def infer_seed_from_run_name(run_name):
    match = re.search(r"_s(\d+)(?:$|_)", run_name)
    return int(match.group(1)) if match else None


def expected_checkpoint_dir(results_root, algo, env, tag_prefix, seed,
                            mean_dwell_iters=60):
    env_short = env.replace("-v2", "")
    run_name = f"{tag_prefix}_{algo}_{env_short}_dw{mean_dwell_iters}_s{seed}"
    return os.path.join(results_root, run_name, "checkpoints")


def build_eval_items(results_root, envs, algos, tag_prefix=None, seeds=None,
                     mean_dwell_iters=60, require_checkpoint=True):
    """Build deterministic checkpoint-eval items.

    With --seeds and --tag_prefix this uses the paper run naming convention
    instead of glob discovery, so sharded CPU jobs all agree on item ordering.
    """
    items = []
    for env in envs:
        for algo in algos:
            if seeds is not None:
                if not tag_prefix:
                    raise ValueError("--seeds requires --tag_prefix")
                ckpt_dirs = [
                    expected_checkpoint_dir(
                        results_root, algo, env, tag_prefix, seed,
                        mean_dwell_iters=mean_dwell_iters)
                    for seed in seeds
                ]
            else:
                ckpt_dirs = find_checkpoints(results_root, algo, env, tag_prefix)

            for ckpt_dir in ckpt_dirs:
                if require_checkpoint and not has_checkpoint(ckpt_dir, require_replay_buffer=False):
                    print(f"  Missing checkpoint for {algo}/{env}: {ckpt_dir}, skipping")
                    continue
                run_name = checkpoint_run_name(ckpt_dir)
                items.append({
                    "env": env,
                    "algo": algo,
                    "ckpt_dir": ckpt_dir,
                    "run_name": run_name,
                    "seed": infer_seed_from_run_name(run_name),
                })
    return items


def make_eval_config(env_name, algo, run_name):
    varying_params, log_scale = ENV_PARAMS.get(env_name, (["gravity"], 0.75))
    config = Config()
    config.env_name = env_name
    config.algo = algo
    config.varying_params = varying_params
    config.log_scale_limit = log_scale
    config.brax_backend = "spring"
    config.env_type = "discrete_mode"
    config.use_per_transition_belief = True
    config.penalty_scale = 0.5
    config.critic_target_mode = "independent"
    config.use_regime_belief = algo == "bapr" and "legacy" not in run_name
    return config, varying_params, log_scale


def evaluate_checkpoint_item(item, ood_scale=1.5, n_id_tasks=20,
                             n_ood_tasks=20, n_episodes=3):
    env_name = item["env"]
    algo = item["algo"]
    ckpt_dir = item["ckpt_dir"]
    run_name = item.get("run_name") or checkpoint_run_name(ckpt_dir)

    config, varying_params, log_scale = make_eval_config(env_name, algo, run_name)
    env = NonstationaryEnv(env_name, rand_params=varying_params,
                           log_scale_limit=log_scale, seed=9999,
                           backend="spring")
    try:
        obs_dim = env.obs_dim
        act_dim = env.act_dim

        id_tasks = env.sample_tasks(n_id_tasks)
        orig_limit = env.log_scale_limit
        env.log_scale_limit = orig_limit * ood_scale
        ood_tasks = env.sample_tasks(n_ood_tasks)
        env.log_scale_limit = orig_limit

        safe_print(f"  Loading {ckpt_dir} ...", flush=True)
        agent = load_agent_from_checkpoint(ckpt_dir, algo, obs_dim, act_dim, config)

        policy_graphdef = nnx.graphdef(eval_policy_module(agent))
        context_graphdef = None
        if hasattr(agent, 'context_net'):
            context_graphdef = nnx.graphdef(agent.context_net)
        env.build_rollout_fn(policy_graphdef, context_graphdef)

        id_rews = eval_on_tasks(agent, env, id_tasks, config, n_episodes)
        ood_rews = eval_on_tasks(agent, env, ood_tasks, config, n_episodes)
        return {
            **item,
            "id_mean": float(np.mean(id_rews)),
            "ood_mean": float(np.mean(ood_rews)),
            "id_episodes": int(len(id_rews)),
            "ood_episodes": int(len(ood_rews)),
        }
    finally:
        env.close()


def _evaluate_checkpoint_payload(payload):
    item, opts = payload
    return evaluate_checkpoint_item(item, **opts)


def aggregate_item_results(item_results, envs, algos):
    all_results = {}
    for env in envs:
        env_results = {}
        for algo in algos:
            rows = [r for r in item_results
                    if r.get("env") == env and r.get("algo") == algo]
            if not rows:
                continue
            id_rewards = np.array([float(r["id_mean"]) for r in rows])
            ood_rewards = np.array([float(r["ood_mean"]) for r in rows])
            env_results[algo] = {
                "id_mean": float(np.mean(id_rewards)),
                "id_std": float(np.std(id_rewards)),
                "ood_mean": float(np.mean(ood_rewards)),
                "ood_std": float(np.std(ood_rewards)),
                "n_seeds": len(rows),
            }
        if env_results:
            all_results[env] = env_results
    return all_results


def evaluate_id_ood(results_root, env_name, algos, ood_scale=1.5,
                    n_id_tasks=20, n_ood_tasks=20, n_episodes=3,
                    tag_prefix=None):
    """Run ID/OOD evaluation for all algos on one environment."""
    varying_params, log_scale = ENV_PARAMS.get(env_name, (["gravity"], 0.75))

    # Create eval env
    env = NonstationaryEnv(env_name, rand_params=varying_params,
                           log_scale_limit=log_scale, seed=9999,
                           backend="spring")
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Sample ID and OOD tasks
    id_tasks = env.sample_tasks(n_id_tasks)

    orig_limit = env.log_scale_limit
    env.log_scale_limit = orig_limit * ood_scale
    ood_tasks = env.sample_tasks(n_ood_tasks)
    env.log_scale_limit = orig_limit

    results = {}
    for algo in algos:
        ckpt_dirs = find_checkpoints(results_root, algo, env_name, tag_prefix)
        if not ckpt_dirs:
            print(f"  No checkpoint for {algo}/{env_name}, skipping")
            continue

        id_rewards_all = []
        ood_rewards_all = []

        for ckpt_dir in ckpt_dirs:
            print(f"  Loading {ckpt_dir} ...")
            run_name = os.path.basename(os.path.dirname(ckpt_dir))
            config, _, _ = make_eval_config(env_name, algo, run_name)
            agent = load_agent_from_checkpoint(ckpt_dir, algo, obs_dim, act_dim, config)

            # Build rollout fn for eval
            policy_graphdef = nnx.graphdef(eval_policy_module(agent))
            context_graphdef = None
            if hasattr(agent, 'context_net'):
                context_graphdef = nnx.graphdef(agent.context_net)
            env.build_rollout_fn(policy_graphdef, context_graphdef)

            id_rews = eval_on_tasks(agent, env, id_tasks, config, n_episodes)
            ood_rews = eval_on_tasks(agent, env, ood_tasks, config, n_episodes)
            id_rewards_all.append(float(np.mean(id_rews)))
            ood_rewards_all.append(float(np.mean(ood_rews)))

        results[algo] = {
            "id_mean": float(np.mean(id_rewards_all)),
            "id_std": float(np.std(id_rewards_all)),
            "ood_mean": float(np.mean(ood_rewards_all)),
            "ood_std": float(np.std(ood_rewards_all)),
            "n_seeds": len(ckpt_dirs),
        }
        print(f"  {algo}: ID={results[algo]['id_mean']:.0f}±{results[algo]['id_std']:.0f}  "
              f"OOD={results[algo]['ood_mean']:.0f}±{results[algo]['ood_std']:.0f}")

    env.close()
    return results


def plot_id_ood_comparison(all_results, envs, algos, output=None, figsize=(12, 5)):
    """Plot ID vs OOD performance comparison (grouped bar chart)."""
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=figsize)
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        if env not in all_results:
            continue
        results = all_results[env]

        algo_names, id_means, ood_means, id_stds, ood_stds = [], [], [], [], []
        for algo in algos:
            if algo not in results:
                continue
            r = results[algo]
            algo_names.append(ALGO_LABELS.get(algo, algo))
            id_means.append(r["id_mean"])
            ood_means.append(r["ood_mean"])
            id_stds.append(r["id_std"])
            ood_stds.append(r["ood_std"])

        if not algo_names:
            continue

        x = np.arange(len(algo_names))
        width = 0.35
        ax.bar(x - width/2, id_means, width, yerr=id_stds,
               label='ID', color='#4CAF50', alpha=0.7, capsize=3, edgecolor='black')
        ax.bar(x + width/2, ood_means, width, yerr=ood_stds,
               label='OOD', color='#F44336', alpha=0.7, capsize=3, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=30, fontsize=8)
        ax.set_title(env.replace('-v2', ''), fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel('Mean Episode Reward', fontsize=10)
            ax.legend(fontsize=9)

        # Annotate degradation %
        for i in range(len(algo_names)):
            if abs(id_means[i]) > 1e-6:
                deg = (id_means[i] - ood_means[i]) / abs(id_means[i]) * 100
                y_pos = max(id_means[i], ood_means[i]) + max(id_stds[i], ood_stds[i])
                color = 'red' if deg > 20 else 'orange' if deg > 10 else 'green'
                ax.text(x[i], y_pos * 1.02, f"{deg:+.0f}%",
                        ha='center', fontsize=7, color=color, fontweight='bold')

    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        fig.savefig(output.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved ID/OOD plot to: {output}")
    else:
        plt.show()
    plt.close(fig)


def print_summary_table(all_results, envs, algos):
    print(f"\n{'='*80}")
    print("ID/OOD Summary (mean reward)")
    print(f"{'='*80}")
    header = f"{'Algo':<20}" + "".join(f"{'ID':>8} {'OOD':>8} {'Δ%':>6}  " for _ in envs)
    print("               " + "  ".join(f"{e.replace('-v2',''):^22}" for e in envs))
    print(header)
    for algo in algos:
        row = f"{ALGO_LABELS.get(algo, algo):<20}"
        for env in envs:
            r = all_results.get(env, {}).get(algo)
            if r:
                deg = (r['id_mean'] - r['ood_mean']) / max(abs(r['id_mean']), 1) * 100
                row += f"{r['id_mean']:8.0f} {r['ood_mean']:8.0f} {deg:+5.0f}%  "
            else:
                row += f"{'---':>8} {'---':>8} {'---':>6}  "
        print(row)


def resolve_workers(value):
    if value is None:
        return 1
    text = str(value).strip().lower()
    if text == "auto":
        text = os.environ.get("SCHEDULEURM_CPU_WORKERS", "1")
    try:
        return max(1, int(text))
    except ValueError:
        return 1


def resolve_shard_bounds(args, total_items):
    start = args.shard_start
    end = args.shard_end
    if start is None:
        env_start = os.environ.get("SCHEDULEURM_CPU_SHARD_START")
        start = int(env_start) if env_start is not None else 0
    if end is None:
        env_end = os.environ.get("SCHEDULEURM_CPU_SHARD_END")
        end = int(env_end) if env_end is not None else total_items
    start = max(0, min(int(start), total_items))
    end = max(start, min(int(end), total_items))
    return start, end


def run_shard(items, args):
    start, end = resolve_shard_bounds(args, len(items))
    selected = items[start:end]
    workers = min(resolve_workers(args.workers), max(1, len(selected)))
    opts = {
        "ood_scale": args.ood_scale,
        "n_id_tasks": args.n_tasks,
        "n_ood_tasks": args.n_tasks,
        "n_episodes": args.n_episodes,
    }
    print(f"Evaluating shard [{start},{end}) / {len(items)} with workers={workers}")

    results = []
    if not selected:
        return start, end, results
    if workers <= 1:
        for item in selected:
            results.append(evaluate_checkpoint_item(item, **opts))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_evaluate_checkpoint_payload, (item, opts)): item
                for item in selected
            }
            for fut in as_completed(futures):
                item = futures[fut]
                result = fut.result()
                print(f"  Done {item['env']}/{item['algo']} seed={item.get('seed')}", flush=True)
                results.append(result)
    results.sort(key=lambda r: (r.get("env", ""), r.get("algo", ""),
                                -1 if r.get("seed") is None else r.get("seed"),
                                r.get("run_name", "")))
    return start, end, results


def write_shard_output(path, args, total_items, start, end, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "kind": "id_ood_shard",
        "results_root": args.results_root,
        "envs": args.envs,
        "algos": args.algos,
        "tag_prefix": args.tag_prefix,
        "seeds": args.seeds,
        "n_tasks": args.n_tasks,
        "n_episodes": args.n_episodes,
        "ood_scale": args.ood_scale,
        "total_items": total_items,
        "shard_start": start,
        "shard_end": end,
        "n_results": len(results),
        "items": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved shard results to: {path}")


def expand_shard_inputs(paths):
    out = []
    for path in paths or []:
        if os.path.isdir(path):
            out.extend(sorted(glob.glob(os.path.join(path, "*.json"))))
        else:
            out.append(path)
    return out


def load_shard_results(paths):
    item_results = []
    for path in expand_shard_inputs(paths):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        item_results.extend(payload.get("items", []))
    # Deduplicate by checkpoint path in case a shard was rerun.
    dedup = {}
    for row in item_results:
        dedup[row.get("ckpt_dir") or row.get("run_name")] = row
    return list(dedup.values())


def main():
    parser = argparse.ArgumentParser(description="ID/OOD Generalization Evaluation")
    parser.add_argument("--results_root", type=str, default="jax_experiments/results")
    parser.add_argument("--envs", nargs="+",
                        default=["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"])
    parser.add_argument("--algos", nargs="+", default=["bapr", "escp", "resac"])
    parser.add_argument("--ood_scale", type=float, default=1.5,
                        help="OOD range = ood_scale × training range")
    parser.add_argument("--n_tasks", type=int, default=20)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--tag_prefix", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--mean_dwell_iters", "--mean-dwell-iters",
                        type=int, default=60)
    parser.add_argument("--shard_start", "--shard-start", type=int, default=None)
    parser.add_argument("--shard_end", "--shard-end", type=int, default=None)
    parser.add_argument("--workers", type=str, default="1")
    parser.add_argument("--shard_output", "--shard-output", type=str, default=None)
    parser.add_argument("--merge_shards", "--merge-shards", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.merge_shards:
        item_results = load_shard_results(args.merge_shards)
        all_results = aggregate_item_results(item_results, args.envs, args.algos)
        if args.output:
            plot_id_ood_comparison(all_results, args.envs, args.algos, args.output)
        print_summary_table(all_results, args.envs, args.algos)
        return

    if args.shard_output or args.shard_start is not None or args.shard_end is not None:
        items = build_eval_items(
            args.results_root, args.envs, args.algos,
            tag_prefix=args.tag_prefix,
            seeds=args.seeds,
            mean_dwell_iters=args.mean_dwell_iters)
        start, end, results = run_shard(items, args)
        if args.shard_output:
            write_shard_output(args.shard_output, args, len(items), start, end, results)
        all_results = aggregate_item_results(results, args.envs, args.algos)
        print_summary_table(all_results, args.envs, args.algos)
        return

    all_results = {}
    for env in args.envs:
        print(f"\n{'='*60}")
        print(f"  Evaluating ID/OOD on {env}")
        print(f"{'='*60}")
        all_results[env] = evaluate_id_ood(
            args.results_root, env, args.algos,
            ood_scale=args.ood_scale,
            n_id_tasks=args.n_tasks,
            n_ood_tasks=args.n_tasks,
            n_episodes=args.n_episodes,
            tag_prefix=args.tag_prefix)

    if args.output:
        plot_id_ood_comparison(all_results, args.envs, args.algos, args.output)

    print_summary_table(all_results, args.envs, args.algos)


if __name__ == "__main__":
    main()
