"""Sensitivity analysis: sweep over key BAPR hyperparameters and plot heatmaps.

Generates grid sweep configs and a heatmap plotter for:
  - BOCD hazard rate
  - Penalty scale
  - Surprise signal weights (w_r, w_q, w_κ)

Usage:
    # Generate sweep configs
    python -m jax_experiments.analysis.sensitivity_sweep --mode generate \
        --env HalfCheetah-v2 --output_dir jax_experiments/sensitivity_runs

    # After running all configs, plot heatmaps
    python -m jax_experiments.analysis.sensitivity_sweep --mode plot \
        --results_dir jax_experiments/sensitivity_runs \
        --output paper/fig_sensitivity.pdf
"""
import argparse
import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════
#  Sweep configuration
# ═══════════════════════════════════════════════════════════════════════
SWEEP_PARAMS = {
    "hazard_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "penalty_scale": [1.0, 2.5, 5.0, 7.5, 10.0],
    "surprise_weights": [
        (0.5, 0.3, 0.2),   # default
        (0.8, 0.1, 0.1),   # reward-dominant
        (0.2, 0.6, 0.2),   # Q-std-dominant
        (0.33, 0.33, 0.34), # uniform
        (0.1, 0.1, 0.8),   # reg-dominant
    ],
}


def generate_sweep_scripts(env, output_dir, base_seed=8, max_iters=500):
    """Generate shell scripts for sensitivity sweep experiments."""
    os.makedirs(output_dir, exist_ok=True)

    configs = []

    # 1. Hazard rate × penalty scale 2D grid
    for hr, ps in itertools.product(SWEEP_PARAMS["hazard_rate"],
                                     SWEEP_PARAMS["penalty_scale"]):
        name = f"sens_hr{hr}_ps{ps}_{env}"
        configs.append({
            "name": name,
            "hazard_rate": hr,
            "penalty_scale": ps,
            "env": env,
            "seed": base_seed,
            "max_iters": max_iters,
        })

    # 2. Surprise weights (1D sweep)
    for i, (wr, wq, wk) in enumerate(SWEEP_PARAMS["surprise_weights"]):
        name = f"sens_sw{i}_wr{wr}_wq{wq}_{env}"
        configs.append({
            "name": name,
            "surprise_weights": [wr, wq, wk],
            "env": env,
            "seed": base_seed,
            "max_iters": max_iters,
        })

    # Write configs
    config_path = os.path.join(output_dir, "sweep_configs.json")
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"Wrote {len(configs)} configs to {config_path}")

    # Generate shell script
    script_lines = [
        "#!/bin/bash",
        "# Auto-generated sensitivity sweep",
        "set -e",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15",
        "",
    ]

    for cfg in configs:
        cmd = (f"python -u -m jax_experiments.train --algo bapr "
               f"--env {cfg['env']} --seed {cfg['seed']} "
               f"--max_iters {cfg['max_iters']} "
               f"--save_root {output_dir} --run_name {cfg['name']} "
               f"--backend spring")

        if 'hazard_rate' in cfg:
            cmd += f" --hazard_rate {cfg['hazard_rate']}"
        if 'penalty_scale' in cfg:
            cmd += f" --penalty_scale {cfg['penalty_scale']}"

        cfg_name = cfg["name"]
        script_lines.append(f"echo '=== Running {cfg_name} ==='")
        script_lines.append(cmd + " &")
        script_lines.append("sleep 5")
        script_lines.append("")

    script_lines.append("wait")
    script_lines.append("echo 'All sensitivity runs complete!'")

    script_path = os.path.join(output_dir, "run_sensitivity.sh")
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    os.chmod(script_path, 0o755)
    print(f"Wrote run script to {script_path}")


def plot_sensitivity_heatmap(results_dir, output=None, figsize=(16, 5)):
    """Plot sensitivity heatmap from completed sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Load results
    config_path = os.path.join(results_dir, "sweep_configs.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            configs = json.load(f)
    else:
        print(f"No sweep_configs.json found in {results_dir}")
        return

    # 1. Hazard rate × Penalty scale heatmap
    hr_vals = sorted(set(c.get("hazard_rate", 0) for c in configs if "hazard_rate" in c))
    ps_vals = sorted(set(c.get("penalty_scale", 0) for c in configs if "penalty_scale" in c))

    if hr_vals and ps_vals:
        grid = np.full((len(hr_vals), len(ps_vals)), np.nan)
        for cfg in configs:
            if "hazard_rate" not in cfg:
                continue
            log_dir = os.path.join(results_dir, cfg["name"], "logs")
            eval_rew = None
            eval_path = os.path.join(log_dir, "eval_reward.npy")
            if os.path.exists(eval_path):
                eval_rew = np.load(eval_path)

            if eval_rew is not None and len(eval_rew) > 0:
                # Use last 20% of training as final perf
                n_last = max(1, len(eval_rew) // 5)
                final_perf = float(np.mean(eval_rew[-n_last:]))
            else:
                final_perf = np.nan

            i = hr_vals.index(cfg["hazard_rate"])
            j = ps_vals.index(cfg["penalty_scale"])
            grid[i, j] = final_perf

        im = axes[0].imshow(grid, aspect='auto', origin='lower', cmap='RdYlGn')
        axes[0].set_xticks(range(len(ps_vals)))
        axes[0].set_xticklabels([f"{v}" for v in ps_vals], fontsize=8)
        axes[0].set_yticks(range(len(hr_vals)))
        axes[0].set_yticklabels([f"{v}" for v in hr_vals], fontsize=8)
        axes[0].set_xlabel('Penalty Scale', fontsize=10)
        axes[0].set_ylabel('Hazard Rate', fontsize=10)
        axes[0].set_title('Final Eval Reward', fontsize=11)
        fig.colorbar(im, ax=axes[0], shrink=0.8)

        # Annotate cells
        for i in range(len(hr_vals)):
            for j in range(len(ps_vals)):
                if not np.isnan(grid[i, j]):
                    axes[0].text(j, i, f"{grid[i, j]:.0f}",
                               ha='center', va='center', fontsize=7)

    # 2. Surprise weights bar chart
    sw_configs = [c for c in configs if "surprise_weights" in c]
    if sw_configs:
        labels = []
        perfs = []
        for cfg in sw_configs:
            w = cfg["surprise_weights"]
            labels.append(f"r={w[0]}\nq={w[1]}\nκ={w[2]}")
            log_dir = os.path.join(results_dir, cfg["name"], "logs")
            eval_path = os.path.join(log_dir, "eval_reward.npy")
            if os.path.exists(eval_path):
                e = np.load(eval_path)
                n_last = max(1, len(e) // 5)
                perfs.append(float(np.mean(e[-n_last:])))
            else:
                perfs.append(0.0)

        colors = ['#2196F3' if i == 0 else '#90CAF9' for i in range(len(labels))]
        axes[1].bar(range(len(labels)), perfs, color=colors, edgecolor='black')
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, fontsize=7)
        axes[1].set_ylabel('Final Eval Reward', fontsize=10)
        axes[1].set_title('Surprise Signal Weights', fontsize=11)

    # 3. β_eff trajectory comparison across hazard rates (fixed penalty_scale=5.0)
    # Shows how different hazard rates affect adaptive conservatism dynamics
    hr_traces = {}
    for cfg in configs:
        if "hazard_rate" not in cfg or cfg.get("penalty_scale", 5.0) != 5.0:
            continue
        log_dir = os.path.join(results_dir, cfg["name"], "logs")
        lw_path = os.path.join(log_dir, "weighted_lambda.npy")
        if os.path.exists(lw_path):
            lw = np.load(lw_path)
            hr_traces[cfg["hazard_rate"]] = lw

    if hr_traces:
        cmap_hr = plt.cm.get_cmap('viridis', len(hr_traces))
        for i, (hr, lw) in enumerate(sorted(hr_traces.items())):
            axes[2].plot(lw, color=cmap_hr(i), linewidth=0.8, alpha=0.8,
                        label=f'h={hr}')
        axes[2].set_xlabel('Iteration', fontsize=9)
        axes[2].set_ylabel('λ_w', fontsize=10)
        axes[2].set_title('λ_w Dynamics vs Hazard Rate\n(penalty_scale=5.0)', fontsize=10)
        axes[2].legend(fontsize=7)
    else:
        axes[2].text(0.5, 0.5, 'No λ_w data\navailable',
                    transform=axes[2].transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
        axes[2].set_title('λ_w Dynamics', fontsize=11)

    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        fig.savefig(output.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved sensitivity plot to: {output}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "plot"], required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--output_dir", type=str, default="jax_experiments/sensitivity_runs")
    parser.add_argument("--results_dir", type=str, default="jax_experiments/sensitivity_runs")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_sweep_scripts(args.env, args.output_dir)
    else:
        plot_sensitivity_heatmap(args.results_dir, args.output)


if __name__ == "__main__":
    main()
