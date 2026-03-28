#!/bin/bash

# BAPR 训练启动脚本

# 1. 训练 Baseline RE-SAC (原版代码 + 开启地狱难度模式切换环境)
echo "Starting Stress Baseline (Original RE-SAC + Mode Switch) training..."
python sac_ensemble_baseline_stress.py \
    --max_episodes 500 \
    --enable_mode_switch \
    --mode_switch_min 1800 \
    --mode_switch_max 7200 \
    --save_root bapr_experiments \
    --run_name baseline_stress \
    > train_baseline.log 2>&1 &

# 2. 训练 BA-PR SAC v4 (修复 + Context Conditioning: BOCD + RMDM + adaptive beta)
echo "Starting BA-PR SAC v4 (context-conditioned) training..."
python sac_ensemble_bapr.py \
    --max_episodes 500 \
    --enable_mode_switch \
    --mode_switch_min 1800 \
    --mode_switch_max 7200 \
    --max_run_length 20 \
    --hazard_rate 0.05 \
    --penalty_decay_rate 0.1 \
    --penalty_scale 5.0 \
    --ep_dim 4 \
    --repr_loss_weight 1.0 \
    --context_warmup_episodes 50 \
    --save_root bapr_experiments \
    --run_name bapr_v4_ctx \
    > train_bapr.log 2>&1 &

# 3. 训练 ESCP Baseline (Context-Sensitive Policy + RMDM Loss)
echo "Starting ESCP Baseline training..."
python sac_ensemble_escp_stress.py \
    --max_episodes 500 \
    --enable_mode_switch \
    --mode_switch_min 1800 \
    --mode_switch_max 7200 \
    --save_root bapr_experiments \
    --run_name escp_baseline \
    > train_escp.log 2>&1 &

echo "Experiments started in background."
echo "Check logs with:"
echo "  tail -f train_baseline.log"
echo "  tail -f train_bapr.log"
echo "  tail -f train_escp.log"
