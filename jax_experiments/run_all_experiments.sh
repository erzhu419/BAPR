#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  BAPR Full Experiment Suite — Master Runner
#  Covers P0 (ablation, BOCD viz, Bad-BAPR), P1 (baselines, sensitivity,
#  multi-seed), P2 (adaptation speed, ID/OOD, wall-clock), P3 (unsupervised)
# ═══════════════════════════════════════════════════════════════════════════
set -e

# GPU memory management
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# ── Configuration ──────────────────────────────────────────────────────────
PYTHON="python -u"
MODULE="jax_experiments.train"
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
MAX_ITERS=2000
LOG_SCALE=0.75
CHANGING_PERIOD=40000

# Environments: use best-known config per env
# HalfCheetah/Hopper: gravity, Ant: gravity, Walker2d: dof_damping+body_mass
declare -A ENV_PARAMS
ENV_PARAMS["HalfCheetah-v2"]="gravity"
ENV_PARAMS["Hopper-v2"]="gravity"
ENV_PARAMS["Walker2d-v2"]="dof_damping body_mass"
ENV_PARAMS["Ant-v2"]="gravity"

declare -A ENV_SCALE
ENV_SCALE["HalfCheetah-v2"]=0.75
ENV_SCALE["Hopper-v2"]=0.75
ENV_SCALE["Walker2d-v2"]=1.5
ENV_SCALE["Ant-v2"]=0.75

# Seeds for multi-seed runs
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# ── Helper function ────────────────────────────────────────────────────────
run_experiment() {
    local ALGO=$1
    local ENV=$2
    local SEED=$3
    local TAG=$4
    local EXTRA_ARGS=$5
    local MEM_FRAC=${6:-0.15}

    local PARAMS="${ENV_PARAMS[$ENV]}"
    local SCALE="${ENV_SCALE[$ENV]}"
    local NAME="${ALGO}_${ENV}_${SEED}_${TAG}"

    export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC

    echo "=== Starting: $NAME ==="
    $PYTHON -m $MODULE \
        --algo $ALGO --env $ENV --seed $SEED \
        --max_iters $MAX_ITERS \
        --save_root $SAVE_ROOT \
        --run_name $NAME \
        --backend $BACKEND \
        --log_scale_limit $SCALE \
        --changing_period $CHANGING_PERIOD \
        --varying_params $PARAMS \
        $EXTRA_ARGS \
        > "jax_experiments/logs/${NAME}.log" 2>&1
    echo "=== Finished: $NAME ==="
}

mkdir -p jax_experiments/logs

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: P0 — Core ablations + Bad-BAPR (on HalfCheetah, most stable)
# ═══════════════════════════════════════════════════════════════════════════
phase1_ablations() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: P0 — Ablation Study + Bad-BAPR (HalfCheetah-v2)  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local ENV="HalfCheetah-v2"
    local SEED=8
    local MEM=0.10

    # 4 ablations + 1 bad-bapr, run in parallel
    run_experiment "bapr_no_bocd"       $ENV $SEED "ablation" "" $MEM &
    sleep 10
    run_experiment "bapr_no_rmdm"       $ENV $SEED "ablation" "" $MEM &
    sleep 10
    run_experiment "bapr_no_adapt_beta" $ENV $SEED "ablation" "" $MEM &
    sleep 10
    run_experiment "bapr_fixed_decay"   $ENV $SEED "ablation" "" $MEM &
    sleep 10
    run_experiment "bad_bapr"           $ENV $SEED "ablation" "" $MEM &

    wait
    echo "Phase 1 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: P1 — SAC baseline + multi-seed main results
# ═══════════════════════════════════════════════════════════════════════════
phase2_baselines_multiseed() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: P1 — SAC baseline + 10-seed main results         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local MEM=0.10

    for ENV in "HalfCheetah-v2" "Hopper-v2" "Walker2d-v2" "Ant-v2"; do
        for SEED in "${SEEDS[@]}"; do
            for ALGO in "bapr" "escp" "resac" "sac"; do
                run_experiment $ALGO $ENV $SEED "ms" "" $MEM &
                sleep 5

                # Limit parallel jobs (GPU memory)
                while [ $(jobs -r | wc -l) -ge 4 ]; do
                    sleep 30
                done
            done
        done
    done

    wait
    echo "Phase 2 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: P1 — Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════
phase3_sensitivity() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: P1 — Sensitivity Analysis (HalfCheetah-v2)       ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local ENV="HalfCheetah-v2"
    local SEED=8
    local ITERS=500  # shorter for sweeps
    local MEM=0.08

    # Hazard rate sweep
    for HR in 0.01 0.03 0.1 0.2; do
        run_experiment "bapr" $ENV $SEED "sens_hr${HR}" \
            "--max_iters $ITERS --hazard_rate $HR" $MEM &
        sleep 5
    done

    # Penalty scale sweep
    for PS in 1.0 2.5 7.5 10.0; do
        run_experiment "bapr" $ENV $SEED "sens_ps${PS}" \
            "--max_iters $ITERS --penalty_scale $PS" $MEM &
        sleep 5
    done

    wait
    echo "Phase 3 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: P2 — Ablations on all 4 envs (not just HalfCheetah)
# ═══════════════════════════════════════════════════════════════════════════
phase4_ablations_all_envs() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: P2 — Ablations on all envs + adaptation speed    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local SEED=8
    local MEM=0.10

    for ENV in "Hopper-v2" "Walker2d-v2" "Ant-v2"; do
        for ABLATION in "bapr_no_bocd" "bapr_fixed_decay"; do
            run_experiment $ABLATION $ENV $SEED "ablation" "" $MEM &
            sleep 10

            while [ $(jobs -r | wc -l) -ge 4 ]; do
                sleep 30
            done
        done
    done

    wait
    echo "Phase 4 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: P3 — Unsupervised RMDM
# ═══════════════════════════════════════════════════════════════════════════
phase5_unsupervised() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 5: P3 — Unsupervised RMDM (BOCD pseudo-labels)     ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local SEED=8
    local MEM=0.12

    for ENV in "HalfCheetah-v2" "Hopper-v2"; do
        run_experiment "bapr_unsupervised" $ENV $SEED "unsup" "" $MEM &
        sleep 10
    done

    wait
    echo "Phase 5 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 6: Post-processing — generate all figures
# ═══════════════════════════════════════════════════════════════════════════
phase6_analysis() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 6: Generate all paper figures                        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    PAPER_DIR="paper"
    mkdir -p $PAPER_DIR

    echo "--- BOCD Money Plot ---"
    for ENV in "HalfCheetah-v2" "Hopper-v2" "Ant-v2"; do
        LOG_DIR=$(ls -d ${SAVE_ROOT}/bapr_${ENV}_8*/logs 2>/dev/null | head -1)
        if [ -n "$LOG_DIR" ]; then
            $PYTHON -m jax_experiments.analysis.plot_bocd_money \
                --log_dir "$LOG_DIR" \
                --output "${PAPER_DIR}/fig_bocd_money_${ENV}.pdf" || true
        fi
    done

    echo "--- Multi-seed curves ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 Hopper-v2 Walker2d-v2 Ant-v2 \
        --algos bapr escp resac sac \
        --output "${PAPER_DIR}/fig_multiseed_curves.pdf" || true

    echo "--- Multi-seed results table ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 Hopper-v2 Walker2d-v2 Ant-v2 \
        --algos bapr escp resac sac \
        --table || true

    echo "--- Ablation curves ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 \
        --algos bapr bapr_no_bocd bapr_no_rmdm bapr_no_adapt_beta bapr_fixed_decay \
        --output "${PAPER_DIR}/fig_ablation_curves.pdf" || true

    echo "--- Bad-BAPR divergence ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 \
        --algos bapr bad_bapr \
        --output "${PAPER_DIR}/fig_bad_bapr_divergence.pdf" || true

    echo "--- Adaptation speed ---"
    for ENV in "HalfCheetah-v2" "Hopper-v2" "Ant-v2"; do
        $PYTHON -m jax_experiments.analysis.adaptation_speed \
            --results_root $SAVE_ROOT --env $ENV \
            --output "${PAPER_DIR}/fig_adaptation_speed_${ENV}.pdf" || true
    done

    echo "--- Embedding visualization ---"
    LOG_DIR=$(ls -d ${SAVE_ROOT}/bapr_HalfCheetah-v2_8*/logs 2>/dev/null | head -1)
    if [ -n "$LOG_DIR" ]; then
        $PYTHON -m jax_experiments.analysis.embedding_viz \
            --log_dir "$LOG_DIR" \
            --output "${PAPER_DIR}/fig_embedding_tsne.pdf" || true
    fi

    echo "Phase 6 complete! All figures saved to ${PAPER_DIR}/"
}

# ═══════════════════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════════════════
PHASE="${1:-all}"

case "$PHASE" in
    1|ablations)        phase1_ablations ;;
    2|multiseed)        phase2_baselines_multiseed ;;
    3|sensitivity)      phase3_sensitivity ;;
    4|ablations_all)    phase4_ablations_all_envs ;;
    5|unsupervised)     phase5_unsupervised ;;
    6|analysis)         phase6_analysis ;;
    all)
        phase1_ablations
        phase2_baselines_multiseed
        phase3_sensitivity
        phase4_ablations_all_envs
        phase5_unsupervised
        phase6_analysis
        ;;
    *)
        echo "Usage: $0 {1|2|3|4|5|6|all|ablations|multiseed|sensitivity|ablations_all|unsupervised|analysis}"
        echo ""
        echo "Phases:"
        echo "  1/ablations      P0: Core ablation study + Bad-BAPR"
        echo "  2/multiseed      P1: Multi-seed main results + SAC baseline"
        echo "  3/sensitivity    P1: Sensitivity analysis sweeps"
        echo "  4/ablations_all  P2: Ablations on all 4 environments"
        echo "  5/unsupervised   P3: Unsupervised RMDM variant"
        echo "  6/analysis       Generate all paper figures"
        echo "  all              Run everything sequentially"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All requested phases complete!"
echo "═══════════════════════════════════════════════════════════════"
