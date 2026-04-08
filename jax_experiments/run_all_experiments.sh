#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  BAPR Full Experiment Suite — Server Runner
#  Designed for GPU server (≥16GB VRAM). Adjust MAX_PARALLEL for your GPU.
#
#  Phases:
#    1. Multi-seed main results   (4 algo × 4 env × 5 seeds = 80 runs)
#    2. Ablation + Bad-BAPR       (5 ablations on HC, 1 seed)
#    3. Sensitivity heatmap       (5×5 HR×PS grid, 1 seed)
#    4. Ablations on all envs     (2 key ablations × 3 extra envs)
#    5. ID/OOD generalization     (load checkpoints, eval on wider param range)
#    6. Generate all paper figures (BOCD money, t-SNE, adaptation, curves, table)
#
#  Usage:
#    bash jax_experiments/run_all_experiments.sh all       # everything
#    bash jax_experiments/run_all_experiments.sh 1         # only phase 1
#    bash jax_experiments/run_all_experiments.sh 2 3       # phases 2 and 3
# ═══════════════════════════════════════════════════════════════════════════
set -e

# ── GPU / JAX setup ──────────────────────────────────────────────────────
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# Auto-detect CUDA libs from pip nvidia packages
NVIDIA_CANDIDATES=(
    "/home/erzhu419/anaconda3/envs/jax-rl/lib/python3.11/site-packages/nvidia"
    "$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/nvidia"
)
for NVIDIA_ROOT in "${NVIDIA_CANDIDATES[@]}"; do
    if [ -d "$NVIDIA_ROOT" ]; then
        CUDA_LIB_DIRS=""
        for subdir in "$NVIDIA_ROOT"/*/lib; do
            [ -d "$subdir" ] && CUDA_LIB_DIRS="${CUDA_LIB_DIRS:+$CUDA_LIB_DIRS:}$subdir"
        done
        export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
        break
    fi
done

# ── Configuration ────────────────────────────────────────────────────────
# Adjust MAX_PARALLEL based on GPU memory:
#   8GB  → 3-4 parallel
#   16GB → 6-8 parallel
#   24GB → 8-10 parallel
MAX_PARALLEL=${MAX_PARALLEL:-4}
MEM_FRACTION=${MEM_FRACTION:-0.12}

# Use conda if available, else plain python
if command -v conda &>/dev/null && conda info --envs 2>/dev/null | grep -q jax-rl; then
    PYTHON="conda run --no-capture-output -n jax-rl python -u"
else
    PYTHON="python -u"
fi

MODULE="jax_experiments.train"
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
MAX_ITERS=2000
CHANGING_PERIOD=40000
LOG_INTERVAL=5
EVAL_EPISODES=5
SAVE_INTERVAL=50

# Seeds
SEEDS=(0 1 2 3 4)

# Environment configs
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

ENVS=("HalfCheetah-v2" "Hopper-v2" "Walker2d-v2" "Ant-v2")
ALGOS=("bapr" "escp" "resac" "sac")

mkdir -p jax_experiments/logs

# ── Helper ───────────────────────────────────────────────────────────────
run() {
    local ALGO=$1 ENV=$2 SEED=$3 TAG=$4 EXTRA=$5
    local PARAMS="${ENV_PARAMS[$ENV]}"
    local SCALE="${ENV_SCALE[$ENV]}"
    local NAME="${ALGO}_${ENV}_${SEED}"
    [ -n "$TAG" ] && NAME="${NAME}_${TAG}"

    # Skip if already completed
    local EVAL_FILE="${SAVE_ROOT}/${NAME}/logs/eval_reward.npy"
    if [ -f "$EVAL_FILE" ]; then
        local N_EVALS
        N_EVALS=$($PYTHON -c "import numpy as np; print(len(np.load('$EVAL_FILE')))" 2>/dev/null || echo 0)
        local EXPECTED=$((MAX_ITERS / LOG_INTERVAL))
        # Allow override via EXTRA --max_iters
        if echo "$EXTRA" | grep -q "max_iters"; then
            EXPECTED=$(echo "$EXTRA" | grep -oP '(?<=max_iters )\d+')
            EXPECTED=$((EXPECTED / LOG_INTERVAL))
        fi
        if [ "$N_EVALS" -ge "$((EXPECTED * 9 / 10))" ]; then
            echo "    [SKIP] $NAME already done ($N_EVALS evals)"
            return
        fi
    fi

    export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION
    echo ">>> Starting: $NAME"
    $PYTHON -m $MODULE \
        --algo $ALGO --env $ENV --seed $SEED \
        --max_iters $MAX_ITERS \
        --log_interval $LOG_INTERVAL \
        --eval_episodes $EVAL_EPISODES \
        --save_interval $SAVE_INTERVAL \
        --save_root $SAVE_ROOT --run_name "$NAME" \
        --backend $BACKEND \
        --log_scale_limit $SCALE \
        --changing_period $CHANGING_PERIOD \
        --varying_params $PARAMS \
        --resume \
        $EXTRA \
        > "jax_experiments/logs/${NAME}.log" 2>&1
    echo "<<< Finished: $NAME (exit=$?)"
}

throttle() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 15
    done
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: Multi-seed main results — 4 algo × 4 env × 5 seeds = 80 runs
#  Paper: Fig. training curves (Q1) + Table main results
# ═══════════════════════════════════════════════════════════════════════════
phase1_multiseed() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: Multi-seed Main Results                           ║"
    echo "║  ${#ALGOS[@]} algos × ${#ENVS[@]} envs × ${#SEEDS[@]} seeds = $((${#ALGOS[@]}*${#ENVS[@]}*${#SEEDS[@]})) runs  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    for ENV in "${ENVS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for ALGO in "${ALGOS[@]}"; do
                run $ALGO $ENV $SEED "" "" &
                sleep 3
                throttle
            done
        done
    done
    wait
    echo "Phase 1 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: Ablation + Bad-BAPR on HalfCheetah (6 runs, seed=0)
#  Paper: Fig. ablation study (Q2) + Bad-BAPR divergence (Q4)
# ═══════════════════════════════════════════════════════════════════════════
phase2_ablation() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: Ablation + Bad-BAPR (HalfCheetah-v2)             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local ENV="HalfCheetah-v2"
    local SEED=0
    for ABL in "bapr_no_bocd" "bapr_no_rmdm" "bapr_no_adapt_beta" "bapr_fixed_decay" "bad_bapr"; do
        run $ABL $ENV $SEED "ablation" "" &
        sleep 3
        throttle
    done
    wait
    echo "Phase 2 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: Sensitivity sweep — HR × PS 5×5 grid on HalfCheetah
#  Paper: Fig. sensitivity heatmap (Q5)
# ═══════════════════════════════════════════════════════════════════════════
phase3_sensitivity() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: Sensitivity Sweep (HalfCheetah-v2, 500 iters)    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local ENV="HalfCheetah-v2"
    local SEED=0
    local SENS_ITERS=500

    for HR in 0.01 0.03 0.05 0.1 0.2; do
        for PS in 1.0 2.5 5.0 7.5 10.0; do
            run "bapr" $ENV $SEED "sens_hr${HR}_ps${PS}" \
                "--max_iters $SENS_ITERS --hazard_rate $HR --penalty_scale $PS" &
            sleep 2
            throttle
        done
    done
    wait
    echo "Phase 3 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: Ablations on all 4 envs (key ablations only)
#  Paper: Table cross-env ablation robustness
# ═══════════════════════════════════════════════════════════════════════════
phase4_ablations_all_envs() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: Key Ablations × All Envs                         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local SEED=0
    for ENV in "Hopper-v2" "Walker2d-v2" "Ant-v2"; do
        for ABL in "bapr_no_bocd" "bapr_fixed_decay"; do
            run $ABL $ENV $SEED "ablation" "" &
            sleep 3
            throttle
        done
    done
    wait
    echo "Phase 4 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: ID/OOD generalization evaluation
#  Paper: Fig. ID/OOD bar chart — evaluates trained checkpoints on wider
#         parameter ranges to show robustness to distribution shift
# ═══════════════════════════════════════════════════════════════════════════
phase5_id_ood() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 5: ID/OOD Generalization Evaluation                  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    PAPER_DIR="paper"
    mkdir -p $PAPER_DIR

    $PYTHON -m jax_experiments.analysis.eval_id_ood \
        --results_root $SAVE_ROOT \
        --envs "${ENVS[@]}" \
        --algos bapr escp resac \
        --ood_scale 1.5 \
        --n_tasks 20 \
        --n_episodes 3 \
        --output "${PAPER_DIR}/fig_id_ood.pdf" || true

    echo "Phase 5 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 6: Generate all paper figures from completed experiments
# ═══════════════════════════════════════════════════════════════════════════
phase6_analysis() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 6: Generate All Paper Figures                        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    PAPER_DIR="paper"
    mkdir -p $PAPER_DIR

    # --- Multi-seed training curves ---
    echo "--- Multi-seed training curves ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs "${ENVS[@]}" \
        --algos bapr escp resac sac \
        --output "${PAPER_DIR}/fig_training_curves.pdf" || true

    # --- Multi-seed results table (LaTeX) ---
    echo "--- Results table ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs "${ENVS[@]}" \
        --algos bapr escp resac sac \
        --table 2>&1 | tee "${PAPER_DIR}/results_table.txt" || true

    # --- Ablation curves (HalfCheetah) ---
    echo "--- Ablation curves ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 \
        --algos bapr bapr_no_bocd bapr_no_rmdm bapr_no_adapt_beta bapr_fixed_decay \
        --output "${PAPER_DIR}/fig_ablation_curves.pdf" || true

    # --- Bad-BAPR divergence ---
    echo "--- Bad-BAPR divergence ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 \
        --algos bapr bad_bapr \
        --output "${PAPER_DIR}/fig_bad_bapr_divergence.pdf" || true

    # --- BOCD Money Plot (per env) ---
    echo "--- BOCD money plots ---"
    for ENV in "${ENVS[@]}"; do
        LOG_DIR=$(ls -d ${SAVE_ROOT}/bapr_${ENV}_0/logs 2>/dev/null | head -1)
        [ -z "$LOG_DIR" ] && LOG_DIR=$(ls -d ${SAVE_ROOT}/bapr_${ENV}_*/logs 2>/dev/null | head -1)
        if [ -n "$LOG_DIR" ]; then
            $PYTHON -m jax_experiments.analysis.plot_bocd_money \
                --log_dir "$LOG_DIR" \
                --output "${PAPER_DIR}/fig_bocd_money_${ENV}.pdf" || true
        fi
    done

    # --- Adaptation speed ---
    echo "--- Adaptation speed ---"
    for ENV in "${ENVS[@]}"; do
        $PYTHON -m jax_experiments.analysis.adaptation_speed \
            --results_root $SAVE_ROOT --env $ENV \
            --algos bapr escp resac sac \
            --output "${PAPER_DIR}/fig_adaptation_speed_${ENV}.pdf" || true
    done

    # --- Embedding t-SNE ---
    echo "--- Embedding t-SNE ---"
    LOG_DIR=$(ls -d ${SAVE_ROOT}/bapr_HalfCheetah-v2_0/logs 2>/dev/null | head -1)
    [ -z "$LOG_DIR" ] && LOG_DIR=$(ls -d ${SAVE_ROOT}/bapr_HalfCheetah-v2_*/logs 2>/dev/null | head -1)
    if [ -n "$LOG_DIR" ]; then
        $PYTHON -m jax_experiments.analysis.embedding_viz \
            --log_dir "$LOG_DIR" \
            --output "${PAPER_DIR}/fig_embedding_tsne.pdf" || true
    fi

    # --- Sensitivity heatmap ---
    echo "--- Sensitivity heatmap ---"
    $PYTHON -m jax_experiments.analysis.sensitivity_sweep \
        --mode plot \
        --results_dir $SAVE_ROOT \
        --output "${PAPER_DIR}/fig_sensitivity.pdf" || true

    echo ""
    echo "Phase 6 complete! All figures saved to ${PAPER_DIR}/"
    ls -la ${PAPER_DIR}/*.pdf 2>/dev/null || echo "  (no PDF files generated)"
}

# ═══════════════════════════════════════════════════════════════════════════
#  Dispatch — supports multiple phases: ./run_all_experiments.sh 1 2 3
# ═══════════════════════════════════════════════════════════════════════════
PHASES="${@:-all}"

run_phase() {
    case "$1" in
        1|multiseed)        phase1_multiseed ;;
        2|ablation)         phase2_ablation ;;
        3|sensitivity)      phase3_sensitivity ;;
        4|ablations_all)    phase4_ablations_all_envs ;;
        5|id_ood)           phase5_id_ood ;;
        6|analysis)         phase6_analysis ;;
        *)
            echo "Unknown phase: $1"
            echo "Usage: $0 {1|2|3|4|5|6|all} [phase2 ...]"
            echo ""
            echo "Phases:"
            echo "  1/multiseed      Multi-seed main results (4 algo × 4 env × 5 seeds)"
            echo "  2/ablation       Ablation + Bad-BAPR on HalfCheetah"
            echo "  3/sensitivity    Hazard rate × penalty scale heatmap"
            echo "  4/ablations_all  Key ablations on all 4 environments"
            echo "  5/id_ood         ID/OOD generalization evaluation"
            echo "  6/analysis       Generate all paper figures"
            echo "  all              Run everything sequentially"
            exit 1
            ;;
    esac
}

if [ "$PHASES" = "all" ]; then
    phase1_multiseed
    phase2_ablation
    phase3_sensitivity
    phase4_ablations_all_envs
    phase5_id_ood
    phase6_analysis
else
    for phase in $PHASES; do
        run_phase "$phase"
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All requested phases complete!"
echo "═══════════════════════════════════════════════════════════════"
