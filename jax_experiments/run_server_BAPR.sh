#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  BAPR Server Runner — Auto-detect GPU resources and run full experiment suite
#
#  Handles:
#   - Multiple GPUs (detects via nvidia-smi)
#   - Shared GPUs with other users (queries free VRAM dynamically)
#   - Resume on interruption (checkpoint-based)
#   - Auto-skip already-completed runs
#   - PID-file lock prevents concurrent scheduler instances
#
#  Usage (one-liner):
#    bash jax_experiments/run_server_BAPR.sh
#
#  Environment overrides:
#    PER_RUN_MB=1800 bash jax_experiments/run_server_BAPR.sh
#    MAX_PARALLEL=4 bash jax_experiments/run_server_BAPR.sh
#    SEEDS_OVERRIDE="0 1 2" bash jax_experiments/run_server_BAPR.sh
#    PHASES="1 2 6" bash jax_experiments/run_server_BAPR.sh  # partial phases
#    PHASES="" bash ... is an error (use unset or specific phases)
# ═══════════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."   # cd to repo root (BAPR/)

# ═══════════════════════════════════════════════════════════════════════════
#  PID-file lock: prevent multiple scheduler instances from running concurrently
# ═══════════════════════════════════════════════════════════════════════════
LOCK_FILE="${LOCK_FILE:-/tmp/bapr_runner.lock}"
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: Another scheduler instance (PID $OLD_PID) is already running." >&2
        echo "       Lock file: $LOCK_FILE" >&2
        echo "       To force-start: rm $LOCK_FILE && bash $0" >&2
        exit 1
    else
        echo "WARNING: stale lock file (PID $OLD_PID not running). Removing." >&2
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM

# ═══════════════════════════════════════════════════════════════════════════
#  Resource detection
# ═══════════════════════════════════════════════════════════════════════════
PER_RUN_MB=${PER_RUN_MB:-2800}   # realistic per-process after contention observation
SAFETY_MB=${SAFETY_MB:-1500}     # keep this much free per GPU (buffer)
# With 12GB GPU, SAFETY=1500: (12288-1500)/2800 = 3 slots per GPU → 6 total
MAX_PARALLEL_CAP=${MAX_PARALLEL:-16}  # absolute cap (ignore if detected < cap)

detect_gpus() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "ERROR: nvidia-smi not found. JAX will run on CPU (very slow)." >&2
        N_GPUS=0
        return
    fi
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $N_GPUS GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.free,memory.total \
        --format=csv,noheader
}

compute_slots() {
    # Populates GPU_MAX_SLOTS[i] for i in 0..N_GPUS-1. Echoes TOTAL_SLOTS.
    GPU_MAX_SLOTS=()
    GPU_USED_SLOTS=()
    local total=0
    for i in $(seq 0 $((N_GPUS - 1))); do
        local free_mb
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i | tr -d ' ')
        local slots=$(( (free_mb - SAFETY_MB) / PER_RUN_MB ))
        [ $slots -lt 0 ] && slots=0
        [ $slots -gt 8 ] && slots=8   # hard cap per GPU to avoid thrashing
        GPU_MAX_SLOTS[$i]=$slots
        GPU_USED_SLOTS[$i]=0
        total=$((total + slots))
        echo "  GPU $i: ${free_mb}MB free → $slots parallel slots"
    done
    if [ $total -gt $MAX_PARALLEL_CAP ]; then
        # Scale down proportionally to respect cap
        local ratio_num=$MAX_PARALLEL_CAP
        local new_total=0
        for i in $(seq 0 $((N_GPUS - 1))); do
            GPU_MAX_SLOTS[$i]=$(( GPU_MAX_SLOTS[$i] * ratio_num / total ))
            [ ${GPU_MAX_SLOTS[$i]} -lt 1 ] && [ $ratio_num -gt 0 ] && GPU_MAX_SLOTS[$i]=1
            new_total=$((new_total + GPU_MAX_SLOTS[$i]))
        done
        total=$new_total
    fi
    TOTAL_SLOTS=$total
}

detect_gpus
compute_slots
echo "Total parallel slots: $TOTAL_SLOTS"

# If currently no slots, wait for memory to free up (e.g., other users'
# jobs finishing) — re-detect every 2 minutes rather than exiting.
MAX_WAIT_MIN=${MAX_WAIT_MIN:-720}   # give up after 12h if still no memory
waited=0
while [ $TOTAL_SLOTS -lt 1 ]; do
    if [ $waited -ge $((MAX_WAIT_MIN * 60)) ]; then
        echo "ERROR: Waited ${MAX_WAIT_MIN} min but no GPU memory freed. Giving up." >&2
        exit 1
    fi
    echo "Not enough GPU memory (total slots=0). Waiting 120s and re-checking..."
    echo "Current free memory:"
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
    sleep 120
    waited=$((waited + 120))
    compute_slots
    echo "[$(date '+%H:%M:%S')] Re-detected: $TOTAL_SLOTS slots"
done

# ═══════════════════════════════════════════════════════════════════════════
#  JAX / Python environment setup
# ═══════════════════════════════════════════════════════════════════════════
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
# MEM_FRACTION is fraction of TOTAL VRAM; with PREALLOCATE=false it's just a cap
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(awk "BEGIN{printf \"%.2f\", $PER_RUN_MB/12000}")

setup_conda() {
    # Source conda from common locations even if not on PATH
    for conda_sh in \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/home/huiwei/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$conda_sh" ]; then
            source "$conda_sh"
            return 0
        fi
    done
    return 1
}
setup_conda || true

detect_python() {
    # Prefer conda envs (try bapr-jax first, then jax-rl for backwards compat)
    if command -v conda &>/dev/null; then
        for env in bapr-jax jax-rl; do
            if conda info --envs 2>/dev/null | grep -qE "^${env}[[:space:]]"; then
                echo "conda run --no-capture-output -n $env python"
                return
            fi
        done
    fi
    echo "python"
}

# CUDA lib dirs from pip nvidia packages (if present)
setup_cuda_libs() {
    local py
    py=$(detect_python)
    local nvidia_root
    nvidia_root=$($py -c "import site, os; [print(os.path.join(p,'nvidia')) for p in site.getsitepackages() if os.path.isdir(os.path.join(p,'nvidia'))]" 2>/dev/null | head -1)
    if [ -n "$nvidia_root" ] && [ -d "$nvidia_root" ]; then
        local cuda_libs=""
        for d in "$nvidia_root"/*/lib; do
            [ -d "$d" ] && cuda_libs="${cuda_libs:+$cuda_libs:}$d"
        done
        export LD_LIBRARY_PATH="${cuda_libs}:${LD_LIBRARY_PATH:-}"
    fi
}
setup_cuda_libs

PYTHON="$(detect_python) -u"
echo "Python: $PYTHON"

# ═══════════════════════════════════════════════════════════════════════════
#  Experiment configuration
# ═══════════════════════════════════════════════════════════════════════════
MODULE="jax_experiments.train"
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
MAX_ITERS=${MAX_ITERS:-2000}
CHANGING_PERIOD=40000
LOG_INTERVAL=5
EVAL_EPISODES=5
SAVE_INTERVAL=50

# Seeds: 5 by default for paper-grade statistics
if [ -n "$SEEDS_OVERRIDE" ]; then
    read -ra SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(0 1 2 3 4)
fi
echo "Seeds: ${SEEDS[*]}"

declare -A ENV_PARAMS
# v5 design: DISCRETE-MODE regimes (sample_tasks samples extremes only, not continuous).
# Each env's parameters chosen so that log_scale extremes create genuinely different
# optimal policies (BAPR's piecewise-stationary assumption), not just parameter
# tolerance within a continuous family (ESCP's sweet spot).
ENV_PARAMS["HalfCheetah-v2"]="dof_damping body_mass"  # 4 modes: stiff/fluid × light/heavy
ENV_PARAMS["Hopper-v2"]="gravity"                     # 2 modes: low-g / high-g
ENV_PARAMS["Walker2d-v2"]="gravity"                   # 2 modes: low-g / high-g
ENV_PARAMS["Ant-v2"]="gravity body_mass"              # 4 modes: low-g × light_body combos

declare -A ENV_SCALE
# Extremes only (discrete sampling): scale = separation between 2 modes.
ENV_SCALE["HalfCheetah-v2"]=0.75  # damping ×0.47 vs ×2.12  +  mass ×0.47 vs ×2.12
ENV_SCALE["Hopper-v2"]=0.3        # gravity ×0.74 vs ×1.35 (mild for stability)
ENV_SCALE["Walker2d-v2"]=0.3      # gravity ×0.74 vs ×1.35
ENV_SCALE["Ant-v2"]=0.75          # gravity ×0.47 vs ×2.12  +  mass ×0.47 vs ×2.12

# Per-env physics backend (spring=fast; generalized=accurate, needed for Hopper/Walker)
declare -A ENV_BACKEND
ENV_BACKEND["HalfCheetah-v2"]="spring"
ENV_BACKEND["Hopper-v2"]="generalized"      # spring causes unlearnable dynamics
ENV_BACKEND["Walker2d-v2"]="generalized"    # same rationale
ENV_BACKEND["Ant-v2"]="spring"

ENVS=("HalfCheetah-v2" "Hopper-v2" "Walker2d-v2" "Ant-v2")
MAIN_ALGOS=("bapr" "escp" "resac" "sac")
ABL_ALGOS=("bapr_no_bocd" "bapr_no_rmdm" "bapr_no_adapt_beta" "bapr_fixed_decay" "bad_bapr")

mkdir -p jax_experiments/logs "$SAVE_ROOT"

# ═══════════════════════════════════════════════════════════════════════════
#  GPU slot scheduler — pin each job to the least-loaded GPU
# ═══════════════════════════════════════════════════════════════════════════
declare -A JOB_TO_GPU   # pid -> gpu_id

reap_finished_jobs() {
    for pid in "${!JOB_TO_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local gpu=${JOB_TO_GPU[$pid]}
            GPU_USED_SLOTS[$gpu]=$((GPU_USED_SLOTS[$gpu] - 1))
            unset "JOB_TO_GPU[$pid]"
        fi
    done
}

pick_gpu() {
    # Sets PICKED_GPU global to GPU id with most free slots, or -1 if none.
    # (Using global instead of stdout-return to avoid subshell state loss.)
    local best=-1 best_free=0
    for i in $(seq 0 $((N_GPUS - 1))); do
        local free=$(( GPU_MAX_SLOTS[$i] - GPU_USED_SLOTS[$i] ))
        if [ $free -gt $best_free ]; then
            best_free=$free; best=$i
        fi
    done
    PICKED_GPU=$best
}

wait_for_slot() {
    # Sets PICKED_GPU to an available GPU id. Blocks until slot frees up.
    # CRITICAL: must run in parent shell, not $(...), so that
    # reap_finished_jobs persistently updates GPU_USED_SLOTS and JOB_TO_GPU.
    local poll_count=0
    while true; do
        reap_finished_jobs
        pick_gpu
        [ $PICKED_GPU -ge 0 ] && return
        # Every ~3 min (18 polls × 10s), re-detect GPU memory in case more
        # capacity opened up from external jobs finishing.
        poll_count=$((poll_count + 1))
        if [ $((poll_count % 18)) -eq 0 ]; then
            local old_total=$TOTAL_SLOTS
            compute_slots
            if [ $TOTAL_SLOTS -ne $old_total ]; then
                echo "[$(date '+%H:%M:%S')] Re-detected GPU slots: $old_total → $TOTAL_SLOTS"
            fi
        fi
        sleep 10
    done
}

run_job() {
    # Args: ALGO ENV SEED TAG EXTRA_ARGS
    local ALGO=$1 ENV=$2 SEED=$3 TAG=$4 EXTRA=$5
    local PARAMS="${ENV_PARAMS[$ENV]}"
    local SCALE="${ENV_SCALE[$ENV]}"
    local BACKEND_THIS="${ENV_BACKEND[$ENV]:-$BACKEND}"   # per-env override
    local NAME="${ALGO}_${ENV}_${SEED}"
    [ -n "$TAG" ] && NAME="${NAME}_${TAG}"

    # Skip if already completed
    local EVAL_FILE="${SAVE_ROOT}/${NAME}/logs/eval_reward.npy"
    if [ -f "$EVAL_FILE" ]; then
        local N_EVALS
        N_EVALS=$($PYTHON -c "import numpy as np; print(len(np.load('$EVAL_FILE')))" 2>/dev/null || echo 0)
        local LOCAL_MAX_ITERS=$MAX_ITERS
        if echo "$EXTRA" | grep -q 'max_iters'; then
            LOCAL_MAX_ITERS=$(echo "$EXTRA" | grep -oP '(?<=max_iters )\d+')
        fi
        local EXPECTED=$((LOCAL_MAX_ITERS / LOG_INTERVAL))
        if [ "$N_EVALS" -ge "$((EXPECTED * 9 / 10))" ]; then
            echo "    [SKIP] $NAME already done ($N_EVALS/$EXPECTED evals)"
            return
        fi
    fi

    # Extra safety: don't re-launch if process for this run is already alive
    if pgrep -f "run_name $NAME " >/dev/null 2>&1; then
        echo "    [SKIP] $NAME already running (process alive)"
        return
    fi

    wait_for_slot   # sets PICKED_GPU in parent shell
    local gpu=$PICKED_GPU
    GPU_USED_SLOTS[$gpu]=$((GPU_USED_SLOTS[$gpu] + 1))

    echo ">>> [GPU $gpu | $(date +%H:%M:%S)] Starting: $NAME"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        $PYTHON -m $MODULE \
            --algo "$ALGO" --env "$ENV" --seed "$SEED" \
            --max_iters "$MAX_ITERS" \
            --log_interval "$LOG_INTERVAL" \
            --eval_episodes "$EVAL_EPISODES" \
            --save_interval "$SAVE_INTERVAL" \
            --save_root "$SAVE_ROOT" --run_name "$NAME" \
            --backend "$BACKEND_THIS" \
            --log_scale_limit "$SCALE" \
            --changing_period "$CHANGING_PERIOD" \
            --varying_params $PARAMS \
            --resume \
            $EXTRA \
            > "jax_experiments/logs/${NAME}.log" 2>&1
    ) &
    local pid=$!
    JOB_TO_GPU[$pid]=$gpu
}

wait_all() {
    while [ ${#JOB_TO_GPU[@]} -gt 0 ]; do
        reap_finished_jobs
        [ ${#JOB_TO_GPU[@]} -eq 0 ] && break
        sleep 15
    done
    wait 2>/dev/null || true
}

# ═══════════════════════════════════════════════════════════════════════════
#  Phase definitions
# ═══════════════════════════════════════════════════════════════════════════
phase1_multiseed() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: Multi-seed Main Results                           ║"
    echo "║  ${#MAIN_ALGOS[@]} algos × ${#ENVS[@]} envs × ${#SEEDS[@]} seeds = $((${#MAIN_ALGOS[@]} * ${#ENVS[@]} * ${#SEEDS[@]})) runs"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    for ENV in "${ENVS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for ALGO in "${MAIN_ALGOS[@]}"; do
                run_job "$ALGO" "$ENV" "$SEED" "" ""
                sleep 2
            done
        done
    done
    wait_all
    echo "Phase 1 complete!"
}

phase2_ablation() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: Ablation + Bad-BAPR (HalfCheetah-v2)             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    local ENV="HalfCheetah-v2" SEED=0
    for ABL in "${ABL_ALGOS[@]}"; do
        run_job "$ABL" "$ENV" "$SEED" "ablation" ""
        sleep 2
    done
    wait_all
    echo "Phase 2 complete!"
}

phase3_sensitivity() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: Sensitivity Sweep (HalfCheetah-v2, 500 iters)    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    local ENV="HalfCheetah-v2" SEED=0 SENS_ITERS=500
    for HR in 0.01 0.03 0.05 0.1 0.2; do
        for PS in 1.0 2.5 5.0 7.5 10.0; do
            run_job "bapr" "$ENV" "$SEED" "sens_hr${HR}_ps${PS}" \
                "--max_iters $SENS_ITERS --hazard_rate $HR --penalty_scale $PS"
            sleep 2
        done
    done
    wait_all
    echo "Phase 3 complete!"
}

phase4_ablations_all_envs() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: Key Ablations × Hopper/Walker/Ant                ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    local SEED=0
    for ENV in "Hopper-v2" "Walker2d-v2" "Ant-v2"; do
        for ABL in "bapr_no_bocd" "bapr_fixed_decay"; do
            run_job "$ABL" "$ENV" "$SEED" "ablation" ""
            sleep 2
        done
    done
    wait_all
    echo "Phase 4 complete!"
}

phase5_id_ood() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 5: ID/OOD Generalization Evaluation                  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    mkdir -p paper
    # Run on first GPU (sequential eval, not parallelized)
    export CUDA_VISIBLE_DEVICES=0
    $PYTHON -m jax_experiments.analysis.eval_id_ood \
        --results_root "$SAVE_ROOT" \
        --envs "${ENVS[@]}" \
        --algos bapr escp resac \
        --ood_scale 1.5 --n_tasks 20 --n_episodes 3 \
        --output paper/fig_id_ood.pdf 2>&1 | tee paper/id_ood_log.txt || true
    unset CUDA_VISIBLE_DEVICES
    echo "Phase 5 complete!"
}

phase6_analysis() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 6: Generate All Paper Figures                        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    local P=paper
    mkdir -p $P
    export CUDA_VISIBLE_DEVICES=0   # figures use little GPU

    echo "--- Multi-seed training curves ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root "$SAVE_ROOT" \
        --envs "${ENVS[@]}" --algos bapr escp resac sac \
        --output "$P/fig_training_curves.pdf" || true

    echo "--- Results table ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root "$SAVE_ROOT" \
        --envs "${ENVS[@]}" --algos bapr escp resac sac \
        --table 2>&1 | tee "$P/results_table.txt" || true

    echo "--- Ablation curves ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root "$SAVE_ROOT" \
        --envs HalfCheetah-v2 \
        --algos bapr bapr_no_bocd bapr_no_rmdm bapr_no_adapt_beta bapr_fixed_decay \
        --output "$P/fig_ablation_curves.pdf" || true

    echo "--- Bad-BAPR divergence ---"
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root "$SAVE_ROOT" \
        --envs HalfCheetah-v2 --algos bapr bad_bapr \
        --output "$P/fig_bad_bapr_divergence.pdf" || true

    echo "--- BOCD money plots ---"
    for ENV in "${ENVS[@]}"; do
        local LOG_DIR
        LOG_DIR=$(ls -d "$SAVE_ROOT"/bapr_${ENV}_0/logs 2>/dev/null | head -1)
        [ -z "$LOG_DIR" ] && LOG_DIR=$(ls -d "$SAVE_ROOT"/bapr_${ENV}_*/logs 2>/dev/null | head -1)
        if [ -n "$LOG_DIR" ]; then
            $PYTHON -m jax_experiments.analysis.plot_bocd_money \
                --log_dir "$LOG_DIR" \
                --output "$P/fig_bocd_money_${ENV}.pdf" || true
        fi
    done

    echo "--- Adaptation speed ---"
    for ENV in "${ENVS[@]}"; do
        $PYTHON -m jax_experiments.analysis.adaptation_speed \
            --results_root "$SAVE_ROOT" --env "$ENV" \
            --algos bapr escp resac sac \
            --output "$P/fig_adaptation_speed_${ENV}.pdf" || true
    done

    echo "--- Embedding t-SNE ---"
    local LOG_DIR
    LOG_DIR=$(ls -d "$SAVE_ROOT"/bapr_HalfCheetah-v2_0/logs 2>/dev/null | head -1)
    [ -z "$LOG_DIR" ] && LOG_DIR=$(ls -d "$SAVE_ROOT"/bapr_HalfCheetah-v2_*/logs 2>/dev/null | head -1)
    if [ -n "$LOG_DIR" ]; then
        $PYTHON -m jax_experiments.analysis.embedding_viz \
            --log_dir "$LOG_DIR" \
            --output "$P/fig_embedding_tsne.pdf" || true
    fi

    echo "--- Sensitivity heatmap ---"
    $PYTHON -m jax_experiments.analysis.sensitivity_sweep \
        --mode plot --results_dir "$SAVE_ROOT" \
        --output "$P/fig_sensitivity.pdf" || true

    unset CUDA_VISIBLE_DEVICES
    echo ""
    echo "Phase 6 complete! Figures in $P/"
    ls -la $P/*.pdf 2>/dev/null || true
}

# ═══════════════════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════════════════
# PHASES env var:
#   unset  → default to all phases "1 2 3 4 5 6"
#   set and non-empty → use as-is (e.g. "1 6", "2")
#   set but empty "" → explicit no-op, exit cleanly (useful for tests)
if [ -z "${PHASES+x}" ]; then
    PHASES_TO_RUN="1 2 3 4 5 6"
elif [ -z "$PHASES" ] || [ "$PHASES" = " " ]; then
    echo "PHASES is explicitly empty — exiting without launching any jobs."
    exit 0
else
    PHASES_TO_RUN="$PHASES"
fi

START_TIME=$(date +%s)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  BAPR Server Runner — Starting at $(date)"
echo "  Phases: $PHASES_TO_RUN"
echo "═══════════════════════════════════════════════════════════════"

for phase in $PHASES_TO_RUN; do
    case "$phase" in
        1) phase1_multiseed ;;
        2) phase2_ablation ;;
        3) phase3_sensitivity ;;
        4) phase4_ablations_all_envs ;;
        5) phase5_id_ood ;;
        6) phase6_analysis ;;
        *) echo "Unknown phase: $phase" ;;
    esac
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All phases complete!"
echo "  Total time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
echo "  Results in: $SAVE_ROOT/"
echo "  Figures in: paper/"
echo "═══════════════════════════════════════════════════════════════"
