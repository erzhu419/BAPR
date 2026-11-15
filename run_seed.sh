#!/bin/bash
# Unified per-seed launcher for paper experiments.
# Args: ALGO ENV SEED [MAX_ITERS] [DWELL] [TARGET_MODE] [TAG]
# ALGO ∈ {bapr, escp, resac, sac, bapr_no_bocd, bapr_no_rmdm,
#         bapr_no_adapt_beta, bapr_fixed_decay, bad_bapr,
#         bapr_unsupervised}
# TAG (optional): extra label for run_name (e.g. "no2" / "main")
set -e
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d /cm/local/apps/cuda-driver/libs/535.261.03/lib64 ]; then
    export LD_LIBRARY_PATH="/cm/local/apps/cuda-driver/libs/535.261.03/lib64:${LD_LIBRARY_PATH:-}"
fi
if [ -d /cm/local/apps/cuda-driver/libs/535.261.03/bin ]; then
    export PATH="/cm/local/apps/cuda-driver/libs/535.261.03/bin:${PATH:-}"
fi
if [ -n "${BAPR_PYTHON:-}" ]; then
    PYTHON="$BAPR_PYTHON"
else
    for py in \
        /home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python \
        /home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python \
        /tmp/scheduleurm_envs/resac-jax-535-py310-final/bin/python \
        /home/zhengliang01/scheduleurm_work/conda_envs/resac-jax-535-py310-final/bin/python \
        /home/erzhu419/anaconda3/envs/bapr-jax/bin/python \
        /home/erzhu419/miniconda3/envs/bapr-jax/bin/python \
        /home/erzhu419/.conda/envs/bapr-jax/bin/python \
        /home/erzhu419/resac_envs/resac-jax-535-py310-final/bin/python \
        /home/erzhu419/.conda/envs/resac-jax/bin/python \
        /home/erzhu419/miniconda3/envs/resac-jax/bin/python; do
        [ -x "$py" ] && { PYTHON="$py"; break; }
    done
fi
if [ -z "${PYTHON:-}" ]; then
    for cdir in /home/erzhu419/miniconda3 /home/erzhu419/anaconda3 ~/.conda; do
        [ -f "$cdir/etc/profile.d/conda.sh" ] && { source "$cdir/etc/profile.d/conda.sh"; break; }
    done
    conda activate bapr-jax
    PYTHON=python
fi
NVIDIA_LIB_DIR=$("$PYTHON" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [ -n "$NVIDIA_LIB_DIR" ]; then
    for d in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
fi
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

ALGO=$1; ENV=$2; SEED=$3
MAX_ITERS=${4:-1500}; DWELL=${5:-60}; TGT=${6:-independent}; TAG=${7:-paper}
EXTRA_ARGS=("${@:8}")
RUN_NAME="${TAG}_${ALGO}_${ENV%-v2}_dw${DWELL}_s${SEED}"
RUN_DIR="jax_experiments/results_paper/${RUN_NAME}"

cd "${BAPR_ROOT:-$SCRIPT_DIR}"
mkdir -p jax_experiments/results_paper
mkdir -p "${RUN_DIR}/logs"
export TMPDIR="${BAPR_TMPDIR:-$PWD/${RUN_DIR}/tmp}"
mkdir -p "$TMPDIR"

LAUNCHER_LOG="${RUN_DIR}/logs/launcher_diagnostics.log"
LAUNCHER_HEARTBEAT="${RUN_DIR}/logs/launcher_heartbeat.tsv"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

launcher_ts() {
  date -Is 2>/dev/null || date
}

launcher_log() {
  printf '[launcher %s] %s\n' "$(launcher_ts)" "$*"
}

dump_launcher_state() {
  local label=${1:-snapshot}
  {
    printf '[launcher %s] state=%s host=%s pwd=%s pid=%s ppid=%s user=%s\n' \
      "$(launcher_ts)" "$label" "$(hostname 2>/dev/null || echo unknown)" \
      "$PWD" "$$" "$PPID" "${USER:-unknown}"
    printf '[launcher %s] python=%s cuda_visible=%s jax_platforms=%s xla_mem_fraction=%s tmpdir=%s\n' \
      "$(launcher_ts)" "${PYTHON:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}" \
      "${JAX_PLATFORMS:-unset}" "${XLA_PYTHON_CLIENT_MEM_FRACTION:-unset}" \
      "${TMPDIR:-unset}"
    printf '[launcher %s] command_args algo=%s env=%s seed=%s max_iters=%s dwell=%s target=%s tag=%s run_name=%s\n' \
      "$(launcher_ts)" "$ALGO" "$ENV" "$SEED" "$MAX_ITERS" "$DWELL" "$TGT" "$TAG" "$RUN_NAME"
    printf '[launcher %s] resource: free\n' "$(launcher_ts)"
    free -h 2>/dev/null || true
    printf '[launcher %s] resource: df\n' "$(launcher_ts)"
    df -h "$PWD" /tmp "${TMPDIR:-/tmp}" 2>/dev/null || true
    if command -v nvidia-smi >/dev/null 2>&1; then
      printf '[launcher %s] resource: nvidia-smi\n' "$(launcher_ts)"
      nvidia-smi 2>/dev/null || true
    fi
    printf '[launcher %s] limits\n' "$(launcher_ts)"
    ulimit -a 2>/dev/null || true
    printf '[launcher %s] selected env\n' "$(launcher_ts)"
    env | sort | grep -E '^(BAPR_|CUDA|JAX|XLA|OMP|OPENBLAS|MKL|NUMEXPR|TMPDIR|PATH=|LD_LIBRARY_PATH=|USER=|HOSTNAME=)' || true
  } >> "$LAUNCHER_LOG"
}

CHILD_PID=""
MONITOR_PID=""
on_launcher_exit() {
  local rc=$?
  trap - EXIT
  launcher_log "EXIT rc=${rc} child_pid=${CHILD_PID:-none} monitor_pid=${MONITOR_PID:-none}"
  if [ -n "${CHILD_PID:-}" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    launcher_log "child still alive at launcher EXIT; ps follows"
    ps -o pid,ppid,pgid,stat,etime,pcpu,pmem,rss,vsz,cmd -p "$CHILD_PID" 2>/dev/null || true
  fi
  dump_launcher_state "exit_rc_${rc}"
  printf '%s\t%s\t%s\n' "$(launcher_ts)" "$rc" "${CHILD_PID:-none}" > "${RUN_DIR}/logs/launcher_exit.status" 2>/dev/null || true
}

on_launcher_signal() {
  local sig=$1
  launcher_log "received signal ${sig}; forwarding to child_pid=${CHILD_PID:-none}"
  dump_launcher_state "signal_${sig}"
  if [ -n "${CHILD_PID:-}" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill "-${sig}" "$CHILD_PID" 2>/dev/null || kill "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  exit 128
}

trap on_launcher_exit EXIT
trap 'on_launcher_signal TERM' TERM
trap 'on_launcher_signal INT' INT
trap 'on_launcher_signal HUP' HUP

launcher_log "launcher start"
dump_launcher_state "start"
printf 'ts\tchild_pid\talive\tgpu_snapshot\n' > "$LAUNCHER_HEARTBEAT"

# Algo-conditional flags. Redesigned BAPR defaults to context + ensemble mean
# with a surprise-gated recent-replay path. Legacy BOCD/belief conditioning is
# still available via BAPR_ADAPTATION_MODE=legacy or BAPR_BELIEF_MODE=joint.
EXTRA=""
BAPR_PENALTY_SCALE_VALUE="${BAPR_PENALTY_SCALE:-0.5}"
BAPR_ADAPTATION_MODE_VALUE="${BAPR_ADAPTATION_MODE:-gate}"
BAPR_BELIEF_MODE_VALUE="${BAPR_BELIEF_MODE:-}"
if [ -z "$BAPR_BELIEF_MODE_VALUE" ]; then
  if [ "$BAPR_ADAPTATION_MODE_VALUE" = "legacy" ]; then
    BAPR_BELIEF_MODE_VALUE="joint"
  else
    BAPR_BELIEF_MODE_VALUE="none"
  fi
fi
case "$ALGO" in
  bapr|bapr_no_bocd|bapr_no_rmdm|bapr_no_adapt_beta|bapr_fixed_decay|bad_bapr)
    EXTRA="--penalty_scale ${BAPR_PENALTY_SCALE_VALUE} --critic_target_mode ${TGT} --bapr_adaptation_mode ${BAPR_ADAPTATION_MODE_VALUE}"
    if [ "$BAPR_BELIEF_MODE_VALUE" = "joint" ]; then
      EXTRA="$EXTRA --use_regime_belief"
    fi
    if [ "${BAPR_PER_TRANS_BELIEF:-1}" = "0" ]; then
      EXTRA="$EXTRA --no_per_trans_belief"
    fi
    if [ -n "${BAPR_GRAD_CLIP_NORM:-}" ]; then
      EXTRA="$EXTRA --bapr_grad_clip_norm ${BAPR_GRAD_CLIP_NORM}"
    fi
    if [ -n "${BAPR_BETA:-}" ]; then
      EXTRA="$EXTRA --beta ${BAPR_BETA}"
    fi
    if [ -n "${BAPR_ACTOR_OBJECTIVE:-}" ]; then
      EXTRA="$EXTRA --actor_objective ${BAPR_ACTOR_OBJECTIVE}"
    fi
    if [ -n "${BAPR_WEIGHT_REG:-}" ]; then
      EXTRA="$EXTRA --weight_reg ${BAPR_WEIGHT_REG}"
    fi
    if [ -n "${BAPR_BETA_OOD:-}" ]; then
      EXTRA="$EXTRA --beta_ood ${BAPR_BETA_OOD}"
    fi
    if [ "${BAPR_USE_EMA_EVAL:-0}" = "1" ]; then
      EXTRA="$EXTRA --use_ema_eval"
    fi
    if [ "${BAPR_USE_EMA_ROLLOUT:-0}" = "1" ]; then
      EXTRA="$EXTRA --use_ema_rollout"
    fi
    if [ -n "${BAPR_EMA_ROLLOUT_START_ITER:-}" ]; then
      EXTRA="$EXTRA --ema_rollout_start_iter ${BAPR_EMA_ROLLOUT_START_ITER}"
    fi
    if [ "${BAPR_EMA_ROLLOUT_REQUIRE_REG_LATCHED:-0}" = "1" ]; then
      EXTRA="$EXTRA --ema_rollout_require_reg_latched"
    fi
    if [ -n "${BAPR_GATE_WARMUP_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_gate_warmup_iters ${BAPR_GATE_WARMUP_ITERS}"
    fi
    if [ -n "${BAPR_SURPRISE_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_surprise_threshold ${BAPR_SURPRISE_THRESHOLD}"
    fi
    if [ -n "${BAPR_GATE_GAIN:-}" ]; then
      EXTRA="$EXTRA --bapr_gate_gain ${BAPR_GATE_GAIN}"
    fi
    if [ -n "${BAPR_GATE_EMA_ALPHA:-}" ]; then
      EXTRA="$EXTRA --bapr_gate_ema_alpha ${BAPR_GATE_EMA_ALPHA}"
    fi
    if [ -n "${BAPR_GATE_MAX:-}" ]; then
      EXTRA="$EXTRA --bapr_gate_max ${BAPR_GATE_MAX}"
    fi
    if [ -n "${BAPR_RECENT_FRAC_CAP:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_frac_cap ${BAPR_RECENT_FRAC_CAP}"
    fi
    if [ -n "${BAPR_RECENT_FRAC_FLOOR:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_frac_floor ${BAPR_RECENT_FRAC_FLOOR}"
    fi
    if [ "${BAPR_RECENT_TRUE_FLOOR:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_recent_true_floor"
    fi
    if [ -n "${BAPR_RECENT_FLOOR_MODE:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_floor_mode ${BAPR_RECENT_FLOOR_MODE}"
    fi
    if [ -n "${BAPR_RECENT_FLOOR_RATIO_LOW:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_floor_ratio_low ${BAPR_RECENT_FLOOR_RATIO_LOW}"
    fi
    if [ -n "${BAPR_RECENT_FLOOR_RATIO_HIGH:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_floor_ratio_high ${BAPR_RECENT_FLOOR_RATIO_HIGH}"
    fi
    if [ -n "${BAPR_RECENT_FLOOR_MID_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_floor_mid_frac ${BAPR_RECENT_FLOOR_MID_FRAC}"
    fi
    if [ -n "${BAPR_RECENT_FLOOR_EXTREME_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_floor_extreme_frac ${BAPR_RECENT_FLOOR_EXTREME_FRAC}"
    fi
    if [ "${BAPR_RECENT_DISAGREEMENT_GATE:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_recent_disagreement_gate"
    fi
    if [ "${BAPR_RECENT_OPEN_IF_REG_LATCHED:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_recent_open_if_reg_latched"
    fi
    if [ -n "${BAPR_RECENT_QSTD_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_qstd_threshold ${BAPR_RECENT_QSTD_THRESHOLD}"
    fi
    if [ -n "${BAPR_RECENT_QSTD_RATIO_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_recent_qstd_ratio_threshold ${BAPR_RECENT_QSTD_RATIO_THRESHOLD}"
    fi
    if [ "${BAPR_REG_DISAGREEMENT_GATE:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_reg_disagreement_gate"
    fi
    if [ -n "${BAPR_REG_WARMUP_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_warmup_iters ${BAPR_REG_WARMUP_ITERS}"
    fi
    if [ -n "${BAPR_REG_MAX_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_max_iters ${BAPR_REG_MAX_ITERS}"
    fi
    if [ "${BAPR_REG_LATCH:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_reg_latch"
    fi
    if [ "${BAPR_REG_REQUIRE_BOTH:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_reg_require_both"
    fi
    if [ -n "${BAPR_REG_QSTD_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_qstd_threshold ${BAPR_REG_QSTD_THRESHOLD}"
    fi
    if [ -n "${BAPR_REG_QSTD_RATIO_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_qstd_ratio_threshold ${BAPR_REG_QSTD_RATIO_THRESHOLD}"
    fi
    if [ "${BAPR_REG_EMERGENCY_GATE:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_reg_emergency_gate"
    fi
    if [ -n "${BAPR_REG_EMERGENCY_SCALE:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_emergency_scale ${BAPR_REG_EMERGENCY_SCALE}"
    fi
    if [ -n "${BAPR_REG_EMERGENCY_QSTD_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_emergency_qstd_threshold ${BAPR_REG_EMERGENCY_QSTD_THRESHOLD}"
    fi
    if [ -n "${BAPR_REG_EMERGENCY_QSTD_RATIO_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_emergency_qstd_ratio_threshold ${BAPR_REG_EMERGENCY_QSTD_RATIO_THRESHOLD}"
    fi
    if [ -n "${BAPR_REG_LATCHED_SCALE:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_latched_scale ${BAPR_REG_LATCHED_SCALE}"
    fi
    if [ "${BAPR_REG_PERF_COLLAPSE_GATE:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_reg_perf_collapse_gate"
    fi
    if [ -n "${BAPR_REG_PERF_COLLAPSE_DROP_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_perf_collapse_drop_frac ${BAPR_REG_PERF_COLLAPSE_DROP_FRAC}"
    fi
    if [ -n "${BAPR_REG_PERF_COLLAPSE_SCALE:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_perf_collapse_scale ${BAPR_REG_PERF_COLLAPSE_SCALE}"
    fi
    if [ -n "${BAPR_REG_PERF_COLLAPSE_WARMUP_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_reg_perf_collapse_warmup_iters ${BAPR_REG_PERF_COLLAPSE_WARMUP_ITERS}"
    fi
    if [ -n "${BAPR_CONTROLLER_MODE:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_mode ${BAPR_CONTROLLER_MODE}"
    fi
    if [ -n "${BAPR_CONTROLLER_WARMUP_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_warmup_iters ${BAPR_CONTROLLER_WARMUP_ITERS}"
    fi
    if [ -n "${BAPR_CONTROLLER_MIN_PEAK:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_min_peak ${BAPR_CONTROLLER_MIN_PEAK}"
    fi
    if [ -n "${BAPR_CONTROLLER_DROP_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_drop_frac ${BAPR_CONTROLLER_DROP_FRAC}"
    fi
    if [ -n "${BAPR_CONTROLLER_DROP_RAMP:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_drop_ramp ${BAPR_CONTROLLER_DROP_RAMP}"
    fi
    if [ -n "${BAPR_CONTROLLER_SOFT_DECAY:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_soft_decay ${BAPR_CONTROLLER_SOFT_DECAY}"
    fi
    if [ -n "${BAPR_CONTROLLER_SOFT_MIN_SIGNAL:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_soft_min_signal ${BAPR_CONTROLLER_SOFT_MIN_SIGNAL}"
    fi
    if [ -n "${BAPR_CONTROLLER_LOW_QSTD_RATIO:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_low_qstd_ratio ${BAPR_CONTROLLER_LOW_QSTD_RATIO}"
    fi
    if [ -n "${BAPR_CONTROLLER_LOW_QSTD_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_low_qstd_threshold ${BAPR_CONTROLLER_LOW_QSTD_THRESHOLD}"
    fi
    if [ "${BAPR_CONTROLLER_LATCH:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_controller_latch"
    fi
    if [ -n "${BAPR_CONTROLLER_RELEASE_DROP_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_release_drop_frac ${BAPR_CONTROLLER_RELEASE_DROP_FRAC}"
    fi
    if [ -n "${BAPR_CONTROLLER_MAX_ACTIVE_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_max_active_iters ${BAPR_CONTROLLER_MAX_ACTIVE_ITERS}"
    fi
    if [ -n "${BAPR_CONTROLLER_MIN_ACTIVE_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_min_active_iters ${BAPR_CONTROLLER_MIN_ACTIVE_ITERS}"
    fi
    if [ -n "${BAPR_CONTROLLER_EXIT_COOLDOWN_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_exit_cooldown_iters ${BAPR_CONTROLLER_EXIT_COOLDOWN_ITERS}"
    fi
    if [ -n "${BAPR_CONTROLLER_EXIT_SIGNAL_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_exit_signal_threshold ${BAPR_CONTROLLER_EXIT_SIGNAL_THRESHOLD}"
    fi
    if [ -n "${BAPR_CONTROLLER_EXIT_IMPROVE_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_exit_improve_frac ${BAPR_CONTROLLER_EXIT_IMPROVE_FRAC}"
    fi
    if [ -n "${BAPR_CONTROLLER_EXIT_DRAWDOWN_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_exit_drawdown_frac ${BAPR_CONTROLLER_EXIT_DRAWDOWN_FRAC}"
    fi
    if [ -n "${BAPR_CONTROLLER_LATCHED_SIGNAL:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_latched_signal ${BAPR_CONTROLLER_LATCHED_SIGNAL}"
    fi
    if [ -n "${BAPR_CONTROLLER_REG_MULTIPLIER:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_reg_multiplier ${BAPR_CONTROLLER_REG_MULTIPLIER}"
    fi
    if [ -n "${BAPR_CONTROLLER_REG_RECOVER_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_reg_recover_iters ${BAPR_CONTROLLER_REG_RECOVER_ITERS}"
    fi
    if [ -n "${BAPR_CONTROLLER_RECENT_MULTIPLIER:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_recent_multiplier ${BAPR_CONTROLLER_RECENT_MULTIPLIER}"
    fi
    if [ -n "${BAPR_CONTROLLER_RECENT_ADD_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_recent_add_frac ${BAPR_CONTROLLER_RECENT_ADD_FRAC}"
    fi
    if [ -n "${BAPR_CONTROLLER_LCB_MULTIPLIER:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_lcb_multiplier ${BAPR_CONTROLLER_LCB_MULTIPLIER}"
    fi
    if [ -n "${BAPR_CONTROLLER_ACTOR_UPDATE_MULTIPLIER:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_actor_update_multiplier ${BAPR_CONTROLLER_ACTOR_UPDATE_MULTIPLIER}"
    fi
    if [ -n "${BAPR_CONTROLLER_ACTOR_RECOVER_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_controller_actor_recover_iters ${BAPR_CONTROLLER_ACTOR_RECOVER_ITERS}"
    fi
    if [ "${BAPR_ACTOR_LCB_QSTD_GATE:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_qstd_gate"
    fi
    if [ -n "${BAPR_ACTOR_LCB_QSTD_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_qstd_threshold ${BAPR_ACTOR_LCB_QSTD_THRESHOLD}"
    fi
    if [ -n "${BAPR_ACTOR_LCB_QSTD_RATIO_THRESHOLD:-}" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_qstd_ratio_threshold ${BAPR_ACTOR_LCB_QSTD_RATIO_THRESHOLD}"
    fi
    if [ "${BAPR_ACTOR_LCB_REQUIRE_BOTH:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_require_both"
    fi
    if [ -n "${BAPR_ACTOR_LCB_SCALE:-}" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_scale ${BAPR_ACTOR_LCB_SCALE}"
    fi
    if [ "${BAPR_ACTOR_LCB_PERF_GATE:-0}" = "1" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_perf_gate"
    fi
    if [ -n "${BAPR_ACTOR_LCB_PERF_WARMUP_ITERS:-}" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_perf_warmup_iters ${BAPR_ACTOR_LCB_PERF_WARMUP_ITERS}"
    fi
    if [ -n "${BAPR_ACTOR_LCB_PERF_DROP_FRAC:-}" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_perf_drop_frac ${BAPR_ACTOR_LCB_PERF_DROP_FRAC}"
    fi
    if [ -n "${BAPR_ACTOR_LCB_PERF_MIN_PEAK:-}" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_perf_min_peak ${BAPR_ACTOR_LCB_PERF_MIN_PEAK}"
    fi
    if [ -n "${BAPR_ACTOR_LCB_PERF_EMA_ALPHA:-}" ]; then
      EXTRA="$EXTRA --bapr_actor_lcb_perf_ema_alpha ${BAPR_ACTOR_LCB_PERF_EMA_ALPHA}"
    fi
    if [ -n "${BAPR_RESIDUAL_DELTA:-}" ]; then
      EXTRA="$EXTRA --bapr_residual_delta ${BAPR_RESIDUAL_DELTA}"
    fi
    if [ -n "${BAPR_RESIDUAL_GATE_SCALE:-}" ]; then
      EXTRA="$EXTRA --bapr_residual_gate_scale ${BAPR_RESIDUAL_GATE_SCALE}"
    fi
    if [ -n "${BAPR_RESIDUAL_ADV_MARGIN:-}" ]; then
      EXTRA="$EXTRA --bapr_residual_adv_margin ${BAPR_RESIDUAL_ADV_MARGIN}"
    fi
    if [ -n "${BAPR_RESIDUAL_ADV_TEMP:-}" ]; then
      EXTRA="$EXTRA --bapr_residual_adv_temp ${BAPR_RESIDUAL_ADV_TEMP}"
    fi
    if [ -n "${BAPR_RESIDUAL_QSTD_SCALE:-}" ]; then
      EXTRA="$EXTRA --bapr_residual_qstd_scale ${BAPR_RESIDUAL_QSTD_SCALE}"
    fi
    if [ -n "${BAPR_RESIDUAL_BEHAVIOR_WEIGHT:-}" ]; then
      EXTRA="$EXTRA --bapr_residual_behavior_weight ${BAPR_RESIDUAL_BEHAVIOR_WEIGHT}"
    fi
    if [ -n "${BAPR_RESIDUAL_ACTION_PENALTY:-}" ]; then
      EXTRA="$EXTRA --bapr_residual_action_penalty ${BAPR_RESIDUAL_ACTION_PENALTY}"
    fi
    if [ "${BAPR_BELIEF_CONDITIONED:-1}" = "0" ]; then
      EXTRA="$EXTRA --no_belief_conditioning"
    fi
    ;;
  bapr_unsupervised)
    EXTRA="--penalty_scale ${BAPR_PENALTY_SCALE_VALUE} --critic_target_mode ${TGT}"
    ;;
  escp|resac|sac)
    EXTRA=""  # default config; no BAPR-specific flags
    ;;
esac
if [ "${ORACLE_RESET:-0}" = "1" ]; then
  EXTRA="$EXTRA --oracle_reset_on_switch"
fi

launcher_log "python version: $("$PYTHON" --version 2>&1 || true)"
launcher_log "train command: $PYTHON -u -m jax_experiments.train --algo $ALGO --seed $SEED --max_iters $MAX_ITERS --env $ENV --run_name $RUN_NAME"

set +e
"$PYTHON" -u -m jax_experiments.train \
  --algo "$ALGO" --seed "$SEED" --max_iters "$MAX_ITERS" \
  --log_interval 5 --eval_episodes 5 --save_interval 100 \
  --save_root jax_experiments/results_paper --backend spring \
  --env "$ENV" --env_type discrete_mode --mean_dwell_iters "$DWELL" \
  $EXTRA \
  "${EXTRA_ARGS[@]}" \
  --resume \
  --run_name "$RUN_NAME" &
CHILD_PID=$!
launcher_log "python child started pid=${CHILD_PID}"

(
  while kill -0 "$CHILD_PID" 2>/dev/null; do
    gpu_line="nvidia-smi-unavailable"
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_line="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
    fi
    printf '%s\t%s\t1\t%s\n' "$(launcher_ts)" "$CHILD_PID" "$gpu_line" >> "$LAUNCHER_HEARTBEAT"
    sleep "${BAPR_LAUNCHER_HEARTBEAT_SEC:-60}"
  done
  printf '%s\t%s\t0\tchild-exited\n' "$(launcher_ts)" "$CHILD_PID" >> "$LAUNCHER_HEARTBEAT"
) &
MONITOR_PID=$!

wait "$CHILD_PID"
status=$?
if [ -n "${MONITOR_PID:-}" ]; then
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
fi
set -e
launcher_log "python child exited status=${status}"
dump_launcher_state "python_exit_${status}"
if [ "$status" -eq 0 ]; then
  printf 'done run=%s\n' "$RUN_NAME" > "${RUN_DIR}/done.ok"
fi
exit "$status"
