#!/bin/bash
# Next focused redesign sweep after v14-v16.
#
# Rationale from completed local runs:
#   - v16 wreg=0.005 + warm0 + latch250 improved HC/Hopper.
#   - wreg=0.01 destabilized Ant.
#   - Ant needs a separate low-regularization protection branch.
#
# Usage:
#   ./run_redesign_next.sh [MAX_PARALLEL] [MAX_ITERS] [DWELL]
#
# Optional env:
#   SEEDS="0 1 2 3 4"
#   ENVS="Ant-v2 HalfCheetah-v2 Hopper-v2 Walker2d-v2"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BAPR_ROOT:-$SCRIPT_DIR}"

MAX_PARALLEL="${1:-2}"
MAX_ITERS="${2:-1500}"
DWELL="${3:-60}"
SEEDS="${SEEDS:-0 1 2}"
ENVS="${ENVS:-Ant-v2 HalfCheetah-v2 Hopper-v2 Walker2d-v2}"

mkdir -p jax_experiments/results_paper jax_experiments/logs

running_jobs() {
  jobs -r | wc -l
}

wait_for_slot() {
  while [ "$(running_jobs)" -ge "$MAX_PARALLEL" ]; do
    sleep 20
  done
}

launch_if_needed() {
  local tag="$1"
  local env_name="$2"
  local seed="$3"
  shift 3

  local run_name="${tag}_bapr_${env_name%-v2}_dw${DWELL}_s${seed}"
  local run_dir="jax_experiments/results_paper/${run_name}"
  local log_file="jax_experiments/logs/${run_name}.log"

  if [ -f "${run_dir}/done.ok" ]; then
    echo "=== skip done: ${run_name}"
    return 0
  fi

  echo ">>> launch: ${run_name}"
  (
    export "$@"
    ./run_seed.sh bapr "$env_name" "$seed" "$MAX_ITERS" "$DWELL" independent "$tag" \
      > "$log_file" 2>&1
  ) &
  sleep 3
  wait_for_slot
}

for seed in $SEEDS; do
  for env_name in $ENVS; do
    case "$env_name" in
      Ant-v2)
        # Ant was hurt by stronger regularization in v16. Keep latch/warm0 but
        # make regularization gentler and cap recent replay.
        launch_if_needed \
          redesign_gate_v17_ant_guard "$env_name" "$seed" \
          BAPR_WEIGHT_REG=0.003 \
          BAPR_BETA_OOD=0.003 \
          BAPR_REG_DISAGREEMENT_GATE=1 \
          BAPR_REG_WARMUP_ITERS=0 \
          BAPR_REG_MAX_ITERS=250 \
          BAPR_REG_LATCH=1 \
          BAPR_REG_REQUIRE_BOTH=1 \
          BAPR_REG_QSTD_THRESHOLD=5.0 \
          BAPR_REG_QSTD_RATIO_THRESHOLD=0.02 \
          BAPR_RECENT_FRAC_CAP=0.25
        ;;
      *)
        # Best broad candidate from v16: low regularization, immediate gated
        # latch, independent targets, mean actor.
        launch_if_needed \
          redesign_gate_v17_wreg005_warm0_latch250 "$env_name" "$seed" \
          BAPR_WEIGHT_REG=0.005 \
          BAPR_BETA_OOD=0.005 \
          BAPR_REG_DISAGREEMENT_GATE=1 \
          BAPR_REG_WARMUP_ITERS=0 \
          BAPR_REG_MAX_ITERS=250 \
          BAPR_REG_LATCH=1 \
          BAPR_REG_REQUIRE_BOTH=1 \
          BAPR_REG_QSTD_THRESHOLD=5.0 \
          BAPR_REG_QSTD_RATIO_THRESHOLD=0.02
        ;;
    esac
  done
done

wait
echo "Done."
