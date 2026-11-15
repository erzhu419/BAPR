export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
eval "$(conda shell.bash hook)"
conda activate jax-rl
NVIDIA_LIB_DIR=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [ -n "$NVIDIA_LIB_DIR" ]; then
    for d in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
    done
fi
nohup python -u -m jax_experiments.train --algo escp --env Ant-v2 --seed 8 --max_iters 2000 --save_root jax_experiments/results --run_name escp_Ant-v2_8_v4 --backend spring --log_scale_limit 0.75 --changing_period 40000 --varying_params gravity > jax_experiments/logs/escp_Ant-v2_8_v4.log 2>&1 &
echo "Restarted ESCP Ant-v2. PID: \$!"
