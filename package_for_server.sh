#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  Package BAPR code for server deployment.
#  Produces: bapr_server.tar.gz (~1-2 MB, code only, no results)
# ═══════════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

OUT="${1:-bapr_server.tar.gz}"

# Create deterministic tarball of source only
tar --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='results' \
    --exclude='logs' \
    --exclude='paper' \
    --exclude='*.log' \
    --exclude='bapr_experiments' \
    --exclude='run_priority.log' \
    --exclude='run_all_output.log' \
    --exclude='restart_*.sh' \
    --exclude='*.tar.gz' \
    --exclude='run_walker_*.sh' \
    --exclude='run_ant_*.sh' \
    --exclude='jax_experiments/sensitivity_runs' \
    --exclude='jax_experiments/jax_experiments' \
    --transform 's,^,bapr_server/,' \
    -czf "$OUT" \
    jax_experiments/ \
    requirements.txt \
    setup_env.sh \
    README_SERVER.md 2>/dev/null

echo "═══════════════════════════════════════════════════════════════"
echo "  Package created: $OUT"
du -sh "$OUT"
echo ""
echo "Contents:"
tar -tzf "$OUT" | head -30
echo "..."
tar -tzf "$OUT" | wc -l | xargs echo "  Total files:"
echo ""
echo "Upload to server:"
echo "    scp $OUT user@server:~/"
echo ""
echo "On server (one-liner):"
echo "    tar xzf $OUT && cd bapr_server && bash setup_env.sh && bash jax_experiments/run_server.sh"
echo "═══════════════════════════════════════════════════════════════"
