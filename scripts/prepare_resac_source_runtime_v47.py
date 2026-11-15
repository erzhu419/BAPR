"""Prepare the dedicated source-trainer runtime on a scheduler GPU node."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jax_experiments.analysis import submission_extensions_protocol_v47 as p


def ensure_torch(python):
    # ENOSPC can leave version metadata behind without the Torch package files.
    probe = subprocess.run([str(python), "-c",
                            "import torch; import torch.nn; import torch._prims_common"])
    if probe.returncode:
        print("Incomplete Torch installation; reinstalling torch==2.4.1", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "--timeout", "120",
                        "--force-reinstall", "--no-deps", "torch==2.4.1"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=("jtl110gpu", "jtl110gpu2"), required=True)
    args = parser.parse_args()
    python = Path(p.SOURCE_PYTHON)
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(python.parents[1])], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--timeout", "120", "-r",
                    str(ROOT / "reproducibility/resac_source_comparison/requirements.txt")], check=True)
    ensure_torch(python)
    output = p.BUNDLE_ROOT / "source/runtime" / args.node / "ready.json"
    probe = (
        "import json; from pathlib import Path; "
        "from jax_experiments.analysis.run_resac_source_comparison_v47 import source_imports, check_runtime; "
        "api=source_imports(); runtime=check_runtime(); "
        f"out=Path({str(output)!r}); out.parent.mkdir(parents=True, exist_ok=True); "
        f"out.write_text(json.dumps(dict(protocol={p.PROTOCOL!r}, node={args.node!r}, "
        "runtime_versions=runtime, torch=api['torch'].__version__, ready=True)))"
    )
    subprocess.run([str(python), "-c", probe], check=True)
    print(f"SOURCE_RUNTIME_READY node={args.node}", flush=True)


if __name__ == "__main__":
    main()
