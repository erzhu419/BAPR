"""A partially installed Torch must not survive a runtime setup retry."""
from types import SimpleNamespace

from scripts import prepare_resac_source_runtime_v47 as setup


def test_incomplete_torch_reinstalled_without_changing_dependencies(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    setup.ensure_torch("/runtime/bin/python")
    assert len(calls) == 2
    command, kwargs = calls[1]
    assert command[:4] == ["/runtime/bin/python", "-m", "pip", "install"]
    assert "--force-reinstall" in command and "--no-deps" in command
    assert command[-1] == "torch==2.4.1" and kwargs["check"]


def test_healthy_torch_not_reinstalled(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    setup.ensure_torch("/runtime/bin/python")
    assert len(calls) == 1 and "torch._prims_common" in calls[0][-1]
