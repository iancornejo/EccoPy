"""
Checks on the runnable example scripts.

`examples/example_2d_v.py` once sat broken in the repository for several
releases because it passed a `run()` argument that had been removed, and
nothing exercised it. The fast checks below catch that class of drift;
the slow one actually runs each script.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((REPO / "examples").glob("example_*.py")) if (REPO / "examples").is_dir() else []

pytestmark = pytest.mark.skipif(
    not EXAMPLES,
    reason="examples/ is not present (excluded from the built package)",
)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_parses(path):
    ast.parse(path.read_text())


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_has_a_main_and_a_docstring(path):
    tree = ast.parse(path.read_text())
    assert ast.get_docstring(tree), f"{path.name} has no module docstring"
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "main" in names, f"{path.name} defines no main()"


def test_there_is_one_example_per_module():
    names = {p.name for p in EXAMPLES}
    for module in ("1d", "2d_v", "2d_h", "3d"):
        assert f"example_{module}.py" in names, f"missing example for {module}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_uses_only_arguments_run_actually_accepts(path):
    """
    Every keyword passed to a `<module>.run(...)` call must exist in that
    module's signature. This is the check that would have caught the
    removed `vert_params` argument without executing anything.
    """
    import importlib
    import inspect

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id.startswith("eccopy")):
            continue
        mod = importlib.import_module(f"eccopy.{node.func.value.id}")
        accepted = set(inspect.signature(mod.run).parameters)
        passed = {kw.arg for kw in node.keywords if kw.arg is not None}
        unknown = passed - accepted
        assert not unknown, (
            f"{path.name} calls {node.func.value.id}.run() with unknown "
            f"argument(s): {sorted(unknown)}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_runs_end_to_end(path):
    pytest.importorskip("netCDF4")
    if not (REPO / "notebooks" / "data").is_dir():
        pytest.skip("sample data not present")

    proc = subprocess.run([sys.executable, str(path)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"{path.name} exited {proc.returncode}\n{proc.stderr[-2000:]}"
    )
    assert proc.stdout.strip(), f"{path.name} produced no output"
