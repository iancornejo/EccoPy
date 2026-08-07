"""
Structural checks on the shipped workflow notebooks.

These do not execute the notebooks -- that needs the sample data and takes
minutes. They check the committed artefacts for the failure modes that are
invisible until someone opens them on GitHub:

  * a cell that errored during execution
  * a markdown table with a row whose column count does not match the
    header, which makes GitHub-flavoured markdown reject the WHOLE table
    and render it as a run-on paragraph
  * a cell that calls print() or plt.show() but carries no output, which
    means the notebook was committed unexecuted
"""
import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb")) if NOTEBOOK_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(
    not NOTEBOOKS,
    reason="notebooks/ is not present (excluded from the built package)",
)

_SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")


def _load(path):
    return json.loads(path.read_text())


def _columns(row: str) -> int:
    """Cell count of a markdown table row, ignoring the outer pipes."""
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return len(stripped.split("|"))


def _tables(source: str):
    """Yield (header_line_index, lines) for each markdown table in `source`."""
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        is_header = (lines[i].strip().startswith("|")
                     and i + 1 < len(lines)
                     and _SEPARATOR.match(lines[i + 1].strip()))
        if is_header:
            end = i + 2
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            yield i, lines[i:end]
            i = end
        else:
            i += 1


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_error_outputs(path):
    offenders = [
        n for n, cell in enumerate(_load(path)["cells"])
        if any(o.get("output_type") == "error" for o in cell.get("outputs", []))
    ]
    assert not offenders, f"{path.name}: cells with error output: {offenders}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_markdown_tables_are_well_formed(path):
    problems = []
    for n, cell in enumerate(_load(path)["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        for _, table in _tables("".join(cell["source"])):
            header = _columns(table[0])
            if _columns(table[1]) != header:
                problems.append(f"cell {n}: separator has "
                                f"{_columns(table[1])} cols, header has {header}")
            for row in table[2:]:
                if _columns(row) != header:
                    problems.append(f"cell {n}: {_columns(row)} cols vs "
                                    f"{header} in {row.strip()[:60]!r}")
    assert not problems, f"{path.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_was_committed_executed(path):
    """
    Any cell that calls print() or plt.show() must carry output. Cells that
    only assign values legitimately produce nothing, so they are exempt.
    """
    silent = []
    for n, cell in enumerate(_load(path)["cells"]):
        if cell["cell_type"] != "code" or cell.get("outputs"):
            continue
        source = "".join(cell["source"])
        if "print(" in source or "plt.show()" in source:
            silent.append(n)
    assert not silent, (f"{path.name}: cells call print()/plt.show() but have "
                        f"no output (unexecuted?): {silent}")


def test_every_module_has_a_notebook():
    names = {p.name for p in NOTEBOOKS}
    for module in ("eccopy1d", "eccopy2d_v", "eccopy2d_h", "eccopy3d"):
        assert f"{module}_workflow.ipynb" in names, f"missing notebook for {module}"
