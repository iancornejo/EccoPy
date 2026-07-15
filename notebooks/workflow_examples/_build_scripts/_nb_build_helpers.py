"""
Builds the eccopy notebooks/workflow_examples/*.ipynb notebooks with
nbformat. Not shipped as part of the package -- a one-off authoring tool.
Run with: python3 build_notebooks.py
"""
import nbformat as nbf
from pathlib import Path


PATH_SETUP_SOURCE = '''\
import sys
from pathlib import Path

# Walk up from the notebook's location until we find the directory
# containing the `eccopy` package folder, then add it to sys.path.
here = Path.cwd()
for candidate in [here, *here.parents]:
    if (candidate / "eccopy").is_dir():
        sys.path.insert(0, str(candidate))
        break

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.set_printoptions(precision=3, suppress=True)
'''


def make_notebook(cells):
    """cells: list of (kind, source) where kind is 'markdown' or 'code'."""
    nb = nbf.v4.new_notebook()
    out_cells = []
    for kind, source in cells:
        if kind == "markdown":
            out_cells.append(nbf.v4.new_markdown_cell(source))
        else:
            out_cells.append(nbf.v4.new_code_cell(source))
    nb["cells"] = out_cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return nb


def write(nb, path):
    # Anchor to notebooks/workflow_examples/ (the parent of _build_scripts/)
    # rather than the cwd, so the builders write the real notebooks no matter
    # which directory they are invoked from.
    out = Path(__file__).resolve().parent.parent / path
    with open(out, "w") as f:
        nbf.write(nb, f)
    print("wrote", out)
