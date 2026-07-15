"""
Packaging regression tests -- item 5 of the v0.1 checklist.

These guard against two real bugs found while preparing v0.1 for
GitHub/conda release:
  1. The .mat disk-strel/disk-decomp reference data files were silently
     dropped from built wheels (no package-data configuration existed).
  2. `import eccopy` transitively required matplotlib even though it's
     declared as an optional "plot" extra, because core/__init__.py
     imported colormap functions eagerly.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_wheel_includes_mat_data_files(tmp_path):
    """Build a real wheel and confirm the .mat files are inside it --
    the actual bug: package-data wasn't configured, so pip/conda builds
    silently shipped a package that would fail at runtime for anyone
    using enlarge_conv/enlarge_mixed (which load these files).

    Only meaningful when running from a source checkout -- skips
    gracefully when run against an installed package."""
    if not (PKG_ROOT / "pyproject.toml").exists():
        pytest.skip("not running from a source checkout; nothing to build")
    pytest.importorskip("build")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "-o", str(tmp_path)],
        cwd=str(PKG_ROOT), check=True, capture_output=True,
    )
    wheels = list(tmp_path.glob("eccopy-*.whl"))
    assert wheels, "no wheel was built"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    mat_files = [n for n in names if n.endswith(".mat")]
    assert len(mat_files) >= 4, (
        f"expected disk_strel/disk_decomp .mat files in the wheel, found: {mat_files}"
    )
    assert any("disk_strels" in n for n in mat_files)
    assert any("disk_decomp" in n for n in mat_files)


def test_core_package_data_declared_in_pyproject():
    """Lighter-weight guard than actually building a wheel every test
    run: confirm the package-data declaration itself is present and
    covers both .mat directories.

    Only meaningful when running from a source checkout (pyproject.toml
    isn't shipped inside an installed package's site-packages tree) --
    skips gracefully otherwise, e.g. when run via
    `pytest --pyargs eccopy.tests` against an installed copy."""
    pyproject_path = PKG_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        pytest.skip("not running from a source checkout; pyproject.toml not found")
    text = pyproject_path.read_text()
    assert "[tool.setuptools.package-data]" in text
    assert "disk_strels" in text
    assert "disk_decomp" in text


def test_core_getattr_resolves_colormap_names_lazily():
    """core/__init__.py must NOT import matplotlib-dependent colormap
    names at module import time; they should resolve lazily via
    __getattr__ instead (see that module for why)."""
    import eccopy.core as core
    # Accessing a colormap name should work (matplotlib IS installed in
    # the test environment) via the lazy __getattr__ path.
    cmap = core.convectivity_cmap()
    assert cmap is not None
    with pytest.raises(AttributeError):
        core.this_name_does_not_exist


def test_core_getattr_raises_for_unknown_name():
    import eccopy.core as core
    with pytest.raises(AttributeError):
        getattr(core, "not_a_real_attribute")
