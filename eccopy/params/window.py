"""
WindowSpec — physical-unit window sizing for texture calculations.

Generalised to work against per-point spacing ARRAYS (not just 1D
coordinate axes), so a window like WindowSpec((5, 'km')) resolves to a
different number of grid points at every location if the local spacing
varies across the field (e.g. a range-height RHI grid, or a lat/lon grid
converted via haversine to locally-varying km spacing).

Usage
-----
    WindowSpec(7)              # bare pixel radius, unit-agnostic
    WindowSpec((5, 'km'))      # 5 km radius, resolved against a spacing array
    WindowSpec((3, 'minute'))  # 3-minute radius, resolved against a
                                 # time-spacing array (units of seconds)
    WindowSpec((180, 's'))     # equivalent to above

Resolution
----------
    ws.pixel_radius(spacing, index)
        Returns the integer pixel radius at a single grid index, given
        the local spacing array.

    ws.pixel_radius_field(spacing)
        Returns an array of integer pixel radii, one per point, with the
        same shape as `spacing`. This is the typical entry point used
        internally by the texture functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

# Supported physical units, normalised to seconds (time) or metres (length)
_LENGTH_UNITS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "kilometre": 1000.0, "kilometres": 1000.0,
}
_TIME_UNITS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


def _normalise_unit(value: float, unit: str) -> Tuple[float, str]:
    unit_lc = unit.lower()
    if unit_lc in _LENGTH_UNITS:
        return value * _LENGTH_UNITS[unit_lc], "length_m"
    if unit_lc in _TIME_UNITS:
        return value * _TIME_UNITS[unit_lc], "time_s"
    raise ValueError(
        f"Unrecognised unit {unit!r}. Supported length units: "
        f"{sorted(_LENGTH_UNITS)}. Supported time units: {sorted(_TIME_UNITS)}."
    )


@dataclass
class WindowSpec:
    """
    A texture/averaging window specified either as a bare pixel radius
    (int) or a physical size with units (tuple of (value, unit_string)).

    Physical sizes are resolved against a caller-supplied spacing array,
    which must be in the matching base unit: metres for length windows,
    seconds for time windows. EccoPy's public run() functions accept
    spacing in km / seconds and handle this conversion internally, so
    callers normally don't need to think about base units directly.
    """

    size: Union[int, Tuple[float, str]]

    def __post_init__(self):
        if isinstance(self.size, int):
            self._is_pixel = True
            self._base_value = None
            self._base_kind = None
        else:
            self._is_pixel = False
            value, unit = self.size
            self._base_value, self._base_kind = _normalise_unit(float(value), unit)

    @property
    def is_pixel(self) -> bool:
        return self._is_pixel

    @property
    def base_value(self):
        """Physical size in base units (metres for length, seconds for
        time). None if this is a bare-pixel WindowSpec."""
        return self._base_value

    @property
    def base_kind(self):
        """'length_m', 'time_s', or None if this is a bare-pixel
        WindowSpec."""
        return self._base_kind

    def pixel_radius_field(self, spacing: np.ndarray) -> np.ndarray:
        """
        Resolve this window into a per-point integer pixel radius.

        Parameters
        ----------
        spacing : np.ndarray
            Local point-to-point spacing, in the SAME base unit as this
            WindowSpec (metres for length windows, seconds for time
            windows). Any shape.

        Returns
        -------
        radius : np.ndarray of int, same shape as `spacing`
            Number of grid points needed, at each location, to span this
            window's physical size. Minimum value is 0.
        """
        spacing = np.asarray(spacing, dtype=float)
        if self._is_pixel:
            return np.full(spacing.shape, int(self.size), dtype=int)

        with np.errstate(divide="ignore", invalid="ignore"):
            radius = self._base_value / spacing
        radius = np.where(np.isfinite(radius), radius, 0.0)
        return np.maximum(np.round(radius).astype(int), 0)

    def pixel_radius(self, spacing: np.ndarray, index: int) -> int:
        """Resolve this window to a single integer pixel radius at `index`."""
        if self._is_pixel:
            return int(self.size)
        s = float(np.asarray(spacing).reshape(-1)[index])
        if not np.isfinite(s) or s <= 0:
            return 0
        return max(int(round(self._base_value / s)), 0)

    def __repr__(self) -> str:
        if self._is_pixel:
            return f"WindowSpec({self.size})"
        return f"WindowSpec({self.size[0]}, {self.size[1]!r})"
