"""
eccopy.stats — generic post-classification statistics.

Operates on the `echo_type` (and, for height statistics, `height`)
arrays returned by any of eccopy1d / eccopy2d_v / eccopy2d_h / eccopy3d
-- same data-agnostic, array-in/array-out convention as the rest of
EccoPy. Nothing here reads or writes files.

Quick start
-----------
    from eccopy import eccopy3d
    from eccopy import stats

    result = eccopy3d.run(dbz, coords_z=z, coords_y=y, coords_x=x, height=height_km)

    stats.convective_percentage(result.echo_type)      # float, 0-100
    stats.n_clumps(result.echo_type)                    # int
    stats.convective_depth(result.echo_type, height_km) # (Y, X) array
    stats.summarize(result.echo_type, height=height_km) # dict bundle
"""

from .basic_stats import (
    codes_for_category,
    echo_type_fractions,
    convective_percentage,
    stratiform_percentage,
    mixed_percentage,
    n_clumps,
    clump_sizes,
    echo_top_height,
    echo_base_height,
    echo_depth,
    convective_top_height,
    convective_base_height,
    convective_depth,
    stratiform_depth,
    summarize,
)

__all__ = [
    "codes_for_category",
    "echo_type_fractions",
    "convective_percentage",
    "stratiform_percentage",
    "mixed_percentage",
    "n_clumps",
    "clump_sizes",
    "echo_top_height",
    "echo_base_height",
    "echo_depth",
    "convective_top_height",
    "convective_base_height",
    "convective_depth",
    "stratiform_depth",
    "summarize",
]
