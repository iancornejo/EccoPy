from .texture import refl_texture_1d, refl_texture_2d
from .convectivity import texture_to_convectivity_linear, texture_to_convectivity_piecewise
from .classification import (class_basic, class_basic_isotropic, class_sub_2d,
                              set_echo_type_3d,
                              CATEGORY_MISSING, CATEGORY_STRATIFORM_LOW,
                              CATEGORY_STRATIFORM_MID, CATEGORY_STRATIFORM_HIGH,
                              CATEGORY_MIXED, CATEGORY_CONVECTIVE_ELEVATED,
                              CATEGORY_CONVECTIVE_SHALLOW, CATEGORY_CONVECTIVE_MID,
                              CATEGORY_CONVECTIVE_DEEP)
from .coords import haversine_distance, latlon_to_xy_spacing, resolve_spacing
from .fill import fill_regions_closest_pixel
from .temperature import isotherm_height, melt_layer_from_temp, broadcast_temp_field
from .debug import (refl_texture_1d_debug, refl_texture_2d_debug,
                    TextureDebug1D, TextureDebug2D)

# colormaps.py imports matplotlib, which is an OPTIONAL dependency
# (pyproject.toml's "plot" extra) -- importing it eagerly here would make
# every `import eccopy` require matplotlib, even for pure classification
# work with no plotting at all. Resolve those names lazily instead (PEP
# 562) so `from eccopy.core import echo_type_cmap` still works when
# matplotlib IS installed, but `import eccopy` alone never requires it.
_COLORMAP_NAMES = {
    "echo_type_cmap", "echo_type_norm",
    "basic_echo_type_cmap", "basic_echo_type_norm",
    "vel_cmap", "remap_echo_type",
    "convectivity_cmap", "convectivity_norm", "draw_window_ring",
    "ECHO_TYPE_LABELS", "BASIC_ECHO_TYPE_LABELS",
    "ECHO_TYPE_TO_IDX", "BASIC_ECHO_TYPE_TO_IDX",
}


def __getattr__(name):
    if name in _COLORMAP_NAMES:
        from . import colormaps
        return getattr(colormaps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "refl_texture_1d", "refl_texture_2d",
    "texture_to_convectivity_linear", "texture_to_convectivity_piecewise",
    "class_basic", "class_basic_isotropic", "class_sub_2d", "set_echo_type_3d",
    "CATEGORY_MISSING", "CATEGORY_STRATIFORM_LOW", "CATEGORY_STRATIFORM_MID",
    "CATEGORY_STRATIFORM_HIGH", "CATEGORY_MIXED",
    "CATEGORY_CONVECTIVE_ELEVATED", "CATEGORY_CONVECTIVE_SHALLOW",
    "CATEGORY_CONVECTIVE_MID", "CATEGORY_CONVECTIVE_DEEP",
    "haversine_distance", "latlon_to_xy_spacing", "resolve_spacing",
    "fill_regions_closest_pixel",
    "isotherm_height", "melt_layer_from_temp", "broadcast_temp_field",
    "echo_type_cmap", "echo_type_norm",
    "basic_echo_type_cmap", "basic_echo_type_norm",
    "vel_cmap", "remap_echo_type",
    "convectivity_cmap", "convectivity_norm", "draw_window_ring",
    "refl_texture_1d_debug", "refl_texture_2d_debug",
    "TextureDebug1D", "TextureDebug2D",
]
