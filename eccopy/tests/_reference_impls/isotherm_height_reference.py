"""Frozen pre-Numba reference implementation, for cross-validation only."""
import numpy as np

def isotherm_height_reference(temp_3d, z_levels, target_temp_c):
    temp_3d = np.asarray(temp_3d, dtype=float)
    z_levels = np.asarray(z_levels, dtype=float)
    nz, ny, nx = temp_3d.shape

    ht_grid = np.full((ny, nx), z_levels[0])

    for iy in range(ny):
        for ix in range(nx):
            col = temp_3d[:, iy, ix]

            valid_iz = np.where(~np.isnan(col))[0]
            if valid_iz.size == 0:
                ht_grid[iy, ix] = np.nan
                continue

            bottom_temp = col[valid_iz[0]]
            bottom_ht = z_levels[valid_iz[0]]
            top_temp = col[valid_iz[-1]]
            top_ht = z_levels[valid_iz[-1]]

            ht_found = False

            for j in range(1, valid_iz.size):
                iz_below = valid_iz[j - 1]
                iz_above = valid_iz[j]
                t_below = col[iz_below]
                t_above = col[iz_above]

                crosses = ((t_below >= target_temp_c >= t_above) or
                           (t_below <= target_temp_c <= t_above))
                if crosses:
                    delta_t = t_above - t_below
                    delta_h = z_levels[iz_above] - z_levels[iz_below]
                    if delta_t == 0:
                        interp_ht = z_levels[iz_below]
                    else:
                        interp_ht = z_levels[iz_below] + (
                            (target_temp_c - t_below) / delta_t
                        ) * delta_h
                    ht_grid[iy, ix] = interp_ht
                    ht_found = True
                    break

            if not ht_found:
                if target_temp_c >= bottom_temp:
                    ht_grid[iy, ix] = bottom_ht
                elif target_temp_c <= top_temp:
                    ht_grid[iy, ix] = top_ht

    return ht_grid
