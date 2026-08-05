from pathlib import Path

from numba.pycc import CC
from numba import float32, int64, void

from .kernels import (
    n_build_cube_anatomy,
    n_fast_remap,
    n_get_next_head,
    n_get_previous,
    n_head_to_logic,
    numba_calc_transparency_map,
)

cc = CC('kernels_core')
cc.output_dir = str(Path(__file__).resolve().parent)

# Register kernels with explicit AOT signatures.
# Signature format: return_type(argument_type, ...).
# ``float32[:, :, :]`` describes a 3D float32 array; ``int64`` is a scalar.

# 1. Cube anatomy kernel.
sig_anatomy = void(
    float32[:,:],    # ohlcv_logic
    float32[:,:,:],  # out_cube
    int64,           # t_head
    int64,           # p_head
    int64,         # window
    float32,         # max_refraction
    int64,           # ticks_per_bin
    int64            # avg_kline_len
)

# 2. Transparency and accumulated-energy kernel.
sig_transparency = void(float32[:,:,:], int64, int64, int64, int64, int64)

# 3. Price-axis remapping kernel.
sig_remap = void(float32[:,:,:], float32[:,:,:], int64, int64, int64)

@cc.export('n_get_next_head', 'int64(int64, int64, int64)')
def n_get_next_head_export(current_head, offset, capacity):
    return n_get_next_head(current_head, offset, capacity)

@cc.export('n_head_to_logic', 'int64(int64, int64, int64)')
def n_head_to_logic_export(p_head, phys_idx, P_max):
    return n_head_to_logic(p_head, phys_idx, P_max)

@cc.export('n_get_previous', 'int64(int64, int64)')
def n_get_previous_export(current_head, capacity):
    return n_get_previous(current_head, capacity)

@cc.export('n_build_cube_anatomy', sig_anatomy)
def n_build_cube_anatomy_export(ohlcv_logic, out_cube, t_head, p_head, fast_t_window, max_refraction, ticks_per_bin, avg_kline_len):
    n_build_cube_anatomy(ohlcv_logic, out_cube, t_head, p_head, fast_t_window, max_refraction, ticks_per_bin, avg_kline_len)

@cc.export('numba_calc_transparency_map', sig_transparency)
def numba_calc_transparency_map_export(cube, p_start_logic, p_end_logic, p_head, t_head, offset):
    numba_calc_transparency_map(cube, p_start_logic, p_end_logic, p_head, t_head, offset)

@cc.export('n_fast_remap', sig_remap)
def n_fast_remap_export(old_cube, new_cube, p_head, old_min, new_min):
    n_fast_remap(old_cube, new_cube, p_head, old_min, new_min)


if __name__ == "__main__":
    # Build the platform-specific ``kernels_core`` extension in this directory.
    cc.compile()
