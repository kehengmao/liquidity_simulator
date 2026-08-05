
from enum import IntEnum, unique, auto

import numpy as np
from numba import njit

# ==========================================
# Core feature indices exposed to LiquidEngine.
# ==========================================
@unique
class TPChannel(IntEnum):
    # ==========================================
    # 1. Geometry channels (0-10): candle anatomy.
    # ==========================================
    VOLUME_PER_TICK       = 0   # Volume distributed at each price point.
    OPEN_PT      = auto()   # One at the opening-price tick.
    CLOSE_PT     = auto()   # One at the closing-price tick.
    HIGH_PT      = auto()   # One at the high-price tick.
    LOW_PT       = auto()   # One at the low-price tick.
    SOLID_BODY   = auto()   # Range between open and close, excluding shadows.
    UPPER_SHADOW = auto()   # Range above the body through the high.
    LOWER_SHADOW = auto()   # Range below the body through the low.
    ALL_SHADOWS  = auto()   # Union of upper and lower shadows.
    ALL_BODY     = auto()   # Complete high-to-low range.
    VOLUME_PER_TICK_CACHE = auto()  # Per-candle volume divided by its price span.

    # ==========================================
    # 2. Field channels (11-12): optical attenuation model.
    # ==========================================
    REFRACTION   = auto()  # Resistance coefficient derived from volume density.
    TRANSPARENCY = auto()  # Remaining transmission through historical candles.

    # ==========================================
    # 3. Raw energy flow (13-17): initial energy mapping.
    # ==========================================
    BODY_ENERGY      = auto()  # Base energy distributed across the high-low range.
    OPEN_ENERGY      = auto()  # Impulse centered at the open.
    CLOSE_ENERGY     = auto()  # Impulse centered at the close.
    HIGH_ENERGY      = auto()  # Pressure centered at the high.
    LOW_ENERGY       = auto()  # Support centered at the low.

    # ==========================================
    # 4. Refracted energy (18-23): signals filtered by the field.
    # ==========================================
    BODY_REFR = auto()  # Body energy remaining after attenuation.
    OPEN_REFR = auto()  # Filtered open impulse.
    CLOSE_REFR= auto()  # Filtered close impulse.
    HIGH_REFR = auto()  # Filtered high pressure.
    LOW_REFR  = auto()  # Filtered low support.

    TOTAL_REFR_ENERGY= auto()  # Aggregate field strength across refracted channels.

    TIME_VOL         = auto()  # Total volume at this time step.
    TIME_LEN         = auto()  # High-low span at this time step.
    TIME_ENERGY = auto()  # Volume multiplied by price span.


PI_f32 = np.float32(3.141592653589793)

_test = False

@njit(inline='always')
def _n_reset_physical_frame(cube: np.ndarray, i_real: int):
    """
    Zero a physical time frame before reusing it in the circular buffer.
    """
    # ``fill`` is the most efficient Numba-compatible reset here.
    cube[i_real, :, :].fill(0.0)


@njit(inline='always')
def _n_batch_sucession(cube: np.ndarray, t_head: int, capacity: int, features: int | slice):
    """Copy selected channels from the previous time frame."""
    t_prev = n_get_previous(t_head, capacity)
    cube[t_head, :, features] = cube[t_prev, :, features]

@njit(inline='always')
def _n_clear_pricewise_history(cube: np.ndarray, p_real: int, features: int | slice, default_value: float):
    """Reset selected features across all history at one physical price."""
    cube[:, p_real, features].fill(default_value)

@njit(inline='always')
def _n_batch_expansion(cube: np.ndarray, t_head: int, offset: int, capacity: int, features: int | slice):
    """
    Copy the current result into the next ``offset - 1`` physical frames.
    """
    if offset <= 1:
        return

    # Extract the source [price, feature] slice.
    source_data = cube[t_head, :, features]

    for k in range(1, offset):
        t_fill = n_get_next_head(t_head, k, capacity)
        # Write all selected prices and features at once.
        cube[t_fill, :, features] = source_data


@njit(inline='always')
def _n_distribute_energy_clean(out_cube, t_idx, p_idx, channel_idx, total_energy, kernel, p_head, P_max):
    k_size = len(kernel)
    k_radius = k_size // 2

    # Convert the physical energy center to a logical position.
    center_logic_pos = n_head_to_logic(p_head, p_idx, P_max)

    # The logical kernel start may be negative; bounds below clip it.
    start_logic_idx = center_logic_pos - k_radius

    for i in range(k_size):
        l_idx = start_logic_idx + i
        # Map and write only logical indices inside the window.
        if 0 <= l_idx < P_max:
            p_target = n_get_next_head(p_head, l_idx, P_max)
            out_cube[t_idx, p_target, channel_idx] += total_energy * kernel[i]


@njit
def _n_get_blackman_window(target_size):
    dtype = np.float32
    if target_size <= 0:
        return np.zeros(0, dtype=dtype)

    if target_size == 1:
        return np.ones(1, dtype=dtype)

    # Keep the complete window calculation in float32.
    i = np.arange(target_size, dtype=dtype)
    denom = np.float32(target_size - 1)

    # Explicit float32 constants prevent implicit promotion to float64.
    pi_val = np.float32(np.pi)
    a0 = np.float32(0.42)
    a1 = np.float32(0.5)
    a2 = np.float32(0.08)

    # Evaluate the Blackman window in float32.
    res = a0 - a1 * np.cos(np.float32(2.0) * pi_val * i / denom) + \
               a2 * np.cos(np.float32(4.0) * pi_val * i / denom)

    # Normalize the peak to one.
    max_val = np.max(res)
    if max_val != np.float32(0):
        res = res / max_val

    return res.astype(dtype)  # Keep the return dtype explicit.

@njit(inline='always')
def n_get_next_head(current_head: int, offset: int, capacity: int) -> int:
    """Return a circular-buffer position at the requested offset."""
    return (current_head + offset + capacity) % capacity

@njit(inline='always')
def n_head_to_logic(p_head: int, phys_idx: int, P_max: int) -> int:
    """
    Map a physical price index back to its logical offset.
    """
    return (phys_idx - p_head + P_max) % P_max

@njit(inline='always')
def n_get_previous(current_head: int, capacity: int) -> int:
    """Return the previous circular-buffer position."""
    return n_get_next_head(current_head, -1, capacity)


# @njit
# @njit(nogil=True, parallel=True, cache=False)
@njit
def n_build_cube_anatomy(
    ohlcv_logic: np.ndarray,
    out_cube: np.ndarray,
    t_head: int,
    p_head: int,
    fast_t_window: int,
    max_refraction: float,
    ticks_per_bin: int,
    avg_kline_len: int
    ):
    """
    Build the time-price feature cube.

    Candles are mapped into a ``(time, price, channel)`` tensor. The two
    circular-buffer heads provide O(1) rolling updates without shifting memory.
    OHLC points, shadows, and bodies become spatial features; volume is spread
    over the candle range; and a sine response produces nonlinear attenuation.
    """
    N = ohlcv_logic.shape[0]

    T_max, P_max, C_num = out_cube.shape

    for i in range(N):

        # 1. Resolve and clear the physical time frame.
        i_real = n_get_next_head(t_head, i, T_max)

        # Remove all data left by the previous circular-buffer cycle.
        _n_reset_physical_frame(out_cube, i_real)

        # Unpack OHLCV in logical price coordinates.
        o_l = ohlcv_logic[i, 0]
        h_l = ohlcv_logic[i, 1]
        l_l = ohlcv_logic[i, 2]
        c_l = ohlcv_logic[i, 3]
        v   = ohlcv_logic[i, 4]

        # Time-level aggregate features.
        num_ticks = h_l - l_l + 1
        interaction = v * num_ticks

        # Use an odd kernel length so the impulse has a unique center.
        raw_len = int(round(ticks_per_bin * num_ticks / avg_kline_len))
        local_kernel_len = max(1, raw_len if raw_len % 2 != 0 else raw_len + 1)

        local_kernel = _n_get_blackman_window(local_kernel_len)

        # Store time-level features at a fixed price-axis location.
        out_cube[i_real, 0, TPChannel.TIME_VOL.value] = v
        out_cube[i_real, 0, TPChannel.TIME_LEN.value] = num_ticks
        out_cube[i_real, 0, TPChannel.TIME_ENERGY.value] = interaction
        out_cube[i_real, 0, TPChannel.VOLUME_PER_TICK_CACHE.value] = v / num_ticks if num_ticks > 0 else 0

        # 2. Fill high-low coverage and per-tick volume.
        if num_ticks > 0:
            v_per_tick = out_cube[i_real, 0, TPChannel.VOLUME_PER_TICK_CACHE.value]
            start_p = n_get_next_head(p_head, int(l_l), P_max)

            # Split writes that cross the physical circular-buffer boundary.
            body_energy = v * ticks_per_bin

            end1 = int(start_p + num_ticks)
            if end1 <= P_max:
                out_cube[i_real, start_p : end1, TPChannel.VOLUME_PER_TICK.value] = v_per_tick
                out_cube[i_real, start_p : end1, TPChannel.ALL_BODY.value] = 1.0
                out_cube[i_real, start_p : end1, TPChannel.BODY_ENERGY.value] += body_energy
            else:
                first_len = P_max - start_p
                # Segment 1: start through the physical end.
                out_cube[i_real, start_p : P_max, TPChannel.VOLUME_PER_TICK.value] = v_per_tick
                out_cube[i_real, start_p : P_max, TPChannel.ALL_BODY.value] = 1.0
                out_cube[i_real, start_p : P_max, TPChannel.BODY_ENERGY.value] += body_energy
                # Segment 2: wrap around to physical index zero.
                end_idx = int(num_ticks - first_len)
                out_cube[i_real, 0 : end_idx, TPChannel.VOLUME_PER_TICK.value] = v_per_tick
                out_cube[i_real, 0 : end_idx, TPChannel.ALL_BODY.value] = 1.0
                out_cube[i_real, 0 : end_idx, TPChannel.BODY_ENERGY.value] += body_energy

        # 3. Mark OHLC points and distribute intensity energy.
        p_o = n_get_next_head(p_head, int(o_l), P_max)
        p_c = n_get_next_head(p_head, int(c_l), P_max)
        p_h = n_get_next_head(p_head, int(h_l), P_max)
        p_l = n_get_next_head(p_head, int(l_l), P_max)

        # Mark point locations.
        out_cube[i_real, p_o, TPChannel.OPEN_PT.value] = 1.0
        out_cube[i_real, p_c, TPChannel.CLOSE_PT.value] = 1.0
        out_cube[i_real, p_h, TPChannel.HIGH_PT.value] = 1.0
        out_cube[i_real, p_l, TPChannel.LOW_PT.value] = 1.0

        # Distribute enhanced energy around OHLC points.

        _n_distribute_energy_clean(out_cube, i_real, p_h, TPChannel.HIGH_ENERGY.value, interaction, local_kernel, p_head, P_max)
        _n_distribute_energy_clean(out_cube, i_real, p_l, TPChannel.LOW_ENERGY.value, interaction, local_kernel, p_head, P_max)

        if p_o != p_h or p_o != p_l:
            _n_distribute_energy_clean(out_cube, i_real, p_o, TPChannel.OPEN_ENERGY.value, interaction/2, local_kernel, p_head, P_max)

        if p_c != p_h or p_c != p_l:
            _n_distribute_energy_clean(out_cube, i_real, p_c, TPChannel.CLOSE_ENERGY.value, interaction/2, local_kernel, p_head, P_max)

        # 4. Fill the solid candle body.
        b_min, b_max = (o_l, c_l) if o_l < c_l else (c_l, o_l)
        num_b = int(b_max - b_min + 1)
        start_b = n_get_next_head(p_head, int(b_min), P_max)

        if start_b + num_b <= P_max:
            out_cube[i_real, start_b : start_b + num_b, TPChannel.SOLID_BODY.value] = 1.0
        else:
            f_len = P_max - start_b
            out_cube[i_real, start_b : P_max, TPChannel.SOLID_BODY.value] = 1.0
            out_cube[i_real, 0 : num_b - f_len, TPChannel.SOLID_BODY.value] = 1.0

        # 5. Fill candle shadows.
        # Upper shadow: top of the body through the high.
        if h_l > b_max:
            num_up = int(h_l - b_max)
            s_up = n_get_next_head(p_head, int(b_max + 1), P_max)
            end2 = int(s_up + num_up)
            if end2 <= P_max:
                out_cube[i_real, s_up : end2, TPChannel.UPPER_SHADOW.value] = 1.0
                out_cube[i_real, s_up : end2, TPChannel.ALL_SHADOWS.value] = 1.0
            else:
                f = P_max - s_up
                end3 = int(num_up - f)

                out_cube[i_real, s_up : P_max, TPChannel.UPPER_SHADOW.value] = 1.0
                out_cube[i_real, s_up : P_max, TPChannel.ALL_SHADOWS.value] = 1.0
                out_cube[i_real, 0 : end3, TPChannel.UPPER_SHADOW.value] = 1.0
                out_cube[i_real, 0 : end3, TPChannel.ALL_SHADOWS.value] = 1.0

        # Lower shadow: low through the bottom of the body.
        if l_l < b_min:
            num_low = int(b_min - l_l)
            s_low = n_get_next_head(p_head, int(l_l), P_max)
            end4 = int(s_low + num_low)
            if end4 <= P_max:
                out_cube[i_real, s_low : end4, TPChannel.LOWER_SHADOW.value] = 1.0
                out_cube[i_real, s_low : end4, TPChannel.ALL_SHADOWS.value] = 1.0
            else:
                # Recalculate the lower-shadow offset at the physical boundary.
                f_low = P_max - s_low
                end5 = int(num_low - f_low)
                out_cube[i_real, s_low : P_max, TPChannel.LOWER_SHADOW.value] = 1.0
                out_cube[i_real, s_low : P_max, TPChannel.ALL_SHADOWS.value] = 1.0
                out_cube[i_real, 0 : end5, TPChannel.LOWER_SHADOW.value] = 1.0
                out_cube[i_real, 0 : end5, TPChannel.ALL_SHADOWS.value] = 1.0

        # if _test and i == 0:  # Inspect only the first frame.
        #     print("\n--- [DEBUG DEEP DIVE] Frame Index 0 (i_real:", i_real, ") ---")

        #     # Select important channel groups for inspection.
        #     geometry_channels = [
        #         ("OPEN_PT", TPChannel.OPEN_PT.value),
        #         ("CLOSE_PT", TPChannel.CLOSE_PT.value),
        #         ("SOLID_BODY", TPChannel.SOLID_BODY.value),
        #         ("UPPER_SHADOW", TPChannel.UPPER_SHADOW.value),
        #         ("LOWER_SHADOW", TPChannel.LOWER_SHADOW.value)
        #     ]

        #     energy_channels = [
        #         ("TIME_VOL", TPChannel.TIME_VOL.value),
        #         ("VOLUME_PER_TICK", TPChannel.VOLUME_PER_TICK.value),
        #         ("BODY_ENERGY", TPChannel.BODY_ENERGY.value),
        #         ("REFRACTION", TPChannel.REFRACTION.value)
        #     ]

        #     # Print active geometry indices.
        #     print(">> Geometry Activation (Indices with 1.0):")
        #     for name, ch in geometry_channels:
        #         active_indices = np.where(out_cube[i_real, :, ch] > 0)[0]
        #         print("  -", name, ":", active_indices)


        #     # Cross-check the mapped OHLC points.
        #     p_indices = [p_o, p_h, p_l, p_c]
        #     p_names = ["Open", "High", "Low", "Close"]
        #     print(">> OHLC Point Values (Physical Mapping):")
        #     for name, p_idx in zip(p_names, p_indices):
        #         v_at_p = out_cube[i_real, p_idx, TPChannel.VOLUME_PER_TICK.value]
        #         print("  -", name, "at p-index", p_idx, "Volume_Tick:", v_at_p)

        #     print("--- [DEBUG END] ---\n")

    if fast_t_window> T_max:
        fast_t_window = T_max
    t_now = n_get_next_head(t_head, T_max - 1, T_max)
    t_start = (t_now - fast_t_window + 1)  # Physical start before wrapping.
    end6 = int(t_now + 1)
    if t_start >= 0:
        max_v_per_tick = out_cube[t_start : end6, 0, TPChannel.VOLUME_PER_TICK_CACHE.value].max()
    else:
        max_v_per_tick1 = out_cube[t_start + T_max : T_max, 0, TPChannel.VOLUME_PER_TICK_CACHE.value].max()
        max_v_per_tick2 = out_cube[0 : end6, 0, TPChannel.VOLUME_PER_TICK_CACHE.value].max()

        max_v_per_tick = max(max_v_per_tick1, max_v_per_tick2)

    for i in range(N):
        i_real = n_get_next_head(t_head, i, T_max)
        o_l, h_l, l_l, c_l, v = ohlcv_logic[i]
        # 6. Calculate normalized attenuation.
        p_start = int(l_l)
        p_end = int(h_l)

        # Precompute the normalization factor to avoid division in the loop.
        inv_max_v = 1.0 / max_v_per_tick if max_v_per_tick > 0 else 0.0

        for j in range(p_start, p_end + 1):
            # ``j`` is already a logical price offset.
            p = n_get_next_head(p_head, j, P_max)

            # Read the per-tick volume written above.
            raw_v = out_cube[i_real, p, TPChannel.VOLUME_PER_TICK.value]
            norm_v = raw_v * inv_max_v

            if _test and i == 0 and j == p_start:
                print("--- [TEST C] Refraction Sample ---")
                print("Raw_v at p:", raw_v, "Norm_v:", norm_v, "Inv_max_v:", inv_max_v)

            # Apply the nonlinear response only inside the active range.
            if 0.001 < norm_v <= 1.0:
                out_cube[i_real, p, TPChannel.REFRACTION.value] = np.sin(PI_f32 * np.float32(norm_v)) * max_refraction
            else:
                out_cube[i_real, p, TPChannel.REFRACTION.value] = 0.0

    # if _test:
    #     print("--- [TEST B] Normalization ---")
    #     print("Max V per Tick in fast_t_window:", max_v_per_tick)
    #     # Print the distribution of energy channels.
    #     print(">> Energy/Value Distribution:")
    #     for name, ch in energy_channels:
    #         # Find the maximum and nonzero mean for each channel.
    #         data = out_cube[i_real, :, ch]
    #         non_zero = data[data > 0]
    #         if len(non_zero) > 0:
    #             print("  -", name, "-> Max:", data.max(), "Avg(non-zero):", non_zero.mean(), "Count:", len(non_zero))
    #         else:
    #             print("  -", name, "-> ALL ZERO!")


    #     Extract the REFRACTION channel across the complete cube.
    #     refraction_data = out_cube[:, :, TPChannel.REFRACTION.value]
    #     max_ref = refraction_data.max()
    #     min_ref = refraction_data.min()
    #     non_zero_count = np.count_nonzero(refraction_data)

    #     print("\n=== [FINAL GLOBAL CHECK: REFRACTION] ===")
    #     print("  - Data Shape checked:", refraction_data.shape)
    #     print("  - Max Value in Channel:", max_ref)
    #     print("  - Min Value in Channel:", min_ref)
    #     print("  - Non-zero Element Count:", non_zero_count)

    #     if non_zero_count > 0:
    #         # Print a few nonzero coordinates to locate active frames.
    #         indices = np.argwhere(refraction_data > 0)
    #         print("  - First 5 non-zero samples (t, p):")
    #         for idx in range(min(5, len(indices))):
    #             t_idx, p_idx = indices[idx]
    #             print("    t:", t_idx, "p:", p_idx, "val:", refraction_data[t_idx, p_idx])
    #     else:
    #         print("  - [ALERT]: REFRACTION channel is COMPLETELY EMPTY (all zeros).")

    #     print("--- [TEST END] Build Complete ---")

@njit(nogil=True, cache=False)
def numba_calc_transparency_map(cube: np.ndarray, p_start_logic: int, p_end_logic: int, p_head: int,
                                 t_head: int, offset: int):
    """
    Update transparency and accumulated energy by tracing through history.

    Each price is treated as an independent ray traced backward from the
    current time head. Candle bodies attenuate the ray according to
    ``REFRACTION`` and historical energy accumulates as
    ``target += historical * transparency``. Scanning stops early when the
    remaining transparency is negligible. The function updates the
    ``TRANSPARENCY`` and ``TOTAL_REFR_ENERGY`` channels in place.
    """

    T_max = cube.shape[0]
    P_max = cube.shape[1]

    # 1. Compute an inclusive logical price span.
    num_p = min(p_end_logic - p_start_logic + 1, P_max)

    _n_batch_sucession(cube, t_head, T_max, TPChannel.TOTAL_REFR_ENERGY.value)
    _n_batch_sucession(cube, t_head, T_max, slice(TPChannel.BODY_REFR.value,TPChannel.LOW_REFR.value+1) )

    t_logical_now = n_get_next_head(t_head, offset, T_max)
    t_logical_now = n_get_previous(t_logical_now, T_max)

    # 2. Inherit persistent state and clear the active price range.
    for j in range(num_p):
        # Physical price combines the head, logical start, and local offset.
        p_logic = p_start_logic + j
        p = n_get_next_head(p_head, p_logic, P_max)
        _n_clear_pricewise_history(cube, p, TPChannel.TRANSPARENCY.value, 1.)

        cube[t_head, p, TPChannel.BODY_REFR.value : TPChannel.LOW_REFR.value+1] = 0.0
        cube[t_head, p, TPChannel.TOTAL_REFR_ENERGY.value] = 0.0

    # 3. Trace each price backward through time.
    for j in range(num_p):
        p_logic = p_start_logic + j
        p = n_get_next_head(p_head, p_logic, P_max)

        curr_trans = 1.0

        # Walk backward from the current logical time.
        for step in range(T_max):
            t = n_get_next_head(t_logical_now, -step, T_max)

            # Record the transparency reaching this historical frame.
            cube[t, p, TPChannel.TRANSPARENCY.value] = curr_trans

            # The ray is affected only by frames closer to the observer.
            channel_num = (TPChannel.BODY_REFR.value - TPChannel.BODY_ENERGY.value)
            # Accumulate historical energy after applying current transparency.
            for ch in range(TPChannel.BODY_ENERGY.value, TPChannel.LOW_ENERGY.value + 1):
                # Map each raw-energy channel to its refracted counterpart.
                refr_ch = ch + channel_num
                cube[t_head, p, refr_ch] += cube[t, p, ch] * curr_trans

            # A candle body attenuates energy from earlier frames.
            if cube[t, p, TPChannel.ALL_BODY.value] > 0:
                curr_trans *= (1-cube[t, p, TPChannel.REFRACTION.value])

            # Stop once the remaining contribution is negligible.
            if curr_trans < 1e-4:
                break

        # Aggregate all refracted components.
        cube[t_head, p, TPChannel.TOTAL_REFR_ENERGY.value] = np.sum(
            cube[t_head, p, TPChannel.BODY_REFR.value : TPChannel.LOW_REFR.value+1]
        )

    _n_batch_expansion(cube, t_head, offset, T_max, TPChannel.TOTAL_REFR_ENERGY.value)
    _n_batch_expansion(cube, t_head, offset, T_max, slice(TPChannel.BODY_REFR.value, TPChannel.LOW_REFR.value+1))


@njit(cache=False)
def n_fast_remap(
    old_cube: np.ndarray,
    new_cube: np.ndarray,
    p_head: int,
    old_min: int,
    new_min: int
):
    """
    Realign the price axis when its logical window moves.

    The function copies a physically split circular buffer into a contiguous
    destination while preserving logical price coordinates.
    """
    T, old_cap, C = old_cube.shape
    new_cap = new_cube.shape[1]
    logic_offset = old_min - new_min

    for t in range(T):
        # 1. Copy the segment from ``p_head`` through the physical end.
        len1 = old_cap - p_head
        dst_s1 = logic_offset
        dst_e1 = logic_offset + len1

        # Clip segment 1 to the destination range.
        actual_dst_s1 = max(0, dst_s1)
        actual_dst_e1 = min(new_cap, dst_e1)
        if actual_dst_e1 > actual_dst_s1:
            src_s1 = p_head + (actual_dst_s1 - dst_s1)
            src_e1 = p_head + (actual_dst_e1 - dst_s1)
            new_cube[t, actual_dst_s1:actual_dst_e1, :] = old_cube[t, src_s1:src_e1, :]

        # 2. Copy the wrapped segment before ``p_head``.
        len2 = p_head
        dst_s2 = logic_offset + len1
        dst_e2 = dst_s2 + len2

        # Clip segment 2 to the destination range.
        actual_dst_s2 = max(0, dst_s2)
        actual_dst_e2 = min(new_cap, dst_e2)
        if actual_dst_e2 > actual_dst_s2:
            src_s2 = 0 + (actual_dst_s2 - dst_s2)
            src_e2 = 0 + (actual_dst_e2 - dst_s2)
            new_cube[t, actual_dst_s2:actual_dst_e2, :] = old_cube[t, src_s2:src_e2, :]
