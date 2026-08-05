
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .kernel import TPChannel

try:
    # Prefer the platform-specific AOT extension when one is available.
    from .kernel import kernels_core
except ImportError:
    # Fall back to Numba JIT kernels on unsupported Python/platform combinations.
    from .kernel import kernels as kernels_core

_test = False

@dataclass
class LiquidEngine:
    """
    A dynamic 3D time-price container backed by two circular-buffer pointers.

    Input DataFrames must contain ``open``, ``high``, ``low``, ``close``, and
    ``volume`` columns.

    Core mechanisms:
    1. ``t_head`` overwrites the time dimension in a ring, avoiding array shifts.
    2. ``p_head`` and ``p_min_tick`` map the logical price window onto physical
       storage.
    3. ``_remap_cube`` realigns storage when price moves outside the current
       capacity.

    Price-axis downsampling uses ``bin_size = ticks_per_bin * tick_size``. This
    bounds memory use in volatile markets at the cost of microstructure below
    one bin. Remap and slide operations use integer logical offsets, preserving
    absolute price coordinates as
    ``price = (index + p_min_tick) * bin_size``.

    Because quantization uses ``floor``, floating-point residue near a bin edge
    can move a value by one logical tick.
    """
    tick_size: float = 0.01              # Minimum market price increment.
    max_refraction: float = 0.9          # Upper bound for simulated attenuation.
    fast_t_window: int = 999             # Recent window used for normalization.

    bins_per_interval: int = 5           # Price bins per modeled interval.
    avg_intervals_per_kline: int = 5     # Average intervals per candle.

    cube: Optional[np.ndarray] = None    # Core [time, price, channel] tensor.
    p_min_tick: int = 0                  # Logical price tick at the left edge.
    t_head: int = 0                      # Next physical time position to write.
    p_head: int = 0                      # Physical offset of logical price zero.

    _last_p_min_tick: int = 0            # Minimum tick in the last update.
    _last_p_max_tick: int = 0            # Maximum tick in the last update.

    ticks_per_bin: int = 1               # Raw ticks represented by each bin.
    bin_size: float = 0                  # Real price width of one bin.


    def get_total_energy(self, time_idx: int | tuple):
        """
        Return energy for one logical time point or a logical time range.

        Args:
            time_idx: An integer for one frame or a ``(start, stop)`` tuple for
                a range.

        Returns:
            A ``[price]`` array for an integer, or a ``[time, price]`` array for
            a tuple.

        Circular-buffer mappings on both axes are resolved before returning.
        """
        # Reuse a view of the aggregate energy channel.
        energy_view = self.cube[:, :, TPChannel.TOTAL_REFR_ENERGY]
        if _test:
            print(f"\n--- [GET_ENERGY DEBUG] ---")
            print(f"Current self.t_head: {self.t_head}")
            print(f"Requested time_idx: {time_idx}")

        if isinstance(time_idx, int):
            # Logical index -1 normally addresses the frame just completed.
            t_real = kernels_core.n_get_next_head(self.t_head,time_idx,self.t_capacity)
            energy = energy_view[t_real, :].copy()

            if _test:
                print(f"Target Physical Index (t_real): {t_real}")
                print(f"Raw Max Energy in this frame: {np.max(energy)}")

        elif isinstance(time_idx, tuple):
            start_idx = time_idx[0]
            stop_idx = time_idx[1]
            length = stop_idx - start_idx
            t_start = kernels_core.n_get_next_head(self.t_head, start_idx, self.t_capacity)
            if _test:
                print(f"Slice Length: {length}, Physical Start: {t_start}")

            if t_start + length <= self.t_capacity:
                energy = energy_view[t_start : t_start + length, :].copy()
            else:
                break_point = self.t_capacity - t_start
                part1 = energy_view[t_start:, :]
                part2 = energy_view[: length - break_point, :]
                energy = np.concatenate([part1, part2], axis=0)
            if _test:
                print(f"Slice Max Energy: {np.max(energy)}")
        else:
            raise TypeError("Only int or tuple are supported")

        # Restore logical order on the price axis.
        if self.p_head != 0:
            if _test:
                print(f"Rolling P-axis by: {-self.p_head}")
            energy = np.roll(energy, -self.p_head, axis=-1)

        if _test:
            print(f"Final Output Max Energy: {np.max(energy)}")
            print(f"--- [DEBUG END] ---\n")

        return energy

    def get_p_min_tick(self) -> int:
        """Return the logical price tick at the container's left edge."""
        return self.p_min_tick

    def load_data(self, df: pd.DataFrame, tick_size: float | None = None):
        """Load new OHLCV rows and update the time-price cube."""
        if df is None or len(df) == 0: return

        if _test:
            print(f"\n>>> [LOAD_DATA START] Entry T_HEAD: {self.t_head}, Entry P_HEAD: {self.p_head}")

        if tick_size is not None:
            self.tick_size = tick_size

        if self.cube is None:
            self._calculate_ticks_per_bin(df)

        # Limit a batch to the time capacity to avoid circular-write conflicts.
        if self.cube is not None and len(df) > self.cube.shape[0]:
            df = df.iloc[-self.cube.shape[0]:]

        p_min, p_max = df['low'].min(), df['high'].max()
        self._last_p_min_tick, self._last_p_max_tick = self._price_to_tick(p_min), self._price_to_tick(p_max)

        if _test:
            print("\n" + "="*40)
            print(">>> [LOAD_DATA CORE VALUES]")
            print(f" - Raw price range: {p_min:.2f} to {p_max:.2f}")
            print(f" - Logical tick range: {self._last_p_min_tick} to {self._last_p_max_tick}")
            print(f" - Container start before update: {self.p_min_tick}")

        if self.cube is None:

            self._init_cube(p_min, p_max, len(df))

        else:
            self._ensure_price_range(p_min, p_max)

        if _test:
            print(f" - Container start after update: {self.p_min_tick}")
            print(f" - Container maximum tick: {self.p_max_tick}")
            print(f" - Expected index offset: {self._last_p_max_tick - self.p_min_tick}")
            print("="*40 + "\n")

        if _test:
            print(f"--- [PRICE ALIGN] MinTick: {self.p_min_tick}, P_HEAD after check: {self.p_head}")

        ohlcv_logic = self._to_ohlcv(df)  # Already offset into logical space.

        kernels_core.n_build_cube_anatomy( ohlcv_logic, self.cube, self.t_head, self.p_head, self.fast_t_window, self.max_refraction, self.bins_per_interval, self.avg_intervals_per_kline * self.bins_per_interval)

        p_start = self._last_p_min_tick - 20- self.p_min_tick
        p_end = self._last_p_max_tick + 20- self.p_min_tick
        p_start_safe = max(p_start, 0)
        p_end_safe = min(p_end, self.cube.shape[1])
        # print(f'p_start {p_start} p_end {p_end} p_start_safe {p_start_safe} p_end_safe {p_end_safe} self.cube.shape[1] {self.cube.shape[1]}')
        kernels_core.numba_calc_transparency_map( self.cube, p_start_safe, p_end_safe, self.p_head, self.t_head, len(df))

        self._update_t_head(len(df))

    def _calculate_ticks_per_bin(self, df: pd.DataFrame) -> int:
        """Convert the mean candle range into an adaptive number of ticks per bin."""
        # Measure the range of each candle.
        ranges = df['high'] - df['low']
        kline_len = ranges.mean()

        # Convert the range to ticks, retaining at least one tick per bin.
        avg_ticks_per_kline = kline_len / self.tick_size
        self.ticks_per_bin = max(1, int(avg_ticks_per_kline / self.avg_intervals_per_kline / self.bins_per_interval))
        self.bin_size = self.tick_size * self.ticks_per_bin

    def _update_t_head(self, n: int):
        # Advance the time pointer after a batch write.
        old_head = self.t_head
        self.t_head = kernels_core.n_get_next_head(self.t_head, n, self.cube.shape[0])

    def _price_to_tick(self, price: float) -> int:
        """Convert a real price to its logical bin index."""
        return int(np.floor(price / self.bin_size))

    def _init_cube(self, p_min_raw: float, p_max_raw: float, n_steps: int) -> int:
        """
        Allocate a price range with half of the raw range padded on each side.
        """
        # 1. Find the raw tick boundaries.
        raw_min_tick = int(np.floor(p_min_raw / self.bin_size))
        raw_max_tick = int(np.ceil(p_max_raw / self.bin_size))
        raw_range = raw_max_tick - raw_min_tick

        # 2. Reserve half the raw range above and below.
        padding = raw_range // 2

        # 3. Move the logical origin down by the lower padding.
        self.p_min_tick = raw_min_tick - padding

        # 4. Total width is lower padding + raw range + upper padding.
        # Add one to include both boundary points.
        price_range = raw_range + (2 * padding) + 1
        # Preallocate the [time, price, channel] tensor.
        self.cube = np.zeros((n_steps, price_range, len(TPChannel)), dtype=np.float32)

    def _to_ohlcv(self, df: pd.DataFrame):
        """
        Convert a DataFrame into a tick-aligned float32 index matrix.

        The returned columns are ``open_idx``, ``high_idx``, ``low_idx``,
        ``close_idx``, and ``volume``.
        """

        # Extract price coordinates in one contiguous block.
        ohlc = df[['open', 'high', 'low', 'close']].values

        # ``floor`` provides deterministic bin alignment; float32 matches the cube.
        ohlc_idx = (np.floor(ohlc / self.bin_size) - self.p_min_tick).astype(np.float32)
        v_arr = df['volume'].values.astype(np.float32).reshape(-1, 1)

        # Join logical OHLC indices with volume.
        ohlcv_logic = np.ascontiguousarray(np.hstack((ohlc_idx, v_arr)))

        return ohlcv_logic

    def _remap_cube(self, new_min: int, new_capacity: int):
        """
        Remap data when price moves beyond the current buffer range.

        ``logic_offset`` preserves absolute prices even when ``new_min`` changes
        relative array coordinates.
        """
        if self.cube is None:
            return

        if _test:
            print(f"!!! [REMAP TRIGGERED] Old P_HEAD: {self.p_head}, Old P_MIN: {self.p_min_tick}")
            print(f"!!! [REMAP GOAL] New MinTick: {new_min}, New Capacity: {new_capacity}")

        old_cube = self.cube
        t_dim, old_cap, c_dim = old_cube.shape

        # 1. Allocate zeroed destination storage.
        new_cube = np.zeros((t_dim, new_capacity, c_dim), dtype=np.float32)

        # 2. Remap through the accelerated kernel.
        kernels_core.n_fast_remap(
            old_cube,
            new_cube,
            self.p_head,
            self.p_min_tick,
            new_min
        )

        # 3. Publish the remapped storage and reset its physical origin.
        self.cube = new_cube
        self.p_min_tick = new_min
        self.p_head = 0

        if _test:
            print(f"!!! [REMAP COMPLETE] P_HEAD reset to 0, P_MIN updated to {self.p_min_tick}")


    def _ensure_price_range(self, p_min_raw: float, p_max_raw: float):
        """Ensure that the cube can represent the incoming price range."""
        target_min_tick = int(np.floor(p_min_raw / self.bin_size))
        target_max_tick = int(np.ceil(p_max_raw / self.bin_size))

        # print(f'{target_min_tick} vs {target_max_tick} in {self.bin_size}')

        # Combined logical range.
        total_min = min(target_min_tick, self.p_min_tick)
        total_max = max(target_max_tick, self.p_max_tick)
        new_required_span = total_max - total_min + 1

        # Case A: recenter within the fixed capacity.
        if new_required_span > self.p_capacity:
            # Keep the original capacity rather than growing without a bound.
            new_capacity =self.p_capacity
            # Center the current range in the new logical window.
            current_center = (target_min_tick + target_max_tick) // 2

            # Recalculate the origin around that center.
            new_min = current_center - (new_capacity // 2)
            self._remap_cube(new_min, new_capacity)
            return

        # Case B: slide the price head when the target crosses a boundary.
        if target_min_tick < self.p_min_tick or target_max_tick > self.p_max_tick:
            delta = target_min_tick - self.p_min_tick
            old_p_head = self.p_head
            self.p_head = kernels_core.n_get_next_head(self.p_head, delta, self.p_capacity)
            self.p_min_tick = target_min_tick

            if _test:
                print(f"--- [P_HEAD SLIDE] Delta: {delta}, P_HEAD: {old_p_head} -> {self.p_head}")
        else:
            # Case C: the incoming range already fits.
            if _test:
                print(f"--- [PRICE IN RANGE] No slide needed. MinTick: {self.p_min_tick} ---")

    @property
    def t_capacity(self) -> int:
        return self.cube.shape[0] if self.cube is not None else 0

    @property
    def p_capacity(self) -> int:
        return self.cube.shape[1] if self.cube is not None else 0

    @property
    def p_max_tick(self) -> int:
        """Return the highest logical tick represented by the cube."""
        if self.cube is None:
            return 0
        return self.p_min_tick + self.cube.shape[1] - 1
